"""
parse_routine.py
----------------
Este módulo convierte rutinas de CrossFit en formato PDF
al esquema JSON estructurado que usa el sistema RAG.

Usa pdfplumber para extraer el texto digital del PDF y luego el LLM (Claude)
para interpretar ese texto y estructurarlo, validando el resultado
con Pydantic para garantizar que el JSON siempre tenga la estructura correcta.

Flujo:
    PDF --> pdfplumber --> texto plano --> LLM --> dict Python --> validación Pydantic --> JSON (.json)
"""

import json
from pathlib import Path
from typing import Optional
from groq import Groq
import os
import pdfplumber
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Carga las variables de entorno desde el archivo .env
# Esto permite tener la API key fuera del código fuente
# ---------------------------------------------------------------------------
load_dotenv()


# ===========================================================================
# MODELOS PYDANTIC
# Definen la estructura exacta que debe tener cada parte de la rutina.
# Si el LLM devuelve un campo con tipo incorrecto o falta un campo obligatorio,
# Pydantic lanza un error antes de que el dato corrupto llegue al RAG.
# ===========================================================================

class Ejercicio(BaseModel):
    """Representa un ejercicio individual dentro de cualquier bloque."""

    nombre: str = Field(..., description="Nombre del ejercicio, ej: 'Wall Ball'")
    reps: Optional[str] = Field(None, description="Repeticiones, ej: '12', '10/10', '30s'")
    escala: Optional[str] = Field(None, description="Versión alternativa del ejercicio, ej: 'Knee Raises 15'")


class Core(BaseModel):
    rondas: Optional[int] = Field(None, description="Cantidad de rondas del bloque core")
    ejercicios: list[Ejercicio] = Field(default_factory=list, description="Lista de ejercicios del bloque core")

class BloquePrincipal(BaseModel):
    """
    Bloque de fuerza o trabajo olímpico (OLY).
    Es el bloque de mayor carga del día y define el estímulo principal.
    """
    tipo: str = Field(..., description="Tipo de bloque: 'FUERZA' o 'OLY'")
    descripcion: Optional[str] = Field(None, description="Descripción completa, ej: 'Front Squat 5x4'")
    movimiento: Optional[str] = Field(None, description="Movimiento principal, ej: 'Front Squat'")
    sets: Optional[int] = Field(None, description="Cantidad de series")
    reps: Optional[str] = Field(None, description="Repeticiones por serie")
    variante_principiante: Optional[str] = Field(None, description="Versión para principiante")

class EjercicioWOD(BaseModel):
    """Ejercicio dentro del WOD, puede tener escala de dificultad."""

    nombre: str = Field(..., description="Nombre del ejercicio en el WOD")
    reps: Optional[str] = Field(None, description="Repeticiones o distancia, ej: '20', '200 m'")
    escala: Optional[str] = Field(None, description="Alternativa de menor dificultad para el mismo ejercicio")


class WOD(BaseModel):
    """
    El WOD (Workout of the Day) es el bloque de alta intensidad.
    El formato define cómo se mide el rendimiento: por tiempo, por rondas, etc.
    """

    formato: str = Field(..., description="Formato del WOD: 'AMRAP', 'For Time', 'EMOM', 'Rounds', 'Escalera'")
    duracion: Optional[int] = Field(None, description="Duración en minutos (para AMRAP y time cap)")
    rondas: Optional[int] = Field(None, description="Cantidad de rondas (para formato Rounds fijos)")
    time_cap: Optional[int] = Field(None, description="Tiempo límite en minutos para WODs For Time")
    ejercicios: list[EjercicioWOD] = Field(..., description="Lista de ejercicios que componen el WOD")


class Accesorios(BaseModel):
    """Bloque final del día, enfocado en trabajo complementario y corrección muscular."""

    rondas: Optional[int] = Field(None, description="Cantidad de rondas del bloque accesorio")
    ejercicios: list[Ejercicio] = Field(default_factory=list, description="Lista de ejercicios accesorios")


class Metadata(BaseModel):
    """
    Información adicional generada automáticamente por el LLM.
    Esta metadata es la que usa el RAG para recuperar rutinas similares
    de forma inteligente, sin tener que leer el contenido completo de cada una.
    """

    grupos_musculares: list[str] = Field(
        ...,
        description="Grupos musculares principales trabajados, ej: ['piernas', 'core', 'empuje']"
    )
    movimientos_olimpicos: list[str] = Field(
        default_factory=list,
        description="Movimientos olímpicos presentes, ej: ['clean', 'snatch', 'jerk']"
    )
    intensidad_estimada: str = Field(
        ...,
        description="Nivel de intensidad percibida: 'baja', 'media', 'media-alta', 'alta'"
    )
    patron_movimiento_fuerza: Optional[str] = Field(
        None,
        description="Patrón del bloque de fuerza: 'sentadilla', 'empuje', 'tirón', 'bisagra', o 'OLY'"
    )

class DiaRutina(BaseModel):
    dia: str = Field(..., description="Nombre del día, ej: 'Lunes'")
    fecha: str = Field(..., description="Fecha en formato YYYY-MM-DD")
    tipo_bloque_principal: Optional[str] = Field(None, description="'FUERZA' o 'OLY'")
    core: Core
    bloque_principal: Optional[BloquePrincipal] = Field(None)
    wod: WOD
    accesorios: Optional[Accesorios] = Field(None)  # ← ahora completamente opcional
    metadata: Metadata

class SemanaRutina(BaseModel):
    """
    Representa una semana completa de programación.
    Es la unidad principal que se vectoriza y almacena en ChromaDB.
    """

    semana_id: str = Field(..., description="Identificador único de la semana, ej: 'semana_2025_W06'")
    fecha_inicio: str = Field(..., description="Fecha del lunes de esa semana en formato YYYY-MM-DD")
    dias: list[DiaRutina] = Field(..., description="Lista de los días de entrenamiento de la semana (5 días)")


# ===========================================================================
# PROMPT DEL SISTEMA
# Instrucciones fijas que le dicen al LLM exactamente qué hacer y
# en qué formato devolver la respuesta.
# ===========================================================================

SYSTEM_PROMPT = """
Eres un asistente especializado en programación de CrossFit.
Tu tarea es convertir rutinas escritas en texto libre al formato JSON estructurado que se te indica.

Reglas estrictas:
1. Devolvé ÚNICAMENTE el JSON, sin texto adicional, sin markdown, sin explicaciones.
2. El JSON debe ser válido y parseable directamente con json.loads().
3. Si un campo no está presente en el texto, usá null (no omitas el campo).
4. Para la metadata, inferí los grupos musculares y la intensidad basándote en los ejercicios.
5. Las fechas deben estar en formato YYYY-MM-DD.
6. El semana_id debe tener el formato 'semana_YYYY_WNN' donde NN es el número de semana del año.
7. Para movimientos_olimpicos, incluí solo: 'clean', 'snatch', 'jerk', 'clean_and_jerk', 'deadlift', 'push press' si aparecen.
8. Para patron_movimiento_fuerza usá: 'sentadilla', 'empuje', 'tirón', 'bisagra', o 'OLY'.

El esquema JSON que debés respetar es el siguiente:
{schema}
"""


# ===========================================================================
# FUNCIONES DE EXTRACCIÓN DE TEXTO
# ===========================================================================

def extraer_texto_pdf(ruta_pdf: str) -> str:
    """
    Extrae todo el texto de un PDF digital usando pdfplumber.

    pdfplumber recorre cada página del PDF y extrae el texto manteniendo
    el orden de lectura. Funciona bien con PDFs generados digitalmente
    (Word exportado a PDF, Google Docs, etc.) pero NO con escaneos.

    Args:
        ruta_pdf: Ruta al archivo .pdf con la rutina de la semana.

    Returns:
        String con todo el texto extraído del PDF, páginas separadas
        por un salto de línea doble para preservar la estructura visual.

    Raises:
        FileNotFoundError: Si el archivo PDF no existe en la ruta indicada.
        ValueError: Si el PDF no contiene texto extraíble (ej: es un escaneo).

    Example:
        >>> texto = extraer_texto_pdf("data/raw/semana_01.pdf")
        >>> print(texto[:100])
        'LUNES 2/2 Día 1\\nCORE: 3 Rounds\\n• Dead bug con disco 12...'
    """
    # Verifica que el archivo exista antes de intentar abrirlo
    ruta = Path(ruta_pdf)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo PDF: {ruta_pdf}")

    # Acumula el texto de cada página en esta lista
    paginas_texto = []

    # pdfplumber abre el PDF y lo cierra automáticamente al salir del bloque with
    with pdfplumber.open(ruta_pdf) as pdf:

        # Guarda la cantidad total de páginas para el mensaje final
        total_paginas = len(pdf.pages)

        # Itera sobre cada página (una semana puede ocupar varias páginas)
        for numero_pagina, pagina in enumerate(pdf.pages, start=1):

            # extract_text() devuelve el texto de la página como string, o None si está vacía
            texto_pagina = pagina.extract_text()

            # Solo agrega la página si tiene contenido real
            if texto_pagina and texto_pagina.strip():
                paginas_texto.append(texto_pagina)
            else:
                # Avisa si una página no tiene texto, puede indicar imágenes o decoración
                print(f"⚠️  Página {numero_pagina} sin texto extraíble, puede contener imágenes")

    # Verifica que se haya extraído al menos algo de texto del PDF completo
    if not paginas_texto:
        raise ValueError(
            f"No se pudo extraer texto del PDF: {ruta_pdf}. "
            "Verificá que no sea un PDF escaneado."
        )

    # Une todas las páginas con doble salto de línea para preservar la separación entre días
    texto_completo = "\n\n".join(paginas_texto)

    print(f"📄 PDF procesado: {total_paginas} páginas, {len(texto_completo)} caracteres extraídos")

    return texto_completo


# ===========================================================================
# FUNCIONES DEL PIPELINE RAG
# ===========================================================================

def parsear_rutina_con_llm(texto_rutina: str) -> dict:
    cliente = Groq(api_key=os.getenv("GROQ_API_KEY"))

    schema = json.dumps(SemanaRutina.model_json_schema(), indent=2, ensure_ascii=False)

    respuesta = cliente.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(schema=schema)},
            {"role": "user", "content": f"Convertí esta rutina al formato JSON indicado:\n\n{texto_rutina}"}
        ],
        max_tokens=4096,
        temperature=0.1,
    )

    contenido = respuesta.choices[0].message.content.strip()
    
    # --- DEBUG: muestra los primeros 300 caracteres de la respuesta ---
    print(f"DEBUG respuesta cruda:\n{contenido[:300]}\n")
    # -----------------------------------------------------------------

    # Limpieza más agresiva para distintos formatos de markdown
    import re
    contenido = re.sub(r"^```(?:json)?\s*", "", contenido)
    contenido = re.sub(r"\s*```$", "", contenido)
    contenido = contenido.strip()
    # Limpieza de caracteres de escape inválidos en JSON
    contenido = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', contenido)
    if not contenido:
        raise ValueError("El modelo devolvió una respuesta vacía")

    return json.loads(contenido)

# def parsear_rutina_con_llm(texto_rutina: str) -> dict:
#     """
#     Envía el texto de la rutina al LLM y obtiene el JSON estructurado.

#     Usa el modelo de Anthropic para interpretar el texto libre y extraer
#     cada sección de la rutina en el formato definido por los modelos Pydantic.

#     Args:
#         texto_rutina: Texto plano con la rutina completa de la semana,
#                       extraído del PDF por pdfplumber.

#     Returns:
#         Diccionario Python con la rutina estructurada, listo para validar con Pydantic.

#     Raises:
#         json.JSONDecodeError: Si el LLM devuelve un JSON malformado.
#         ValueError: Si la respuesta del LLM está vacía.
#     """
#     # Inicializa el cliente de Anthropic, toma la API key automáticamente desde .env
#     cliente = Anthropic()

#     # Genera el esquema JSON a partir del modelo Pydantic para incluirlo en el prompt
#     # Esto le muestra al LLM exactamente qué campos y tipos esperamos
#     schema = json.dumps(SemanaRutina.model_json_schema(), indent=2, ensure_ascii=False)

#     # Construye el prompt del sistema reemplazando el placeholder {schema} con el esquema real
#     system_prompt = SYSTEM_PROMPT.format(schema=schema)

#     # Llama a la API de Anthropic con el texto extraído del PDF
#     respuesta = cliente.messages.create(
#         model="claude-opus-4-6",    # Modelo con mejor capacidad de razonamiento estructurado
#         max_tokens=4096,             # Suficiente para una semana completa en JSON
#         system=system_prompt,        # Instrucciones del sistema con el esquema Pydantic
#         messages=[
#             {
#                 "role": "user",
#                 # Le pasamos el texto completo extraído del PDF para que lo estructure
#                 "content": f"Convertí esta rutina al formato JSON indicado:\n\n{texto_rutina}"
#             }
#         ]
#     )

#     # Extrae el texto de la respuesta (content es una lista, tomamos el primer elemento)
#     contenido = respuesta.content[0].text

#     # Verifica que la respuesta no esté vacía antes de intentar parsear
#     if not contenido.strip():
#         raise ValueError("El LLM devolvió una respuesta vacía")

#     # Parsea el string JSON a diccionario Python
#     # json.loads lanza JSONDecodeError con detalles precisos si el formato es inválido
#     return json.loads(contenido)


def validar_rutina(datos: dict) -> SemanaRutina:
    """
    Valida que el diccionario generado por el LLM cumpla con el esquema Pydantic.

    Esta es la red de seguridad del pipeline: garantiza que el JSON que llega
    al RAG siempre tenga la estructura correcta y los tipos de datos esperados.
    Actúa como un contrato entre el LLM y el resto del sistema.

    Args:
        datos: Diccionario Python con la rutina parseada por el LLM.

    Returns:
        Instancia de SemanaRutina validada y tipada correctamente.

    Raises:
        pydantic.ValidationError: Si algún campo tiene tipo incorrecto
                                  o falta un campo obligatorio.
    """
    # Pydantic valida cada campo y sus tipos al instanciar el modelo
    # Si algo no coincide con el esquema, lanza ValidationError con el campo exacto que falló
    return SemanaRutina(**datos)


def guardar_json(semana: SemanaRutina, ruta_salida: str) -> None:
    """
    Guarda la rutina validada como archivo JSON en la carpeta de datos procesados.

    Args:
        semana: Instancia validada de SemanaRutina lista para persistir en disco.
        ruta_salida: Ruta completa donde se guardará el archivo .json resultante.

    Returns:
        None. El efecto secundario es la creación del archivo en disco.
    """
    # Crea automáticamente los directorios intermedios si no existen (ej: data/processed/)
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)

    # Abre el archivo en modo escritura con codificación UTF-8 para soportar tildes y ñ
    with open(ruta_salida, "w", encoding="utf-8") as f:
        # model_dump() convierte el modelo Pydantic a diccionario Python nativo
        # ensure_ascii=False preserva los caracteres especiales en el archivo
        # indent=2 hace el JSON legible para humanos
        json.dump(semana.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"✅ Rutina guardada en: {ruta_salida}")


def parsear_pdf(ruta_pdf: str, ruta_salida: str) -> SemanaRutina:
    """
    Función principal que orquesta el pipeline completo desde PDF hasta JSON.

    Ejecuta los 4 pasos en orden:
        1. Extrae el texto del PDF con pdfplumber
        2. Envía el texto al LLM para estructurarlo como JSON
        3. Valida el JSON resultante con Pydantic
        4. Guarda el JSON validado en disco

    Args:
        ruta_pdf:    Ruta al archivo .pdf con la rutina de la semana.
        ruta_salida: Ruta donde se guardará el JSON procesado resultante.

    Returns:
        Instancia validada de SemanaRutina, útil para uso inmediato
        en tests o para encadenar con el pipeline de ingesta al RAG.

    Example:
        >>> semana = parsear_pdf("data/raw/semana_01.pdf", "data/processed/semana_01.json")
        >>> print(semana.semana_id)
        'semana_2025_W06'
        >>> print(len(semana.dias))
        5
    """
    print(f"\n{'='*50}")
    print(f"📂 Procesando: {ruta_pdf}")
    print(f"{'='*50}")

    # Paso 1: Extrae el texto del PDF con pdfplumber
    texto = extraer_texto_pdf(ruta_pdf)

    print("🤖 Enviando al LLM para estructurar...")
    # Paso 2: El LLM convierte el texto libre al esquema JSON definido
    datos_crudos = parsear_rutina_con_llm(texto)

    print("✔️  Validando estructura con Pydantic...")
    # Paso 3: Pydantic verifica que el JSON tenga todos los campos y tipos correctos
    semana_validada = validar_rutina(datos_crudos)

    # Paso 4: Persiste el JSON validado en disco para uso en el pipeline RAG
    guardar_json(semana_validada, ruta_salida)

    return semana_validada


# ===========================================================================
# PUNTO DE ENTRADA
# Permite ejecutar el script directamente desde la terminal:
#   python scripts/parse_routine.py
# ===========================================================================

if __name__ == "__main__":
    """
    Procesa las 3 semanas de ejemplo en formato PDF.
    Asume que los PDFs están en data/raw/ con el nombre semana_0N.pdf
    y guarda los JSON resultantes en data/processed/semana_0N.json
    """

    # Define los pares (entrada PDF, salida JSON) para cada semana
    semanas = [
        ("data/raw/semana_01.pdf", "data/processed/semana_01.json"),
        ("data/raw/semana_02.pdf", "data/processed/semana_02.json"),
        ("data/raw/semana_03.pdf", "data/processed/semana_03.json"),
    ]

    # Itera sobre cada semana y ejecuta el pipeline completo
    for ruta_pdf, ruta_salida in semanas:
        try:
            # Ejecuta el pipeline completo: PDF → texto → LLM → Pydantic → JSON
            semana = parsear_pdf(ruta_pdf, ruta_salida)
            print(f"🏋️  Semana procesada: {semana.semana_id} ({len(semana.dias)} días)\n")

        except FileNotFoundError as e:
            # El PDF todavía no existe, es normal en cold start, se puede ignorar
            print(f"⚠️  Archivo no encontrado, saltando: {e}\n")

        except Exception as e:
            # Cualquier otro error (LLM, JSON inválido, validación Pydantic) se loguea
            # sin detener el procesamiento del resto de las semanas
            print(f"❌ Error procesando {ruta_pdf}: {e}\n")