"""
generator.py
------------
Este módulo es el corazón del sistema RAG. Recibe el contexto recuperado
por el retriever y el historial de conversación, y usa el LLM para:

1. MODO GENERACIÓN: Crear una rutina semanal nueva basada en las rutinas
   de referencia recuperadas de ChromaDB.

2. MODO EDICIÓN: Modificar una rutina existente según las correcciones
   que el usuario pide por chat, sin regenerar todo desde cero.

El generador siempre devuelve dos cosas:
- La rutina completa actualizada en formato JSON
- Un mensaje explicativo en lenguaje natural para mostrar al usuario

Flujo generación:
    contexto RAG + pedido usuario --> LLM --> rutina nueva (JSON) + mensaje

Flujo edición:
    rutina actual + corrección usuario --> LLM --> rutina modificada (JSON) + mensaje
"""

import json
import os
import re
from typing import Optional

from groq import Groq
from dotenv import load_dotenv

from rag.retriever import recuperar_contexto

# ---------------------------------------------------------------------------
# Carga las variables de entorno desde .env
# ---------------------------------------------------------------------------
load_dotenv()

# ===========================================================================
# PROMPTS DEL SISTEMA
# Instrucciones fijas para cada modo del generador
# ===========================================================================

SYSTEM_PROMPT_GENERACION = """
Eres un coach experto en programación de CrossFit con años de experiencia
planificando semanas de entrenamiento para atletas de todos los niveles.

Tu tarea es generar una rutina semanal de CrossFit NUEVA basándote en las
rutinas de referencia que se te proporcionan como contexto.

REGLAS ESTRICTAS:
1. La rutina debe seguir exactamente la misma estructura que las rutinas de referencia:
   CORE → BLOQUE PRINCIPAL (FUERZA u OLY) → WOD → ACCESORIOS
2. Cada día debe tener su variante para principiantes en el bloque principal.
3. Variá los formatos de WOD (AMRAP, For Time, EMOM, Rounds, Escalera) entre días.
4. Alternás días de FUERZA con días OLY de forma similar a las referencias.
5. No repitas los mismos movimientos principales en días consecutivos.
6. Considerá el balance muscular semanal: tren superior, inferior, tirón, empuje.
7. Incorporá las sugerencias de cambio respecto a las rutinas de referencia.

FORMATO DE RESPUESTA:
Devolvé EXACTAMENTE este formato, sin texto adicional antes ni después:

MENSAJE: [Acá escribís un mensaje breve y amigable explicando la rutina generada,
          qué cambios hiciste respecto a las referencias y por qué. 2-3 oraciones.]

JSON:
{json_aqui}

El JSON debe tener esta estructura para cada día:
{
  "semana_id": "semana_YYYY_WNN",
  "fecha_inicio": "YYYY-MM-DD",
  "dias": [
    {
      "dia": "Lunes",
      "fecha": "YYYY-MM-DD",
      "tipo_bloque_principal": "FUERZA",
      "core": {
        "rondas": 3,
        "ejercicios": [{"nombre": "...", "reps": "...", "escala": null}]
      },
      "bloque_principal": {
        "tipo": "FUERZA",
        "descripcion": "...",
        "movimiento": "...",
        "sets": 5,
        "reps": "4",
        "variante_principiante": "..."
      },
      "wod": {
        "formato": "AMRAP",
        "duracion": 14,
        "rondas": null,
        "time_cap": null,
        "ejercicios": [{"nombre": "...", "reps": "...", "escala": null}]
      },
      "accesorios": {
        "rondas": 3,
        "ejercicios": [{"nombre": "...", "reps": "...", "escala": null}]
      },
      "metadata": {
        "grupos_musculares": ["..."],
        "movimientos_olimpicos": [],
        "intensidad_estimada": "alta",
        "patron_movimiento_fuerza": "sentadilla"
      }
    }
  ]
}
"""

SYSTEM_PROMPT_EDICION = """
Eres un coach experto en programación de CrossFit.

Tu tarea es MODIFICAR una rutina semanal existente según la corrección
que pide el usuario. Modificás ÚNICAMENTE lo que el usuario pide,
el resto de la rutina queda exactamente igual.

REGLAS ESTRICTAS:
1. Solo modificás lo que el usuario pide explícitamente.
2. El resto de la rutina se mantiene idéntico.
3. Respetás la estructura y el nivel de detalle de la rutina original.
4. Si el cambio afecta el balance semanal, lo mencionás en el mensaje.

FORMATO DE RESPUESTA:
Devolvé EXACTAMENTE este formato, sin texto adicional antes ni después:

MENSAJE: [Mensaje breve explicando qué cambiaste y por qué. 1-2 oraciones.]

JSON:
{json_aqui}
"""


# ===========================================================================
# FUNCIONES DE PARSING DE RESPUESTA
# ===========================================================================

def parsear_respuesta_llm(respuesta_cruda: str) -> tuple[str, dict]:
    # Extrae el mensaje entre "MENSAJE:" y "JSON:"
    match_mensaje = re.search(r"MENSAJE:\s*(.+?)(?=JSON:)", respuesta_cruda, re.DOTALL)
    if not match_mensaje:
        raise ValueError("No se encontró el campo MENSAJE en la respuesta del LLM")

    mensaje = match_mensaje.group(1).strip()

    # Extrae todo lo que viene después de "JSON:" incluyendo bloques markdown
    match_json = re.search(r"JSON:\s*```(?:json)?\s*(\{.+?\})\s*```", respuesta_cruda, re.DOTALL)

    # Si no encuentra con markdown, intenta sin markdown
    if not match_json:
        match_json = re.search(r"JSON:\s*(\{.+\})", respuesta_cruda, re.DOTALL)

    if not match_json:
        raise ValueError("No se encontró el campo JSON en la respuesta del LLM")

    json_str = match_json.group(1).strip()

    # Limpia caracteres de escape inválidos
    json_str = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', json_str)

    rutina = json.loads(json_str)

    return mensaje, rutina

# ===========================================================================
# CLIENTE GROQ
# ===========================================================================

def obtener_cliente() -> Groq:
    """
    Inicializa y devuelve el cliente de Groq.

    Returns:
        Cliente de Groq configurado con la API key del .env
    """
    # Toma la API key automáticamente desde las variables de entorno
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


# ===========================================================================
# MODO GENERACIÓN
# ===========================================================================

def generar_rutina(
    pedido_usuario: str,
    fecha_inicio: str,
    historial: list[dict],
    n_referencias: int = 3,
) -> tuple[str, dict]:
    """
    Genera una rutina semanal nueva usando el RAG.

    Recupera rutinas similares de ChromaDB y las usa como contexto
    para que el LLM genere una rutina nueva y variada.

    Args:
        pedido_usuario: Descripción libre de lo que quiere el usuario,
                        ej: "quiero una semana con énfasis en snatch y WODs cortos"
        fecha_inicio:   Fecha del lunes de la semana a generar (YYYY-MM-DD)
        historial:      Historial de mensajes previos de la conversación
        n_referencias:  Cantidad de rutinas de referencia a recuperar del RAG

    Returns:
        Tupla con:
        - mensaje: Explicación de la rutina generada para mostrar al usuario
        - rutina:  Diccionario con la rutina completa en formato JSON
    """
    print("🔍 Recuperando rutinas de referencia...")

    # Paso 1: Recupera rutinas similares del RAG basándose en el pedido
    contexto, _ = recuperar_contexto(
        objetivo=pedido_usuario,
        n_resultados=n_referencias
    )

    print("🤖 Generando rutina con el LLM...")

    cliente = obtener_cliente()

    # Construye los mensajes para el LLM incluyendo el historial previo
    mensajes = []

    # Agrega el historial de conversación previo si existe
    # Esto permite que el LLM tenga contexto de pedidos anteriores
    for msg in historial:
        mensajes.append(msg)

    # Agrega el mensaje actual con el contexto del RAG y el pedido
    mensajes.append({
        "role": "user",
        "content": f"""
RUTINAS DE REFERENCIA (usá estas como base para generar la nueva):
{contexto}

PEDIDO DEL USUARIO:
{pedido_usuario}

FECHA DE INICIO DE LA SEMANA: {fecha_inicio}

Generá una rutina semanal completa (5 días: Lunes a Viernes) siguiendo
las instrucciones del sistema. Incorporá variaciones y mejoras respecto
a las rutinas de referencia según el pedido del usuario.
"""
    })

    # Llama al LLM con el sistema de generación
    respuesta = cliente.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_GENERACION},
            *mensajes
        ],
        max_tokens=8000,    # Suficiente para una semana completa detallada
        temperature=0.7,    # Algo de creatividad para generar variaciones
    )

    respuesta_cruda = respuesta.choices[0].message.content

    # --- DEBUG TEMPORAL ---
    print("=" * 60)
    print("RESPUESTA CRUDA DEL LLM:")
    print(respuesta_cruda[:500])
    print("=" * 60)
    # ----------------------

    # Parsea el mensaje y el JSON de la respuesta
    mensaje, rutina = parsear_respuesta_llm(respuesta_cruda)

    return mensaje, rutina


# ===========================================================================
# MODO EDICIÓN
# ===========================================================================

def editar_rutina(
    correccion: str,
    rutina_actual: dict,
    historial: list[dict],
) -> tuple[str, dict]:
    """
    Modifica una rutina existente según la corrección del usuario.

    En lugar de regenerar toda la semana, el LLM recibe la rutina actual
    completa y solo modifica lo que el usuario pide, manteniendo el resto
    exactamente igual.

    Args:
        correccion:    Descripción de lo que quiere cambiar el usuario,
                       ej: "cambiá el WOD del miércoles a For Time con 15 minutos"
        rutina_actual: Diccionario con la rutina completa que se va a modificar
        historial:     Historial de mensajes previos de la conversación

    Returns:
        Tupla con:
        - mensaje: Explicación de los cambios realizados
        - rutina:  Diccionario con la rutina modificada en formato JSON
    """
    print("✏️  Editando rutina según corrección del usuario...")

    cliente = obtener_cliente()

    # Construye los mensajes incluyendo el historial
    mensajes = []

    for msg in historial:
        mensajes.append(msg)

    # El mensaje actual incluye la rutina completa y la corrección pedida
    mensajes.append({
        "role": "user",
        "content": f"""
RUTINA ACTUAL (modificá solo lo que se pide):
{json.dumps(rutina_actual, indent=2, ensure_ascii=False)}

CORRECCIÓN PEDIDA POR EL USUARIO:
{correccion}

Aplicá ÚNICAMENTE la corrección pedida y devolvé la rutina completa modificada.
"""
    })

    # Llama al LLM con el sistema de edición
    respuesta = cliente.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_EDICION},
            *mensajes
        ],
        max_tokens=8000,
        temperature=0.3,    # Temperatura baja para ediciones precisas y conservadoras
    )

    respuesta_cruda = respuesta.choices[0].message.content

    # Parsea el mensaje y el JSON modificado
    mensaje, rutina = parsear_respuesta_llm(respuesta_cruda)

    return mensaje, rutina


# ===========================================================================
# DETECTOR DE INTENCIÓN
# ===========================================================================

def detectar_intencion(mensaje: str, tiene_rutina: bool) -> str:
    """
    Detecta si el mensaje del usuario es un pedido nuevo o una corrección.

    Analiza el texto del mensaje para determinar si el usuario quiere
    generar una rutina nueva o modificar la que ya tiene.

    Args:
        mensaje:      Texto del mensaje del usuario
        tiene_rutina: True si ya hay una rutina generada en la sesión

    Returns:
        "generar"  si el usuario quiere una rutina nueva
        "editar"   si el usuario quiere modificar la rutina actual
        "aprobar"  si el usuario está conforme con la rutina
        "otro"     si el mensaje no es ninguno de los anteriores
    """
    mensaje_lower = mensaje.lower()

    # Palabras clave que indican aprobación de la rutina
    palabras_aprobar = [
        "aprobar", "guardar", "perfecto", "listo", "ok", "está bien",
        "me gusta", "confirmá", "confirmar", "publicar", "usar esta"
    ]

    # Palabras clave que indican una corrección
    palabras_editar = [
        "cambiá", "cambiar", "modificá", "modificar", "reemplazá", "reemplazar",
        "quitá", "quitar", "agregá", "agregar", "ajustá", "ajustar",
        "bajá", "subí", "menos", "más", "en vez de", "en lugar de"
    ]

    # Verifica primero si es una aprobación
    if any(palabra in mensaje_lower for palabra in palabras_aprobar):
        return "aprobar"

    # Si ya hay una rutina y el mensaje tiene palabras de edición, es una corrección
    if tiene_rutina and any(palabra in mensaje_lower for palabra in palabras_editar):
        return "editar"

    # Si no hay rutina todavía, cualquier mensaje es un pedido de generación
    if not tiene_rutina:
        return "generar"

    # Si hay rutina pero no es una corrección clara, asume que es una corrección igual
    if tiene_rutina:
        return "editar"

    return "otro"