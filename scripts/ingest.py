"""
ingest.py
---------
Este módulo carga las rutinas procesadas (JSON) en ChromaDB, la base de datos
vectorial que usa el sistema RAG para recuperar rutinas similares.

Cada semana se almacena como un documento vectorizado. ChromaDB convierte
el texto de cada rutina en un vector numérico (embedding) que representa
su "significado" en un espacio matemático. Cuando el usuario pide una rutina
nueva, el sistema busca los vectores más cercanos al pedido y los usa como
contexto para la generación.

Flujo:
    JSON procesado --> texto enriquecido --> ChromaDB (embedding + metadata) --> listo para RAG
"""

import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Carga las variables de entorno desde .env (necesario para la API key de OpenAI
# que usa ChromaDB internamente para generar los embeddings)
# ---------------------------------------------------------------------------
load_dotenv()

# ===========================================================================
# CONFIGURACIÓN
# Constantes que definen cómo se conecta y organiza la base de datos vectorial
# ===========================================================================

# Carpeta donde ChromaDB persiste los datos en disco entre ejecuciones
# Si no existe, ChromaDB la crea automáticamente
CHROMA_DB_PATH = "./chroma_db"

# Nombre de la colección dentro de ChromaDB donde se guardan las rutinas
# Una colección es similar a una tabla en SQL
COLLECTION_NAME = "rutinas_crossfit"

# Carpeta donde están los JSONs procesados por parse_routine.py
PROCESSED_DATA_PATH = "./data/processed"


# ===========================================================================
# FUNCIONES DE TRANSFORMACIÓN
# Convierten el JSON estructurado en texto enriquecido para vectorizar
# ===========================================================================

def json_a_texto_enriquecido(semana: dict) -> str:
    partes = []
    partes.append(f"Semana: {semana['semana_id']} | Inicio: {semana['fecha_inicio']}")

    for dia in semana["dias"]:
        partes.append(f"\n{dia['dia'].upper()} - {dia.get('tipo_bloque_principal') or 'N/A'}")

        # --- CORE ---
        core = dia.get("core")
        if core and core.get("ejercicios"):
            ejercicios_core = ", ".join(
                f"{e['nombre']} {e.get('reps', '')}".strip()
                for e in core["ejercicios"]
            )
            partes.append(f"Core ({core.get('rondas', '?')} rondas): {ejercicios_core}")

        # --- BLOQUE PRINCIPAL ---
        bp = dia.get("bloque_principal")
        if bp:
            sets_reps = f"{bp['sets']}x{bp['reps']}" if bp.get("sets") and bp.get("reps") else ""
            partes.append(f"{bp.get('tipo', 'N/A')}: {bp.get('movimiento', 'N/A')} {sets_reps}".strip())
            if bp.get("variante_principiante"):
                partes.append(f"Principiante: {bp['variante_principiante']}")

        # --- WOD ---
        wod = dia.get("wod")
        if wod:
            duracion_str = ""
            if wod.get("duracion"):
                duracion_str = f"{wod['duracion']}'"
            elif wod.get("rondas"):
                duracion_str = f"{wod['rondas']} rounds"
            elif wod.get("time_cap"):
                duracion_str = f"TC {wod['time_cap']}'"

            partes.append(f"WOD {wod['formato']} {duracion_str}:".strip())
            for ej in wod.get("ejercicios", []):
                linea = f"  - {ej['nombre']} {ej.get('reps', '')}".strip()
                if ej.get("escala"):
                    linea += f" (escala: {ej['escala']})"
                partes.append(linea)

        # --- ACCESORIOS ---
        acc = dia.get("accesorios")
        if acc and acc.get("ejercicios"):
            ejercicios_acc = ", ".join(
                f"{e['nombre']} {e.get('reps', '')}".strip()
                for e in acc["ejercicios"]
            )
            partes.append(f"Accesorios ({acc.get('rondas', '?')} rondas): {ejercicios_acc}")

        # --- METADATA ---
        meta = dia.get("metadata")
        if meta:
            partes.append(f"Músculos: {', '.join(meta.get('grupos_musculares', []))}")
            if meta.get("movimientos_olimpicos"):
                partes.append(f"Olímpicos: {', '.join(meta['movimientos_olimpicos'])}")
            partes.append(f"Intensidad: {meta.get('intensidad_estimada', 'N/A')}")
            if meta.get("patron_movimiento_fuerza"):
                partes.append(f"Patrón: {meta['patron_movimiento_fuerza']}")

    return "\n".join(partes)



def extraer_metadata_para_chroma(semana: dict) -> dict:
    """
    Extrae los campos de metadata de la semana para almacenarlos como
    filtros en ChromaDB, independientemente del embedding.

    ChromaDB permite guardar metadata junto a cada documento. Esta metadata
    se puede usar para filtrar resultados ANTES de hacer la búsqueda vectorial,
    lo que mejora mucho la precisión del retrieval. Por ejemplo: "dame solo
    semanas que tengan snatch" sin tener que leer todos los embeddings.

    Solo se pueden guardar tipos simples: str, int, float, bool.
    Las listas se convierten a string separado por comas.

    Args:
        semana: Diccionario con la semana completa.

    Returns:
        Diccionario plano con metadata filtrable, compatible con ChromaDB.

    Example:
        >>> meta = extraer_metadata_para_chroma(semana_dict)
        >>> print(meta)
        {
            'semana_id': 'semana_2025_W06',
            'fecha_inicio': '2025-02-03',
            'movimientos_olimpicos': 'clean, snatch',
            'patrones_fuerza': 'sentadilla, empuje, bisagra',
            'total_dias': 5
        }
    """
    # Recopila los movimientos olímpicos de todos los días de la semana
    # usando un set para eliminar duplicados (ej: clean puede aparecer lunes y martes)
    todos_olimpicos = set()
    todos_patrones = set()
    todos_grupos = set()

    for dia in semana["dias"]:
        meta = dia["metadata"]
        # Extiende los sets con los valores de cada día
        todos_olimpicos.update(meta.get("movimientos_olimpicos", []))
        todos_grupos.update(meta.get("grupos_musculares", []))
        if meta.get("patron_movimiento_fuerza"):
            todos_patrones.add(meta["patron_movimiento_fuerza"])

    # Devuelve un diccionario plano con tipos primitivos (requisito de ChromaDB)
    return {
        "semana_id": semana["semana_id"],
        "fecha_inicio": semana["fecha_inicio"],
        # Las listas se convierten a string porque ChromaDB no acepta listas como metadata
        "movimientos_olimpicos": ", ".join(sorted(todos_olimpicos)),
        "patrones_fuerza": ", ".join(sorted(todos_patrones)),
        "grupos_musculares": ", ".join(sorted(todos_grupos)),
        "total_dias": len(semana["dias"]),
    }


# ===========================================================================
# FUNCIONES DE CONEXIÓN Y CARGA EN CHROMADB
# ===========================================================================

def inicializar_chroma() -> chromadb.Collection:
    """
    Inicializa el cliente de ChromaDB y obtiene (o crea) la colección de rutinas.

    ChromaDB puede correr de dos modos:
    - PersistentClient: guarda los datos en disco, los datos sobreviven entre ejecuciones
    - EphemeralClient: solo en memoria, se pierde al cerrar el programa (útil para tests)

    Usamos PersistentClient para que las rutinas cargadas persistan entre sesiones.
    El modelo de embeddings por defecto de ChromaDB (all-MiniLM-L6-v2) corre localmente
    sin necesidad de API keys externas, lo que simplifica el setup inicial.

    Returns:
        Colección de ChromaDB lista para insertar o consultar documentos.
    """
    # Crea el cliente persistente que guarda los datos en la carpeta CHROMA_DB_PATH
    # Si la carpeta no existe, ChromaDB la crea automáticamente
    cliente = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Usa el modelo de embeddings por defecto de ChromaDB
    # all-MiniLM-L6-v2 corre 100% local, es liviano y funciona bien para texto en español
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    # get_or_create_collection obtiene la colección si ya existe, o la crea si no
    # Esto hace que el script sea idempotente: se puede ejecutar varias veces sin error
    coleccion = cliente.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        # cosine es la métrica de similitud más apropiada para texto
        # mide el ángulo entre vectores, no su magnitud
        metadata={"hnsw:space": "cosine"}
    )

    print(f"✅ ChromaDB inicializado en: {CHROMA_DB_PATH}")
    print(f"📚 Colección '{COLLECTION_NAME}': {coleccion.count()} documentos existentes")

    return coleccion


def cargar_semana(coleccion: chromadb.Collection, ruta_json: str) -> bool:
    """
    Carga una semana de rutina desde un archivo JSON a la colección de ChromaDB.

    Si la semana ya existe en la colección (mismo semana_id), la actualiza
    en lugar de duplicarla. Esto hace que el script sea seguro de ejecutar
    múltiples veces con los mismos archivos.

    Args:
        coleccion: Colección de ChromaDB donde se insertará la rutina.
        ruta_json: Ruta al archivo .json procesado por parse_routine.py

    Returns:
        True si la semana fue cargada exitosamente, False si hubo un error.
    """
    # Lee el archivo JSON de la rutina procesada
    ruta = Path(ruta_json)
    if not ruta.exists():
        print(f"⚠️  Archivo no encontrado: {ruta_json}")
        return False

    # Carga el JSON como diccionario Python
    with open(ruta, "r", encoding="utf-8") as f:
        semana = json.load(f)

    # Usa el semana_id como ID único del documento en ChromaDB
    # ChromaDB usa este ID para detectar duplicados y hacer upserts
    doc_id = semana["semana_id"]

    # Convierte el JSON a texto enriquecido para vectorizar
    texto = json_a_texto_enriquecido(semana)

    # Extrae la metadata filtrable para guardar junto al embedding
    metadata = extraer_metadata_para_chroma(semana)

    # upsert inserta el documento si no existe, o lo actualiza si ya existe
    # Esto evita duplicados si se ejecuta el script varias veces
    coleccion.upsert(
        ids=[doc_id],               # ID único del documento
        documents=[texto],           # Texto que se vectorizará con el modelo de embeddings
        metadatas=[metadata]         # Metadata filtrable (no se vectoriza)
    )

    print(f"✅ Cargada: {doc_id} ({len(semana['dias'])} días, {len(texto)} chars)")
    return True


def cargar_todas_las_semanas(coleccion: chromadb.Collection) -> int:
    """
    Carga todos los archivos JSON de la carpeta data/processed/ en ChromaDB.

    Itera sobre todos los archivos .json en la carpeta de datos procesados
    y los carga uno por uno. Los errores individuales no detienen el proceso.

    Args:
        coleccion: Colección de ChromaDB donde se insertarán las rutinas.

    Returns:
        Cantidad de semanas cargadas exitosamente.
    """
    # Busca todos los archivos .json en la carpeta de datos procesados
    ruta_processed = Path(PROCESSED_DATA_PATH)

    # Verifica que la carpeta exista antes de buscar archivos
    if not ruta_processed.exists():
        print(f"❌ La carpeta {PROCESSED_DATA_PATH} no existe. Ejecutá parse_routine.py primero.")
        return 0

    # glob("*.json") devuelve todos los archivos .json de la carpeta
    archivos_json = list(ruta_processed.glob("*.json"))

    if not archivos_json:
        print(f"⚠️  No se encontraron archivos JSON en {PROCESSED_DATA_PATH}")
        return 0

    print(f"\n📂 Encontrados {len(archivos_json)} archivos JSON para cargar")

    # Contador de semanas cargadas exitosamente
    exitosos = 0

    # Itera sobre cada archivo y lo carga en ChromaDB
    for archivo in sorted(archivos_json):
        # sorted() garantiza que se procesen en orden alfabético (semana_01, semana_02, etc.)
        if cargar_semana(coleccion, str(archivo)):
            exitosos += 1

    return exitosos


def verificar_carga(coleccion: chromadb.Collection) -> None:
    """
    Hace una búsqueda de prueba en ChromaDB para verificar que la carga funcionó.

    Realiza una consulta simple para confirmar que los embeddings fueron generados
    correctamente y que el retrieval básico funciona antes de pasar al siguiente paso.

    Args:
        coleccion: Colección de ChromaDB a verificar.

    Returns:
        None. Imprime el resultado de la búsqueda de prueba en consola.
    """
    print("\n🔍 Verificando carga con búsqueda de prueba...")

    # Consulta de ejemplo: busca rutinas con trabajo de piernas y sentadilla
    # Si la carga funcionó bien, debería devolver la semana del ejemplo que tiene Front Squat
    resultados = coleccion.query(
        query_texts=["rutina con sentadilla y trabajo de piernas"],
        n_results=min(2, coleccion.count()),  # Pide 2 resultados o menos si hay menos documentos
        include=["documents", "metadatas", "distances"]
    )

    # Itera sobre los resultados para mostrarlos en consola
    for i, (doc, meta, dist) in enumerate(zip(
        resultados["documents"][0],
        resultados["metadatas"][0],
        resultados["distances"][0]
    )):
        # La distancia coseno va de 0 (idéntico) a 2 (opuesto)
        # La convertimos a similitud para que sea más intuitiva (1 = idéntico, 0 = sin relación)
        similitud = round(1 - dist, 3)
        print(f"\n  Resultado {i+1}: {meta['semana_id']} (similitud: {similitud})")
        # Muestra solo las primeras 3 líneas del documento para no saturar la consola
        primeras_lineas = "\n".join(doc.split("\n")[:3])
        print(f"  Preview: {primeras_lineas}...")


# ===========================================================================
# PUNTO DE ENTRADA
# Permite ejecutar el script directamente desde la terminal:
#   python scripts/ingest.py
# ===========================================================================

if __name__ == "__main__":
    """
    Pipeline completo de ingesta:
        1. Inicializa ChromaDB
        2. Carga todas las semanas procesadas
        3. Verifica que el retrieval funciona
    """

    print("🚀 Iniciando ingesta de rutinas en ChromaDB...\n")

    # Paso 1: Inicializa ChromaDB y obtiene la colección
    coleccion = inicializar_chroma()

    # Paso 2: Carga todos los JSONs procesados en la colección
    total_cargadas = cargar_todas_las_semanas(coleccion)

    # Muestra resumen de la ingesta
    print(f"\n{'='*50}")
    print(f"📊 Ingesta completada: {total_cargadas} semanas cargadas")
    print(f"📚 Total en ChromaDB: {coleccion.count()} documentos")
    print(f"{'='*50}")

    # Paso 3: Verifica que el retrieval funciona con una búsqueda de prueba
    if coleccion.count() > 0:
        verificar_carga(coleccion)
    else:
        print("⚠️  No hay documentos en la colección, saltando verificación")

    print("\n✅ Listo para usar en el pipeline RAG")