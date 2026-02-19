"""
retriever.py
------------
Este módulo se encarga de recuperar las rutinas más relevantes de ChromaDB
dado un pedido del usuario. Es la "R" del RAG (Retrieval Augmented Generation).

El retriever convierte el pedido del usuario en un vector y busca las rutinas
más similares en ChromaDB. Esas rutinas se usan como contexto para que el LLM
genere una rutina nueva, coherente y variada respecto a las anteriores.

Flujo:
    pedido del usuario --> búsqueda vectorial en ChromaDB --> rutinas similares --> contexto para el LLM
"""

import json
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Carga las variables de entorno desde .env
# ---------------------------------------------------------------------------
load_dotenv()

# ===========================================================================
# CONFIGURACIÓN
# Debe coincidir exactamente con los valores usados en ingest.py
# ===========================================================================

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "rutinas_crossfit"


# ===========================================================================
# CONEXIÓN A CHROMADB
# ===========================================================================

def obtener_coleccion() -> chromadb.Collection:
    """
    Conecta a ChromaDB y devuelve la colección de rutinas.

    Usa PersistentClient para leer los datos que fueron cargados
    previamente por ingest.py. Si la colección no existe o está vacía,
    lanza un error claro antes de intentar hacer búsquedas.

    Returns:
        Colección de ChromaDB con las rutinas vectorizadas.

    Raises:
        ValueError: Si la colección está vacía, indicando que hay que
                    ejecutar ingest.py primero.
    """
    # Conecta al cliente persistente en la misma carpeta que usó ingest.py
    cliente = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Usa la misma función de embeddings que se usó al cargar los datos
    # Es crítico que sea la misma, de lo contrario los vectores son incomparables
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    # Obtiene la colección existente
    coleccion = cliente.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # Verifica que haya datos cargados antes de intentar buscar
    if coleccion.count() == 0:
        raise ValueError(
            "La colección de ChromaDB está vacía. "
            "Ejecutá ingest.py primero para cargar las rutinas."
        )

    return coleccion


# ===========================================================================
# CONSTRUCCIÓN DE LA QUERY DE BÚSQUEDA
# ===========================================================================

def construir_query(
    objetivo: Optional[str] = None,
    evitar_movimientos: Optional[list[str]] = None,
    incluir_movimientos: Optional[list[str]] = None,
    intensidad: Optional[str] = None,
) -> str:
    """
    Construye el texto de búsqueda que se vectorizará para consultar ChromaDB.

    En lugar de buscar con el pedido literal del usuario, construimos una query
    enriquecida con los parámetros relevantes. Esto mejora la calidad del
    retrieval porque el vector de búsqueda es más similar a los vectores
    de las rutinas almacenadas.

    Args:
        objetivo: Foco de la semana, ej: "fuerza en tren inferior", "cardio"
        evitar_movimientos: Movimientos que ya se usaron y no deben repetirse,
                            ej: ["sentadilla", "clean"]
        incluir_movimientos: Movimientos que se quieren incluir,
                             ej: ["snatch", "deadlift"]
        intensidad: Nivel deseado: "baja", "media", "media-alta", "alta"

    Returns:
        String de texto enriquecido listo para vectorizar y buscar en ChromaDB.

    Example:
        >>> query = construir_query(
        ...     objetivo="fuerza en tren superior",
        ...     evitar_movimientos=["sentadilla"],
        ...     incluir_movimientos=["press de banca"],
        ...     intensidad="alta"
        ... )
    """
    # Construye la query como texto descriptivo en lenguaje natural
    # El modelo de embeddings entiende mejor texto natural que listas de palabras clave
    partes = ["rutina de crossfit semanal"]

    # Agrega el objetivo si está definido
    if objetivo:
        partes.append(f"enfocada en {objetivo}")

    # Agrega los movimientos que se quieren incluir
    if incluir_movimientos:
        partes.append(f"con {', '.join(incluir_movimientos)}")

    # Agrega los movimientos que se quieren evitar
    if evitar_movimientos:
        partes.append(f"sin repetir {', '.join(evitar_movimientos)}")

    # Agrega la intensidad deseada
    if intensidad:
        partes.append(f"de intensidad {intensidad}")

    # Une todas las partes en una sola oración descriptiva
    return " ".join(partes)


# ===========================================================================
# FUNCIÓN PRINCIPAL DE RETRIEVAL
# ===========================================================================

def recuperar_rutinas_similares(
    coleccion: chromadb.Collection,
    query: str,
    n_resultados: int = 3,
    filtros: Optional[dict] = None,
) -> list[dict]:
    """
    Busca las rutinas más similares a la query en ChromaDB.

    Convierte la query en un vector y calcula la similitud coseno contra
    todos los documentos de la colección. Devuelve los n_resultados más
    cercanos como contexto para el generador.

    Args:
        coleccion:    Colección de ChromaDB donde buscar.
        query:        Texto descriptivo de la búsqueda, generado por construir_query().
        n_resultados: Cantidad de rutinas a recuperar. Con 3 semanas en la base
                      de datos, recuperamos todas para maximizar el contexto.
        filtros:      Filtros opcionales sobre la metadata, ej:
                      {"movimientos_olimpicos": {"$contains": "snatch"}}

    Returns:
        Lista de diccionarios con las rutinas recuperadas, cada uno con:
        - semana_id: identificador de la semana
        - similitud: score de similitud (0 a 1, donde 1 es idéntico)
        - documento: texto completo de la rutina
        - metadata: metadata filtrable de la semana

    Example:
        >>> rutinas = recuperar_rutinas_similares(coleccion, query, n_resultados=2)
        >>> for r in rutinas:
        ...     print(r['semana_id'], r['similitud'])
    """
    # Limita n_resultados al total de documentos disponibles para evitar error de ChromaDB
    n_resultados = min(n_resultados, coleccion.count())

    # Parámetros base de la consulta
    params = {
        "query_texts": [query],
        "n_results": n_resultados,
        "include": ["documents", "metadatas", "distances"]
    }

    # Agrega filtros de metadata si fueron especificados
    # Los filtros se aplican ANTES de la búsqueda vectorial, reduciendo el espacio de búsqueda
    if filtros:
        params["where"] = filtros

    # Ejecuta la búsqueda vectorial en ChromaDB
    resultados = coleccion.query(**params)

    # Transforma el resultado crudo de ChromaDB en una lista de diccionarios legibles
    rutinas_recuperadas = []

    for doc, meta, dist in zip(
        resultados["documents"][0],
        resultados["metadatas"][0],
        resultados["distances"][0]
    ):
        # Convierte la distancia coseno a similitud (1 = idéntico, 0 = sin relación)
        # ChromaDB devuelve distancia, no similitud, por eso la inversión
        similitud = round(1 - dist, 3)

        rutinas_recuperadas.append({
            "semana_id": meta["semana_id"],
            "similitud": similitud,
            "documento": doc,
            "metadata": meta
        })

    return rutinas_recuperadas


# ===========================================================================
# FUNCIÓN DE FORMATO PARA EL LLM
# ===========================================================================

def formatear_contexto_para_llm(rutinas: list[dict]) -> str:
    """
    Formatea las rutinas recuperadas como contexto legible para el LLM.

    El generador (generator.py) necesita recibir las rutinas de referencia
    en un formato claro para que pueda usarlas como base al crear la nueva rutina.
    Esta función estructura ese contexto de forma que el LLM lo entienda bien.

    Args:
        rutinas: Lista de rutinas recuperadas por recuperar_rutinas_similares().

    Returns:
        String formateado con todas las rutinas de referencia, listo para
        incluir en el prompt del generador.

    Example:
        >>> contexto = formatear_contexto_para_llm(rutinas)
        >>> print(contexto[:200])
        'RUTINAS DE REFERENCIA
         =====================
         [Rutina 1 - semana_2025_W06 (similitud: 0.87)]
         ...'
    """
    if not rutinas:
        return "No se encontraron rutinas de referencia en la base de datos."

    # Encabezado del bloque de contexto
    lineas = [
        "RUTINAS DE REFERENCIA",
        "=" * 50,
        f"Se recuperaron {len(rutinas)} rutinas similares para usar como base.",
        ""
    ]

    # Agrega cada rutina con su identificador y score de similitud
    for i, rutina in enumerate(rutinas, start=1):
        lineas.append(
            f"[Rutina {i} - {rutina['semana_id']} "
            f"(similitud: {rutina['similitud']})]"
        )
        # Agrega la metadata resumida para que el LLM tenga contexto rápido
        meta = rutina["metadata"]
        lineas.append(f"Olímpicos usados: {meta.get('movimientos_olimpicos', 'ninguno')}")
        lineas.append(f"Patrones de fuerza: {meta.get('patrones_fuerza', 'ninguno')}")
        lineas.append(f"Grupos musculares: {meta.get('grupos_musculares', 'no especificado')}")
        lineas.append("")
        # Agrega el contenido completo de la rutina
        lineas.append(rutina["documento"])
        lineas.append("")
        lineas.append("-" * 50)
        lineas.append("")

    return "\n".join(lineas)


# ===========================================================================
# FUNCIÓN PRINCIPAL QUE COMBINA TODO
# ===========================================================================

def recuperar_contexto(
    objetivo: Optional[str] = None,
    evitar_movimientos: Optional[list[str]] = None,
    incluir_movimientos: Optional[list[str]] = None,
    intensidad: Optional[str] = None,
    n_resultados: int = 3,
) -> tuple[str, list[dict]]:
    """
    Función principal del retriever. Orquesta todo el proceso de recuperación.

    Es el punto de entrada que usa generator.py para obtener el contexto.
    Combina la construcción de la query, la búsqueda en ChromaDB y el
    formateo del contexto en una sola llamada.

    Args:
        objetivo:           Foco de la semana nueva, ej: "tren superior"
        evitar_movimientos: Movimientos a no repetir de semanas anteriores
        incluir_movimientos: Movimientos que deben aparecer en la semana
        intensidad:         Nivel de intensidad deseado
        n_resultados:       Cantidad de rutinas a recuperar como contexto

    Returns:
        Tupla con:
        - contexto_formateado: String listo para incluir en el prompt del LLM
        - rutinas_raw: Lista de diccionarios con los datos crudos de las rutinas
                       (útil para análisis o logging)

    Example:
        >>> contexto, rutinas = recuperar_contexto(
        ...     objetivo="fuerza en tren inferior",
        ...     evitar_movimientos=["clean"],
        ...     intensidad="alta"
        ... )
        >>> print(f"Recuperadas {len(rutinas)} rutinas")
        Recuperadas 3 rutinas
    """
    # Paso 1: Conecta a ChromaDB
    coleccion = obtener_coleccion()

    # Paso 2: Construye la query de búsqueda enriquecida
    query = construir_query(
        objetivo=objetivo,
        evitar_movimientos=evitar_movimientos,
        incluir_movimientos=incluir_movimientos,
        intensidad=intensidad
    )

    print(f"🔍 Query de búsqueda: '{query}'")

    # Paso 3: Recupera las rutinas más similares de ChromaDB
    rutinas = recuperar_rutinas_similares(
        coleccion=coleccion,
        query=query,
        n_resultados=n_resultados
    )

    print(f"📚 Rutinas recuperadas: {len(rutinas)}")
    for r in rutinas:
        print(f"   → {r['semana_id']} (similitud: {r['similitud']})")

    # Paso 4: Formatea las rutinas como contexto legible para el LLM
    contexto = formatear_contexto_para_llm(rutinas)

    return contexto, rutinas


# ===========================================================================
# PUNTO DE ENTRADA - PRUEBA DEL RETRIEVER
# ===========================================================================

if __name__ == "__main__":
    """
    Prueba el retriever con distintos escenarios para verificar que
    las búsquedas devuelven resultados coherentes.
    """

    print("🧪 Probando el retriever...\n")

    # Escenario 1: búsqueda general
    print("Escenario 1: búsqueda general")
    print("-" * 40)
    contexto, rutinas = recuperar_contexto(
        objetivo="fuerza en tren inferior con sentadilla"
    )
    print(contexto[:500])
    print()

    # Escenario 2: búsqueda con movimientos específicos
    print("Escenario 2: incluir snatch, evitar clean")
    print("-" * 40)
    contexto, rutinas = recuperar_contexto(
        incluir_movimientos=["snatch"],
        evitar_movimientos=["clean"],
        intensidad="alta"
    )
    print(contexto[:500])
    print()

    print("✅ Retriever funcionando correctamente")