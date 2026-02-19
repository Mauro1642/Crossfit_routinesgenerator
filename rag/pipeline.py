"""
pipeline.py
-----------
Este módulo se encarga de inicializar ChromaDB automáticamente
cuando la app arranca en producción (Streamlit Cloud).

Verifica si ChromaDB está vacía y si es así carga todos los JSONs
disponibles en data/processed/ sin necesidad de intervención manual.

También expone la función procesar_nuevo_pdf() para que Streamlit
pueda cargar nuevas rutinas desde la interfaz web.

Flujo de inicialización:
    App arranca → ChromaDB vacía? → Sí → cargar JSONs → lista para usar
                                  → No → continuar normalmente
"""

from pathlib import Path

from scripts.ingest import inicializar_chroma, cargar_todas_las_semanas
from scripts.parse_routine import parsear_pdf


# ===========================================================================
# INICIALIZACIÓN AUTOMÁTICA
# ===========================================================================

def inicializar_si_vacia() -> int:
    """
    Verifica si ChromaDB está vacía y la puebla automáticamente si es necesario.

    Se llama una sola vez al arrancar la app. Si ChromaDB ya tiene datos
    (por ejemplo en desarrollo local) no hace nada. Si está vacía (producción
    en Streamlit Cloud) carga todos los JSONs disponibles.

    Returns:
        Cantidad de semanas cargadas. 0 si ChromaDB ya tenía datos.
    """
    # Conecta a ChromaDB
    coleccion = inicializar_chroma()

    # Si ya tiene datos no hace nada
    if coleccion.count() > 0:
        print(f"✅ ChromaDB ya inicializada: {coleccion.count()} rutinas disponibles")
        return 0

    # Si está vacía carga todos los JSONs de data/processed/
    print("⚠️  ChromaDB vacía, cargando rutinas desde data/processed/...")
    total = cargar_todas_las_semanas(coleccion)
    print(f"✅ ChromaDB inicializada con {total} rutinas")
    return total


# ===========================================================================
# CARGA DE NUEVAS RUTINAS DESDE LA INTERFAZ
# ===========================================================================

def procesar_nuevo_pdf(ruta_pdf: str) -> dict:
    """
    Procesa un PDF nuevo subido desde la interfaz web y lo carga en ChromaDB.

    Es el punto de entrada que usa Streamlit cuando el usuario sube
    un PDF desde la interfaz. Ejecuta el pipeline completo:
    PDF → JSON → ChromaDB.

    Args:
        ruta_pdf: Ruta temporal donde Streamlit guardó el PDF subido.

    Returns:
        Diccionario con el resultado de la operación:
        - exito: True si se procesó correctamente
        - semana_id: ID de la semana cargada
        - mensaje: Descripción del resultado
    """
    try:
        # Genera la ruta de salida del JSON en data/processed/
        nombre_pdf = Path(ruta_pdf).stem
        ruta_json = f"data/processed/{nombre_pdf}.json"

        # Paso 1: Parsea el PDF a JSON estructurado
        print(f"📄 Procesando PDF: {ruta_pdf}")
        semana = parsear_pdf(ruta_pdf, ruta_json)

        # Paso 2: Carga el JSON en ChromaDB
        coleccion = inicializar_chroma()
        from scripts.ingest import cargar_semana
        cargar_semana(coleccion, ruta_json)

        return {
            "exito": True,
            "semana_id": semana.semana_id,
            "mensaje": f"Rutina {semana.semana_id} cargada correctamente ({len(semana.dias)} días)"
        }

    except Exception as e:
        return {
            "exito": False,
            "semana_id": None,
            "mensaje": f"Error al procesar el PDF: {str(e)}"
        }