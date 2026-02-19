"""
agent.py
--------
Este módulo es el orquestador del sistema. Maneja el estado completo
de la conversación y decide qué hacer con cada mensaje del usuario:

1. Si no hay rutina → llama al generador para crear una nueva
2. Si hay rutina y el usuario pide cambios → llama al generador en modo edición
3. Si el usuario aprueba → guarda la rutina en ChromaDB como ejemplo futuro

El agente mantiene el estado de la sesión en memoria durante la conversación.
Streamlit se encarga de persistir ese estado entre rerenders usando st.session_state.

Flujo completo:
    mensaje usuario
         ↓
    detectar_intencion()
         ↓
    ┌────────────────────────────────────┐
    │ "generar" → generar_rutina() + RAG │
    │ "editar"  → editar_rutina()        │
    │ "aprobar" → guardar_en_chromadb()  │
    └────────────────────────────────────┘
         ↓
    actualizar estado de sesión
         ↓
    devolver respuesta a Streamlit
"""

import json
from datetime import datetime, timedelta
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from rag.generator import generar_rutina, editar_rutina, detectar_intencion
from scripts.ingest import json_a_texto_enriquecido, extraer_metadata_para_chroma

# ---------------------------------------------------------------------------
load_dotenv()
# ---------------------------------------------------------------------------

# ===========================================================================
# CONFIGURACIÓN
# ===========================================================================

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "rutinas_crossfit"


# ===========================================================================
# ESTADO DE SESIÓN
# Representa el estado completo de una conversación con el agente
# ===========================================================================

def crear_estado_inicial() -> dict:
    """
    Crea un estado de sesión vacío para una conversación nueva.

    El estado se almacena en st.session_state en Streamlit y persiste
    durante toda la conversación del usuario.

    Returns:
        Diccionario con el estado inicial vacío de la sesión.
    """
    return {
        # Historial de mensajes en formato OpenAI/Groq
        # [{"role": "user"/"assistant", "content": "..."}]
        "historial": [],

        # Rutina actual en construcción (None hasta que se genera la primera)
        "rutina_actual": None,

        # True cuando el usuario aprueba y se guarda en ChromaDB
        "aprobada": False,

        # Fecha de inicio de la semana a generar (lunes próximo por defecto)
        "fecha_inicio": obtener_proximo_lunes(),

        # Cantidad de ediciones realizadas en esta sesión
        "n_ediciones": 0,
    }


def obtener_proximo_lunes() -> str:
    """
    Calcula la fecha del próximo lunes en formato YYYY-MM-DD.

    Se usa como fecha de inicio por defecto para la semana a generar.
    Si hoy es lunes, devuelve el lunes de la semana que viene.

    Returns:
        String con la fecha del próximo lunes en formato YYYY-MM-DD.
    """
    hoy = datetime.today()
    # weekday() devuelve 0 para lunes, 6 para domingo
    # Calculamos cuántos días faltan para el próximo lunes
    dias_hasta_lunes = (7 - hoy.weekday()) % 7
    # Si hoy es lunes (dias_hasta_lunes == 0), vamos al lunes siguiente
    if dias_hasta_lunes == 0:
        dias_hasta_lunes = 7
    proximo_lunes = hoy + timedelta(days=dias_hasta_lunes)
    return proximo_lunes.strftime("%Y-%m-%d")


# ===========================================================================
# GUARDADO EN CHROMADB
# ===========================================================================

def guardar_en_chromadb(rutina: dict) -> bool:
    """
    Guarda una rutina aprobada en ChromaDB como ejemplo futuro.

    Una vez que el usuario aprueba la rutina, esta se vectoriza y se
    agrega a la base de conocimiento. Las próximas rutinas generadas
    podrán usar esta como referencia, mejorando el sistema con el tiempo.

    Args:
        rutina: Diccionario con la rutina completa en formato JSON.

    Returns:
        True si se guardó correctamente, False si hubo un error.
    """
    try:
        # Conecta a ChromaDB
        cliente = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        coleccion = cliente.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        # Genera el ID único para esta rutina
        doc_id = rutina["semana_id"]

        # Convierte la rutina a texto enriquecido para vectorizar
        texto = json_a_texto_enriquecido(rutina)

        # Extrae la metadata filtrable
        metadata = extraer_metadata_para_chroma(rutina)

        # Guarda en ChromaDB (upsert para evitar duplicados)
        coleccion.upsert(
            ids=[doc_id],
            documents=[texto],
            metadatas=[metadata]
        )

        print(f"✅ Rutina guardada en ChromaDB: {doc_id}")
        print(f"📚 Total de rutinas en la base: {coleccion.count()}")
        return True

    except Exception as e:
        print(f"❌ Error al guardar en ChromaDB: {e}")
        return False


# ===========================================================================
# FUNCIÓN PRINCIPAL DEL AGENTE
# ===========================================================================

def procesar_mensaje(mensaje: str, estado: dict) -> tuple[str, dict]:
    """
    Procesa un mensaje del usuario y actualiza el estado de la sesión.

    Es el punto de entrada principal que llama Streamlit en cada mensaje.
    Detecta la intención del usuario y ejecuta la acción correspondiente.

    Args:
        mensaje: Texto del mensaje del usuario.
        estado:  Estado actual de la sesión (historial + rutina actual).

    Returns:
        Tupla con:
        - respuesta: String con la respuesta del agente para mostrar al usuario
        - estado:    Estado actualizado de la sesión
    """
    tiene_rutina = estado["rutina_actual"] is not None

    # Detecta qué quiere hacer el usuario
    intencion = detectar_intencion(mensaje, tiene_rutina)

    print(f"🧠 Intención detectada: {intencion}")

    # -----------------------------------------------------------------------
    # CASO 1: GENERAR RUTINA NUEVA
    # -----------------------------------------------------------------------
    if intencion == "generar":
        try:
            respuesta_llm, rutina = generar_rutina(
                pedido_usuario=mensaje,
                fecha_inicio=estado["fecha_inicio"],
                historial=estado["historial"],
            )

            # Actualiza el estado con la rutina generada
            estado["rutina_actual"] = rutina
            estado["aprobada"] = False
            estado["n_ediciones"] = 0

            # Agrega el intercambio al historial
            estado["historial"].append({"role": "user", "content": mensaje})
            estado["historial"].append({"role": "assistant", "content": respuesta_llm})

            # Construye la respuesta completa para el usuario
            respuesta = (
                f"{respuesta_llm}\n\n"
                f"📋 La rutina está lista. Podés pedirme cambios o escribir "
                f"**'aprobar'** cuando estés conforme para guardarla."
            )

        except Exception as e:
            respuesta = f"❌ Hubo un error al generar la rutina: {str(e)}"
            print(f"Error en generación: {e}")

    # -----------------------------------------------------------------------
    # CASO 2: EDITAR RUTINA EXISTENTE
    # -----------------------------------------------------------------------
    elif intencion == "editar":
        if not tiene_rutina:
            # No debería llegar acá, pero por las dudas
            respuesta = "Todavía no generé ninguna rutina. Contame qué tipo de semana querés."
        else:
            try:
                respuesta_llm, rutina_modificada = editar_rutina(
                    correccion=mensaje,
                    rutina_actual=estado["rutina_actual"],
                    historial=estado["historial"],
                )

                # Actualiza el estado con la rutina modificada
                estado["rutina_actual"] = rutina_modificada
                estado["n_ediciones"] += 1

                # Agrega el intercambio al historial
                estado["historial"].append({"role": "user", "content": mensaje})
                estado["historial"].append({"role": "assistant", "content": respuesta_llm})

                respuesta = (
                    f"{respuesta_llm}\n\n"
                    f"✏️ Edición #{estado['n_ediciones']} aplicada. "
                    f"Podés seguir pidiendo cambios o escribir **'aprobar'** para guardar."
                )

            except Exception as e:
                respuesta = f"❌ Hubo un error al editar la rutina: {str(e)}"
                print(f"Error en edición: {e}")

    # -----------------------------------------------------------------------
    # CASO 3: APROBAR Y GUARDAR
    # -----------------------------------------------------------------------
    elif intencion == "aprobar":
        if not tiene_rutina:
            respuesta = "No hay ninguna rutina para aprobar. Contame qué tipo de semana querés."
        elif estado["aprobada"]:
            respuesta = "Esta rutina ya fue guardada anteriormente."
        else:
            guardado = guardar_en_chromadb(estado["rutina_actual"])

            if guardado:
                estado["aprobada"] = True
                semana_id = estado["rutina_actual"].get("semana_id", "")

                # Agrega el intercambio al historial
                estado["historial"].append({"role": "user", "content": mensaje})
                estado["historial"].append({
                    "role": "assistant",
                    "content": f"Rutina {semana_id} guardada como ejemplo futuro."
                })

                respuesta = (
                    f"✅ **Rutina guardada correctamente** en la base de conocimiento.\n\n"
                    f"A partir de ahora esta semana va a servir como referencia "
                    f"para generar futuras rutinas. "
                    f"Si querés generar otra semana, contame qué necesitás."
                )
            else:
                respuesta = "❌ Hubo un error al guardar la rutina. Intentá de nuevo."

    # -----------------------------------------------------------------------
    # CASO 4: MENSAJE NO RECONOCIDO
    # -----------------------------------------------------------------------
    else:
        if not tiene_rutina:
            respuesta = (
                "¡Hola! Soy tu asistente de programación CrossFit. "
                "Contame qué tipo de semana querés generar. Por ejemplo:\n\n"
                "- *'Quiero una semana con énfasis en snatch y WODs cortos'*\n"
                "- *'Generá una semana de intensidad media con trabajo de tren superior'*\n"
                "- *'Necesito una semana variada para atletas intermedios'*"
            )
        else:
            respuesta = (
                "No entendí bien tu pedido. Podés:\n"
                "- Pedirme un cambio específico en la rutina\n"
                "- Escribir **'aprobar'** para guardar la rutina actual\n"
                "- Pedirme una rutina completamente nueva"
            )

    return respuesta, estado


# ===========================================================================
# FUNCIÓN AUXILIAR: RENDERIZAR RUTINA
# ===========================================================================

def rutina_a_markdown(rutina: dict) -> str:
    """
    Convierte la rutina en formato JSON a texto Markdown legible.

    Streamlit puede renderizar este Markdown directamente, mostrando
    la rutina de forma clara y bien formateada en la interfaz.

    Args:
        rutina: Diccionario con la rutina completa en formato JSON.

    Returns:
        String en formato Markdown con la rutina completa formateada.
    """
    if not rutina:
        return "_No hay rutina generada todavía._"

    lineas = []
    lineas.append(f"# 🏋️ {rutina.get('semana_id', 'Rutina').replace('_', ' ').title()}")
    lineas.append(f"**Inicio:** {rutina.get('fecha_inicio', '')}")
    lineas.append("")

    for dia in rutina.get("dias", []):
        # Encabezado del día
        lineas.append(f"---")
        lineas.append(f"## 📅 {dia.get('dia', '')} — {dia.get('tipo_bloque_principal', '')}")
        lineas.append("")

        # CORE
        core = dia.get("core")
        if core:
            lineas.append(f"**🔥 CORE** — {core.get('rondas', '?')} rondas")
            for ej in core.get("ejercicios", []):
                reps = f" {ej['reps']}" if ej.get("reps") else ""
                lineas.append(f"- {ej['nombre']}{reps}")
            lineas.append("")

        # BLOQUE PRINCIPAL
        bp = dia.get("bloque_principal")
        if bp:
            tipo = bp.get("tipo", "FUERZA")
            emoji = "🏗️" if tipo == "FUERZA" else "🥇"
            lineas.append(f"**{emoji} {tipo}**")
            lineas.append(f"- {bp.get('descripcion', '')}")
            if bp.get("variante_principiante"):
                lineas.append(f"- 🟡 Principiante: {bp['variante_principiante']}")
            lineas.append("")

        # WOD
        wod = dia.get("wod")
        if wod:
            formato = wod.get("formato", "")
            duracion = wod.get("duracion")
            rondas = wod.get("rondas")
            time_cap = wod.get("time_cap")

            detalle = ""
            if duracion:
                detalle = f"{duracion}'"
            elif rondas:
                detalle = f"{rondas} rounds"
            elif time_cap:
                detalle = f"TC {time_cap}'"

            lineas.append(f"**⏱️ WOD — {formato} {detalle}**".strip())
            for ej in wod.get("ejercicios", []):
                reps = f" {ej['reps']}" if ej.get("reps") else ""
                escala = f" *(escala: {ej['escala']})*" if ej.get("escala") else ""
                lineas.append(f"- {ej['nombre']}{reps}{escala}")
            lineas.append("")

        # ACCESORIOS
        acc = dia.get("accesorios")
        if acc:
            lineas.append(f"**💪 ACCESORIOS** — {acc.get('rondas', '?')} rondas")
            for ej in acc.get("ejercicios", []):
                reps = f" {ej['reps']}" if ej.get("reps") else ""
                lineas.append(f"- {ej['nombre']}{reps}")
            lineas.append("")

    return "\n".join(lineas)