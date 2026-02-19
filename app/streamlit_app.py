"""
streamlit_app.py
----------------
Interfaz web del sistema CrossFit RAG.

Layout de dos columnas:
- Izquierda: chat con el agente
- Derecha: rutina renderizada en tiempo real

Para ejecutar:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
import streamlit as st

# Agrega la raíz del proyecto al path para que encuentre la carpeta rag/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.agent import crear_estado_inicial, procesar_mensaje, rutina_a_markdown
# ===========================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ===========================================================================

st.set_page_config(
    page_title="CrossFit Planner",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===========================================================================
# ESTILOS CSS
# ===========================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');

    .stApp {
        background-color: #0f0f0f;
        color: #e8e8e8;
    }

    .header-container {
        background: linear-gradient(135deg, #1a1a1a 0%, #111 100%);
        border-bottom: 2px solid #e63946;
        padding: 1rem 2rem;
        margin-bottom: 0;
    }

    .header-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.8rem;
        letter-spacing: 4px;
        color: #ffffff;
        margin: 0;
        line-height: 1;
    }

    .header-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 3px;
        color: #e63946;
        text-transform: uppercase;
        margin: 0;
    }

    .msg-user {
        background: #1e1e1e;
        border-left: 3px solid #e63946;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
    }

    .msg-assistant {
        background: #161616;
        border-left: 3px solid #444;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #ccc;
    }

    .dia-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-top: 3px solid #e63946;
        border-radius: 4px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .dia-titulo {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.4rem;
        letter-spacing: 2px;
        color: #fff;
        margin: 0 0 0.25rem 0;
    }

    .dia-tipo {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 2px;
        color: #e63946;
        text-transform: uppercase;
    }

    .bloque-titulo {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 2px;
        color: #888;
        text-transform: uppercase;
        margin: 1rem 0 0.4rem 0;
        border-bottom: 1px solid #222;
        padding-bottom: 0.25rem;
    }

    .ejercicio-item {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #ccc;
        padding: 0.2rem 0;
    }

    .ejercicio-reps {
        color: #e63946;
        font-weight: 600;
    }

    .escala-badge {
        font-size: 0.75rem;
        color: #666;
        font-style: italic;
    }

    .principiante-badge {
        background: #1f2a1f;
        border: 1px solid #2d4a2d;
        border-radius: 3px;
        padding: 0.2rem 0.5rem;
        font-size: 0.75rem;
        color: #5a9e5a;
        display: inline-block;
        margin-top: 0.25rem;
    }

    .wod-formato {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.1rem;
        letter-spacing: 1px;
        color: #e63946;
    }

    .estado-badge {
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 1px;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        display: inline-block;
    }

    .estado-sin-rutina {
        background: #1a1a1a;
        color: #666;
        border: 1px solid #333;
    }

    .estado-en-progreso {
        background: #1a1a0a;
        color: #cc9a00;
        border: 1px solid #443300;
    }

    .estado-aprobada {
        background: #0a1a0a;
        color: #5a9e5a;
        border: 1px solid #2d4a2d;
    }

    .stTextInput > div > div > input {
        background: #1a1a1a !important;
        border: 1px solid #333 !important;
        border-radius: 4px !important;
        color: #e8e8e8 !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #e63946 !important;
        box-shadow: 0 0 0 1px #e63946 !important;
    }

    .stButton > button {
        background: #e63946 !important;
        color: white !important;
        border: none !important;
        border-radius: 3px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        font-size: 0.8rem !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.2s !important;
    }

    .stButton > button:hover {
        background: #c1121f !important;
        transform: translateY(-1px) !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #111; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: #e63946; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# INICIALIZACIÓN DEL ESTADO
# ===========================================================================

if "estado" not in st.session_state:
    st.session_state.estado = crear_estado_inicial()

if "mensajes_ui" not in st.session_state:
    st.session_state.mensajes_ui = [
        {
            "role": "assistant",
            "content": (
                "¡Bienvenido al planificador de CrossFit! 💪\n\n"
                "Contame qué tipo de semana querés generar. Por ejemplo:\n"
                "- *'Quiero una semana con énfasis en snatch y WODs cortos'*\n"
                "- *'Generá una semana de intensidad media con trabajo de tren superior'*\n"
                "- *'Necesito una semana variada para atletas intermedios'*"
            )
        }
    ]


# ===========================================================================
# HEADER
# ===========================================================================

st.markdown("""
<div class="header-container">
    <p class="header-subtitle">Sistema de planificación inteligente</p>
    <h1 class="header-title">CrossFit Planner</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ===========================================================================
# LAYOUT: DOS COLUMNAS
# ===========================================================================

col_chat, col_rutina = st.columns([1, 1.4], gap="large")


# ===========================================================================
# COLUMNA IZQUIERDA: CHAT
# ===========================================================================

with col_chat:

    estado = st.session_state.estado
    tiene_rutina = estado["rutina_actual"] is not None

    # Badge de estado
    if estado["aprobada"]:
        badge = '<span class="estado-badge estado-aprobada">✓ RUTINA GUARDADA</span>'
    elif tiene_rutina:
        ediciones = estado["n_ediciones"]
        badge = f'<span class="estado-badge estado-en-progreso">⬤ EN PROGRESO · {ediciones} ediciones</span>'
    else:
        badge = '<span class="estado-badge estado-sin-rutina">○ SIN RUTINA</span>'

    st.markdown(badge, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Historial del chat
    chat_container = st.container(height=480)
    with chat_container:
        for msg in st.session_state.mensajes_ui:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="msg-user">👤 {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="msg-assistant">🤖 {msg["content"]}</div>',
                    unsafe_allow_html=True
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # Input del chat
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            user_input = st.text_input(
                label="mensaje",
                placeholder="Escribí tu pedido o corrección...",
                label_visibility="collapsed"
            )
        with col_btn:
            submitted = st.form_submit_button("Enviar")

    # Botón aprobar (solo visible cuando hay rutina pendiente)
    if tiene_rutina and not estado["aprobada"]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✅ Aprobar y guardar rutina", use_container_width=True):
            with st.spinner("Guardando en la base de conocimiento..."):
                respuesta, nuevo_estado = procesar_mensaje("aprobar", st.session_state.estado)
                st.session_state.estado = nuevo_estado
                st.session_state.mensajes_ui.append({"role": "assistant", "content": respuesta})
            st.rerun()

    # Botón nueva rutina
    if tiene_rutina:
        if st.button("🔄 Nueva rutina", use_container_width=True):
            st.session_state.estado = crear_estado_inicial()
            st.session_state.mensajes_ui = [
                {
                    "role": "assistant",
                    "content": "Empecemos de nuevo. ¿Qué tipo de semana querés generar?"
                }
            ]
            st.rerun()

    # Procesamiento del mensaje
    if submitted and user_input.strip():
        st.session_state.mensajes_ui.append({"role": "user", "content": user_input})

        with st.spinner("Pensando..."):
            respuesta, nuevo_estado = procesar_mensaje(user_input, st.session_state.estado)

        st.session_state.estado = nuevo_estado
        st.session_state.mensajes_ui.append({"role": "assistant", "content": respuesta})
        st.rerun()


# ===========================================================================
# COLUMNA DERECHA: RUTINA
# ===========================================================================

with col_rutina:

    rutina = st.session_state.estado["rutina_actual"]

    if not rutina:
        st.markdown("""
        <div style="
            height: 500px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: #141414;
            border: 1px dashed #2a2a2a;
            border-radius: 4px;
            text-align: center;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🏋️</div>
            <div style="
                font-family: 'Bebas Neue', sans-serif;
                font-size: 1.5rem;
                letter-spacing: 3px;
                color: #333;
            ">TU RUTINA APARECE ACÁ</div>
            <div style="
                font-family: 'Inter', sans-serif;
                font-size: 0.8rem;
                color: #333;
                margin-top: 0.5rem;
            ">Escribí un pedido en el chat para empezar</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Encabezado de la semana
        semana_id = rutina.get("semana_id", "").replace("_", " ").upper()
        fecha_inicio = rutina.get("fecha_inicio", "")

        st.markdown(f"""
        <div style="margin-bottom: 1.5rem;">
            <div style="
                font-family: 'Bebas Neue', sans-serif;
                font-size: 2rem;
                letter-spacing: 4px;
                color: #fff;
                line-height: 1;
            ">{semana_id}</div>
            <div style="
                font-family: 'Inter', sans-serif;
                font-size: 0.75rem;
                color: #666;
                letter-spacing: 1px;
            ">INICIO: {fecha_inicio}</div>
        </div>
        """, unsafe_allow_html=True)

        # Renderiza cada día
        for dia in rutina.get("dias", []):
            nombre_dia = dia.get("dia", "")
            tipo = dia.get("tipo_bloque_principal", "")

            html_dia = f"""
            <div class="dia-card">
                <div class="dia-titulo">{nombre_dia}</div>
                <div class="dia-tipo">{tipo}</div>
            """

            # CORE
            core = dia.get("core")
            if core and core.get("ejercicios"):
                html_dia += f'<div class="bloque-titulo">🔥 Core — {core.get("rondas", "?")} rondas</div>'
                for ej in core.get("ejercicios", []):
                    reps = f'<span class="ejercicio-reps"> {ej["reps"]}</span>' if ej.get("reps") else ""
                    html_dia += f'<div class="ejercicio-item">· {ej["nombre"]}{reps}</div>'

            # BLOQUE PRINCIPAL
            bp = dia.get("bloque_principal")
            if bp:
                tipo_bp = bp.get("tipo", "FUERZA")
                emoji_bp = "🏗️" if tipo_bp == "FUERZA" else "🥇"
                html_dia += f'<div class="bloque-titulo">{emoji_bp} {tipo_bp}</div>'
                html_dia += f'<div class="ejercicio-item">· {bp.get("descripcion", "")}</div>'
                if bp.get("variante_principiante"):
                    html_dia += f'<div class="principiante-badge">🟡 Principiante: {bp["variante_principiante"]}</div>'

            # WOD
            wod = dia.get("wod")
            if wod:
                formato = wod.get("formato", "")
                duracion = wod.get("duracion")
                rondas_wod = wod.get("rondas")
                time_cap = wod.get("time_cap")

                detalle = ""
                if duracion:
                    detalle = f"{duracion}'"
                elif rondas_wod:
                    detalle = f"{rondas_wod} rounds"
                elif time_cap:
                    detalle = f"TC {time_cap}'"

                html_dia += f'<div class="bloque-titulo">⏱️ WOD</div>'
                html_dia += f'<div class="wod-formato">{formato} {detalle}</div>'
                for ej in wod.get("ejercicios", []):
                    reps = f'<span class="ejercicio-reps"> {ej["reps"]}</span>' if ej.get("reps") else ""
                    escala = f'<span class="escala-badge"> (escala: {ej["escala"]})</span>' if ej.get("escala") else ""
                    html_dia += f'<div class="ejercicio-item">· {ej["nombre"]}{reps}{escala}</div>'

            # ACCESORIOS
            acc = dia.get("accesorios")
            if acc and acc.get("ejercicios"):
                html_dia += f'<div class="bloque-titulo">💪 Accesorios — {acc.get("rondas", "?")} rondas</div>'
                for ej in acc.get("ejercicios", []):
                    reps = f'<span class="ejercicio-reps"> {ej["reps"]}</span>' if ej.get("reps") else ""
                    html_dia += f'<div class="ejercicio-item">· {ej["nombre"]}{reps}</div>'

            html_dia += "</div>"
            st.markdown(html_dia, unsafe_allow_html=True)