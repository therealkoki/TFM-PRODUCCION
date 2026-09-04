"""
app.py — Interfaz principal del agente de la sección 8.4 del TFM.

Conecta: carga_datos (Drive) + router_intencion (clasificación) +
simulacion (análisis de comunicados nuevos) + respuestas (Gemini/plantilla).

Estructura de la página: chat a la izquierda, dashboard de Tableau Public
embebido a la derecha (Nivel 1 de integración, un simple iframe — el usuario
puede filtrar directamente dentro del panel, sin que el agente intervenga).
"""

import streamlit as st

from carga_datos import ACTIVOS_CON_EVIDENCIA, cargar_modelo_sentimiento, cargar_todo
from respuestas import (
    generar_respuesta_consulta_historica,
    generar_respuesta_pregunta_datos,
    generar_respuesta_simulacion,
)
from router_intencion import clasificar_mensaje, detectar_fecha, detectar_ticker
from simulacion import analizar_comunicado_nuevo, consultar_dia_historico

# URL real del dashboard (Historia 1, Tableau Public), con los parámetros
# recomendados por Tableau para incrustar en un iframe (modo compacto, sin
# la barra de navegación de Tableau Public alrededor).
TABLEAU_EMBED_URL = (
    "https://public.tableau.com/views/DashboardTFM-ImpactoComunicacionesenMercados/Historia1"
    "?:language=en-GB&:showVizHome=no&:embed=y"
)


# --------------------------------------------------------------------------
# Lógica de procesamiento de cada mensaje (definida ANTES de usarse en el
# cuerpo principal del script, para evitar NameError en tiempo de ejecución)
# --------------------------------------------------------------------------

def _pedir_ticker() -> str:
    return f"¿Sobre qué activo? Los disponibles son: {', '.join(ACTIVOS_CON_EVIDENCIA)}."


def _pedir_fecha() -> str:
    return "¿Qué fecha? (formato AAAA-MM-DD, por ejemplo 2025-04-09, o \"9 de abril de 2025\")"


def _ejecutar_consulta_historica(ticker: str, fecha: str, datos: dict) -> str:
    if datos.get("modelo_info") is None:
        return ("No puedo consultar ese día ahora mismo: el modelo predictivo "
                "(`modelo_evento_importante.pkl`) no está disponible en Drive todavía.")
    if datos.get("dataset_modelado") is None:
        return ("No puedo consultar ese día ahora mismo: `dataset_modelado.csv` "
                "no está disponible en Drive todavía.")

    try:
        resultado = consultar_dia_historico(
            ticker=ticker,
            fecha=fecha,
            dataset_modelado=datos["dataset_modelado"],
            modelo_info=datos["modelo_info"],
        )
    except ValueError as e:
        return str(e)

    return generar_respuesta_consulta_historica(resultado)


def _ejecutar_simulacion(texto: str, ticker: str, datos: dict) -> str:
    if datos.get("modelo_info") is None:
        return ("No puedo simular ahora mismo: el modelo predictivo (`modelo_evento_importante.pkl`) "
                "no está disponible en Drive todavía.")
    if datos.get("dataset_consolidado_05") is None:
        return ("No puedo simular ahora mismo: `dataset_consolidado_05.csv` "
                "(condiciones de mercado actuales) no está disponible en Drive todavía.")

    with st.spinner("Cargando el modelo de sentimiento (puede tardar unos segundos la primera vez)..."):
        tokenizer, modelo_sentimiento = cargar_modelo_sentimiento()

    try:
        resultado = analizar_comunicado_nuevo(
            texto=texto,
            ticker=ticker,
            dataset_consolidado_05=datos["dataset_consolidado_05"],
            tokenizer=tokenizer,
            modelo=modelo_sentimiento,
            modelo_info=datos["modelo_info"],
            rangos_entrenamiento=datos.get("rangos_entrenamiento", {}),
        )
    except ValueError as e:
        return str(e)

    return generar_respuesta_simulacion(resultado)


def _procesar_mensaje(mensaje_usuario: str, datos: dict, conversacion: dict) -> str:
    pendiente = conversacion["pendiente"]

    # --- Continuación de una simulación a la que le faltaba texto o ticker ---
    if pendiente and pendiente.get("tipo") == "simulacion":
        if pendiente.get("texto_comunicado") is None:
            pendiente["texto_comunicado"] = mensaje_usuario.strip()
        elif pendiente.get("ticker") is None:
            ticker_detectado = detectar_ticker(mensaje_usuario)
            if ticker_detectado is None:
                texto_mayus = mensaje_usuario.strip().upper()
                if texto_mayus in ACTIVOS_CON_EVIDENCIA:
                    ticker_detectado = texto_mayus
            pendiente["ticker"] = ticker_detectado

        if pendiente.get("texto_comunicado") and pendiente.get("ticker"):
            conversacion["pendiente"] = None
            return _ejecutar_simulacion(pendiente["texto_comunicado"], pendiente["ticker"], datos)

        if pendiente.get("ticker") is None:
            conversacion["pendiente"] = pendiente
            return _pedir_ticker()

        conversacion["pendiente"] = pendiente
        return "¿Cuál es el texto del comunicado que quieres analizar?"

    # --- Continuación de una consulta histórica a la que le faltaba ticker o fecha ---
    if pendiente and pendiente.get("tipo") == "consulta_historica":
        if pendiente.get("ticker") is None:
            ticker_detectado = detectar_ticker(mensaje_usuario)
            if ticker_detectado is None:
                texto_mayus = mensaje_usuario.strip().upper()
                if texto_mayus in ACTIVOS_CON_EVIDENCIA:
                    ticker_detectado = texto_mayus
            pendiente["ticker"] = ticker_detectado
        elif pendiente.get("fecha") is None:
            pendiente["fecha"] = detectar_fecha(mensaje_usuario)

        if pendiente.get("ticker") and pendiente.get("fecha"):
            conversacion["pendiente"] = None
            return _ejecutar_consulta_historica(pendiente["ticker"], pendiente["fecha"], datos)

        if pendiente.get("ticker") is None:
            conversacion["pendiente"] = pendiente
            return _pedir_ticker()

        conversacion["pendiente"] = pendiente
        return _pedir_fecha()

    # --- Mensaje nuevo: clasificar desde cero ---
    clasificacion = clasificar_mensaje(mensaje_usuario)

    if clasificacion["tipo"] == "simulacion":
        ticker = clasificacion["ticker"]
        texto = clasificacion["texto_comunicado"]

        if texto and ticker:
            return _ejecutar_simulacion(texto, ticker, datos)

        conversacion["pendiente"] = {
            "tipo": "simulacion",
            "ticker": ticker,
            "texto_comunicado": texto,
        }
        if texto is None:
            return "¿Cuál es el texto del comunicado que quieres analizar?"
        return _pedir_ticker()

    if clasificacion["tipo"] == "consulta_historica":
        ticker = clasificacion["ticker"]
        fecha = clasificacion["fecha"]

        if ticker and fecha:
            return _ejecutar_consulta_historica(ticker, fecha, datos)

        conversacion["pendiente"] = {
            "tipo": "consulta_historica",
            "ticker": ticker,
            "fecha": fecha,
        }
        if ticker is None:
            return _pedir_ticker()
        return _pedir_fecha()

    # --- Pregunta sobre datos ya existentes ---
    return generar_respuesta_pregunta_datos(mensaje_usuario, clasificacion, datos)


# --------------------------------------------------------------------------
# Estado multi-conversación (estilo ChatGPT): varias conversaciones
# independientes, cada una con su propio historial y su propio estado
# "pendiente" (para las simulaciones/consultas a medio completar).
# --------------------------------------------------------------------------

def _crear_conversacion() -> str:
    st.session_state.contador_conversaciones += 1
    id_nueva = f"conv_{st.session_state.contador_conversaciones}"
    st.session_state.conversaciones[id_nueva] = {
        "historial": [],
        "pendiente": None,
        "titulo": "Nueva conversación",
    }
    st.session_state.orden_conversaciones.append(id_nueva)
    return id_nueva


def _inicializar_estado():
    if "conversaciones" not in st.session_state:
        st.session_state.conversaciones = {}
        st.session_state.orden_conversaciones = []
        st.session_state.contador_conversaciones = 0
        id_inicial = _crear_conversacion()
        st.session_state.conversacion_activa = id_inicial


def _conversacion_actual() -> dict:
    return st.session_state.conversaciones[st.session_state.conversacion_activa]


# --------------------------------------------------------------------------
# Cuerpo principal del script (Streamlit ejecuta este fichero de arriba a
# abajo en cada interacción, así que todo lo anterior ya está definido aquí)
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Agente TFM — Impacto de comunicaciones en mercados",
    page_icon=None,
    layout="wide",
)
_inicializar_estado()

# Retoque visual ligero: bordes redondeados en botones y burbujas del chat.
# NOTA: usa clases internas de Streamlit (data-testid), que pueden cambiar
# entre versiones — si un futuro upgrade de Streamlit rompe este estilo, no
# afecta a la funcionalidad del agente, solo a este detalle visual.
st.markdown(
    """
    <style>
    button[kind="primary"], button[kind="secondary"] {
        border-radius: 10px !important;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.5rem 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="padding: 0.5rem 0 1rem 0;">
        <span style="
            background-color: #7C5CFC22;
            color: #A78BFA;
            border: 1px solid #7C5CFC55;
            border-radius: 999px;
            padding: 0.2rem 0.75rem;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        ">TFM · Data Science</span>
        <h1 style="margin: 0.5rem 0 0.25rem 0; font-size: 2rem;">
            Agente de impacto de comunicaciones en mercados financieros
        </h1>
        <p style="color: #9CA3AF; font-size: 0.95rem; margin: 0;">
            Auditor de la evidencia ya generada por el TFM — no un predictor de mercado.
            Pregunta sobre los resultados ya calculados, o pídeme analizar un comunicado nuevo.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

with st.sidebar:
    st.subheader("Conversaciones")
    if st.button("Nueva conversación", use_container_width=True, type="primary"):
        st.session_state.conversacion_activa = _crear_conversacion()
        st.rerun()

    st.divider()

    for id_conv in reversed(st.session_state.orden_conversaciones):
        conv = st.session_state.conversaciones[id_conv]
        es_activa = id_conv == st.session_state.conversacion_activa
        if st.button(
            conv["titulo"], key=f"sel_{id_conv}", use_container_width=True,
            type="primary" if es_activa else "secondary",
        ):
            st.session_state.conversacion_activa = id_conv
            st.rerun()

    st.divider()
    st.caption("TFM — Impacto de comunicaciones públicas en mercados financieros")

with st.spinner("Cargando datos del TFM desde Drive..."):
    datos = cargar_todo()

if datos["errores"]:
    with st.expander("Algunos archivos no se pudieron cargar (el agente seguirá funcionando con lo disponible)"):
        for error in datos["errores"]:
            st.write(f"- {error}")

col_chat, col_dashboard = st.columns([3, 2])

with col_dashboard:
    tab_dashboard, tab_fuentes, tab_metodologia = st.tabs(["Dashboard", "Fuentes de datos", "Metodología"])

    with tab_dashboard:
        st.components.v1.iframe(TABLEAU_EMBED_URL, height=650, scrolling=True)

    with tab_fuentes:
        FUENTES = [
            ("predicciones_hoy.csv", "Predicción diaria por activo (pipeline de producción)"),
            ("dataset_consolidado_05.csv", "Condiciones de mercado en vivo, sin corte de fecha"),
            ("dataset_modelado.csv", "Horizonte de entrenamiento congelado, usado como referencia"),
            ("informe_shap_importancia.csv", "Importancia de variables (capítulo 6, sección 9.1)"),
            ("informe_contribucion_familias.csv", "AUC por familia de variables (sección 9.2)"),
            ("informe_auc_por_activo.csv", "AUC por activo (sección 9.3)"),
            ("informe_comparacion_modelos.csv", "Comparación de modelos baseline (sección 6.1)"),
            ("informe_cv_temporal.csv", "Validación cruzada temporal (sección 7)"),
            ("modelo_evento_importante.pkl", "Modelo LightGBM serializado"),
            ("twitter_roberta_finetuned.zip", "Modelo de sentimiento fine-tuned (capítulo 4)"),
        ]
        tarjetas_html = "".join(
            f"""<div style="
                    border: 1px solid #2D3340; border-radius: 8px;
                    padding: 0.6rem 0.9rem; margin-bottom: 0.5rem;
                    background-color: #161B26;
                ">
                <code style="color: #A78BFA; font-size: 0.85rem;">{nombre}</code>
                <div style="color: #9CA3AF; font-size: 0.8rem; margin-top: 0.15rem;">{descripcion}</div>
            </div>"""
            for nombre, descripcion in FUENTES
        )
        st.markdown(f'<div style="margin-top: 0.5rem;">{tarjetas_html}</div>', unsafe_allow_html=True)

    with tab_metodologia:
        st.markdown(
            "**Cómo funciona este agente**\n\n"
            "Este agente audita la evidencia ya generada por el TFM — no predice el mercado. "
            "El modelo predictivo (LightGBM) se entrenó sobre un horizonte histórico cerrado y "
            "estima la probabilidad de un evento de volatilidad relevante, no la dirección del precio.\n\n"
            "- **Consulta histórica**: reporta hechos ya registrados dentro del horizonte de estudio.\n"
            "- **Simulación**: aplica el modelo, ya entrenado, a condiciones de mercado actuales "
            "combinadas con el sentimiento de un comunicado nuevo.\n\n"
            "En ambos casos, el modelo en sí permanece fijo — solo cambian los datos de entrada."
        )

with col_chat:
    conversacion = _conversacion_actual()

    def _enviar_mensaje(mensaje: str):
        conversacion["historial"].append(("user", mensaje))
        # La primera vez que se manda un mensaje en una conversación nueva, se
        # usa como título en el panel lateral (recortado), igual que hace ChatGPT.
        if conversacion["titulo"] == "Nueva conversación":
            conversacion["titulo"] = mensaje[:40] + ("…" if len(mensaje) > 40 else "")
        with st.spinner("Pensando..."):
            respuesta = _procesar_mensaje(mensaje, datos, conversacion)
        conversacion["historial"].append(("assistant", respuesta))

    for autor, texto in conversacion["historial"]:
        with st.chat_message(autor):
            st.markdown(texto)

    if not conversacion["historial"]:
        st.info("Empieza escribiendo una pregunta abajo, o usa uno de los botones de preguntas rápidas.")

    # Botones de acceso rápido: siempre visibles, no solo al principio, para
    # poder lanzar una pregunta rápida en cualquier punto de la conversación.
    st.divider()
    st.caption("Preguntas rápidas")
    PREGUNTAS_RAPIDAS = [
        ("Predicción de hoy", "¿Qué predice hoy el modelo?"),
        ("Variables importantes", "¿Qué variables pesan más según SHAP?"),
        ("Financieras vs. comunicación", "¿Cuánto pesa el sentimiento frente a las variables financieras?"),
        ("¿Es robusto?", "¿Es un resultado robusto?"),
    ]
    columnas_botones = st.columns(len(PREGUNTAS_RAPIDAS))
    for columna, (etiqueta, pregunta) in zip(columnas_botones, PREGUNTAS_RAPIDAS):
        with columna:
            if st.button(etiqueta, use_container_width=True, key=f"boton_{etiqueta}"):
                _enviar_mensaje(pregunta)
                st.rerun()

    mensaje_usuario = st.chat_input(
        f"Pregunta sobre los resultados, o pide simular un comunicado "
        f"(activos disponibles: {', '.join(ACTIVOS_CON_EVIDENCIA)})"
    )

    if mensaje_usuario:
        _enviar_mensaje(mensaje_usuario)
        st.rerun()

st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #6B7280; font-size: 0.8rem; padding: 1rem 0;">
        TFM — Impacto de comunicaciones públicas en mercados financieros ·
        Sección 8.4: arquitectura de producción
    </div>
    """,
    unsafe_allow_html=True,
)

