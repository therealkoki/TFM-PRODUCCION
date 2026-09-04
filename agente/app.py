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


def _procesar_mensaje(mensaje_usuario: str, datos: dict) -> str:
    pendiente = st.session_state.pendiente

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
            st.session_state.pendiente = None
            return _ejecutar_simulacion(pendiente["texto_comunicado"], pendiente["ticker"], datos)

        if pendiente.get("ticker") is None:
            st.session_state.pendiente = pendiente
            return _pedir_ticker()

        st.session_state.pendiente = pendiente
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
            st.session_state.pendiente = None
            return _ejecutar_consulta_historica(pendiente["ticker"], pendiente["fecha"], datos)

        if pendiente.get("ticker") is None:
            st.session_state.pendiente = pendiente
            return _pedir_ticker()

        st.session_state.pendiente = pendiente
        return _pedir_fecha()

    # --- Mensaje nuevo: clasificar desde cero ---
    clasificacion = clasificar_mensaje(mensaje_usuario)

    if clasificacion["tipo"] == "simulacion":
        ticker = clasificacion["ticker"]
        texto = clasificacion["texto_comunicado"]

        if texto and ticker:
            return _ejecutar_simulacion(texto, ticker, datos)

        st.session_state.pendiente = {
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

        st.session_state.pendiente = {
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
# Cuerpo principal del script (Streamlit ejecuta este fichero de arriba a
# abajo en cada interacción, así que todo lo anterior ya está definido aquí)
# --------------------------------------------------------------------------

st.set_page_config(page_title="Agente TFM — Impacto de comunicaciones en mercados", layout="wide")

st.title("Agente del TFM: impacto de comunicaciones en mercados financieros")
st.caption(
    "Auditor de la evidencia ya generada por el TFM — no un predictor de mercado. "
    "Pregunta sobre los resultados ya calculados, o pídeme analizar un comunicado nuevo."
)

with st.spinner("Cargando datos del TFM desde Drive..."):
    datos = cargar_todo()

if datos["errores"]:
    with st.expander("⚠ Algunos archivos no se pudieron cargar (el agente seguirá funcionando con lo disponible)"):
        for error in datos["errores"]:
            st.write(f"- {error}")

col_chat, col_dashboard = st.columns([3, 2])

with col_dashboard:
    st.subheader("Dashboard")
    st.components.v1.iframe(TABLEAU_EMBED_URL, height=700, scrolling=True)

with col_chat:
    if "historial" not in st.session_state:
        st.session_state.historial = []
    if "pendiente" not in st.session_state:
        st.session_state.pendiente = None

    def _enviar_mensaje(mensaje: str):
        st.session_state.historial.append(("user", mensaje))
        with st.spinner("Pensando..."):
            respuesta = _procesar_mensaje(mensaje, datos)
        st.session_state.historial.append(("assistant", respuesta))

    for autor, texto in st.session_state.historial:
        with st.chat_message(autor):
            st.markdown(texto)

    # Botones de acceso rápido: siempre visibles, no solo al principio, para
    # poder lanzar una pregunta rápida en cualquier punto de la conversación.
    st.caption("Preguntas rápidas:")
    PREGUNTAS_RAPIDAS = [
        ("📊 Predicción de hoy", "¿Qué predice hoy el modelo?"),
        ("🔍 Variables importantes", "¿Qué variables pesan más según SHAP?"),
        ("⚖️ Financieras vs. comunicación", "¿Cuánto pesa el sentimiento frente a las variables financieras?"),
        ("✅ ¿Es robusto?", "¿Es un resultado robusto?"),
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
