"""
app.py — Interfaz principal del agente de la sección 8.4 del TFM.

Conecta: carga_datos (Drive) + router_intencion (clasificación) +
simulacion (análisis de comunicados nuevos) + respuestas (Gemini/plantilla)
+ graficos (Plotly).

Estructura de la página: una sola columna de chat, centrada. Cada respuesta
que se apoya en datos concretos lleva su propio gráfico de Plotly pegado
justo debajo del texto, dentro de la misma burbuja del historial — así el
gráfico queda anclado a la pregunta que lo generó y nunca desaparece al
hacer scroll hacia arriba, ni hay que elegir entre varias pestañas fijas.

(Versión anterior: dashboard de Tableau Public embebido en un panel aparte.
Se sustituyó por gráficos nativos porque el dashboard de Tableau nunca
terminaba de encajar visualmente con el resto de la app —fondo blanco propio,
tamaños fijos, barra de herramientas ajena—, y porque un gráfico fijo no
podía adaptarse a la pregunta concreta del usuario. La Historia de Tableau
sigue publicada y disponible como pieza independiente si se quiere mostrar
aparte en la memoria o la defensa.)
"""

import streamlit as st

from carga_datos import ACTIVOS_CON_EVIDENCIA, cargar_modelo_sentimiento, cargar_todo
import informe
from informe import generar_informe_docx, hay_contenido_exportable
from respuestas import (
    generar_respuesta_consulta_historica,
    generar_respuesta_evolucion_precio,
    generar_respuesta_pregunta_datos,
    generar_respuesta_simulacion,
)
from router_intencion import (
    clasificar_mensaje,
    detectar_fecha,
    detectar_rango_fechas_explicito,
    detectar_ticker,
    es_todo_el_historico,
    hay_senal_de_tema_nuevo,
)
from simulacion import analizar_comunicado_nuevo, consultar_dia_historico


# --------------------------------------------------------------------------
# Lógica de procesamiento de cada mensaje (definida ANTES de usarse en el
# cuerpo principal del script, para evitar NameError en tiempo de ejecución).
# Todas las funciones devuelven (texto, grafico_o_none).
# --------------------------------------------------------------------------

def _pedir_ticker():
    return f"¿Sobre qué activo? Los disponibles son: {', '.join(ACTIVOS_CON_EVIDENCIA)}.", None, None


def _pedir_fecha():
    return "¿Qué fecha? (formato AAAA-MM-DD, por ejemplo 2025-04-09, o \"9 de abril de 2025\")", None, None


def _pedir_rango_fechas():
    return (
        "¿Qué periodo quieres ver? Indícalo con las dos fechas completas, por ejemplo "
        "\"desde 2025-01-01 hasta 2025-06-30\" (también vale \"desde el 1 de enero de 2025 "
        "hasta el 30 de junio de 2025\", con el año en las dos fechas) — o escribe "
        "\"todo el histórico\" si prefieres verlo completo."
    ), None, None


def _ejecutar_consulta_historica(ticker: str, fecha: str, datos: dict):
    if datos.get("modelo_info") is None:
        return ("No puedo consultar ese día ahora mismo: el modelo predictivo "
                "(`modelo_evento_importante.pkl`) no está disponible en Drive todavía."), None, None
    if datos.get("dataset_modelado") is None:
        return ("No puedo consultar ese día ahora mismo: `dataset_modelado.csv` "
                "no está disponible en Drive todavía."), None, None

    try:
        resultado = consultar_dia_historico(
            ticker=ticker,
            fecha=fecha,
            dataset_modelado=datos["dataset_modelado"],
            modelo_info=datos["modelo_info"],
        )
    except ValueError as e:
        return str(e), None, None

    texto, grafico = generar_respuesta_consulta_historica(resultado)
    return texto, grafico, "consulta_historica"


def _ejecutar_evolucion_precio(ticker: str, datos: dict, fecha_inicio: str = None, fecha_fin: str = None):
    texto, grafico = generar_respuesta_evolucion_precio(ticker, datos, fecha_inicio, fecha_fin)
    return texto, grafico, "evolucion_precio"


def _ejecutar_simulacion(texto: str, ticker: str, datos: dict):
    if datos.get("modelo_info") is None:
        return ("No puedo simular ahora mismo: el modelo predictivo (`modelo_evento_importante.pkl`) "
                "no está disponible en Drive todavía."), None, None
    if datos.get("dataset_consolidado_05") is None:
        return ("No puedo simular ahora mismo: `dataset_consolidado_05.csv` "
                "(condiciones de mercado actuales) no está disponible en Drive todavía."), None, None

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
        return str(e), None, None

    texto_respuesta, grafico = generar_respuesta_simulacion(resultado)
    return texto_respuesta, grafico, "simulacion"


def _debe_abandonar_pendiente(mensaje_usuario: str, pendiente: dict) -> bool:
    """
    Decide si un mensaje nuevo, llegado mientras el agente esperaba completar
    una simulación o consulta histórica, señala con claridad que el usuario
    ha cambiado de tema — para no quedarse repitiendo la misma pregunta para
    siempre si el usuario simplemente dejó de responder a lo que se le pedía.
    """
    if hay_senal_de_tema_nuevo(mensaje_usuario, pendiente.get("tipo")):
        return True
    # Caso específico: si se está esperando una FECHA en concreto y el mensaje
    # no contiene ni un solo dígito, es prácticamente imposible que sea un
    # intento genuino de responder con una fecha (hasta una fecha mal escrita
    # como "32 de enero" lleva números) — así que se interpreta como abandono.
    if (pendiente.get("tipo") == "consulta_historica"
            and pendiente.get("ticker") is not None
            and pendiente.get("fecha") is None
            and not any(caracter.isdigit() for caracter in mensaje_usuario)):
        return True
    # Mismo razonamiento para cuando se espera el periodo de una evolución de
    # precio: una respuesta genuina siempre lleva dígitos (fechas) o dice
    # "todo el histórico" — si no es ninguna de las dos cosas, se abandona.
    if (pendiente.get("tipo") == "evolucion_precio"
            and pendiente.get("ticker") is not None
            and not es_todo_el_historico(mensaje_usuario)
            and not any(caracter.isdigit() for caracter in mensaje_usuario)):
        return True
    return False


def _procesar_mensaje(mensaje_usuario: str, datos: dict, conversacion: dict):
    pendiente = conversacion["pendiente"]

    if pendiente and _debe_abandonar_pendiente(mensaje_usuario, pendiente):
        conversacion["pendiente"] = None
        pendiente = None

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
        return "¿Cuál es el texto del comunicado que quieres analizar?", None, None

    # --- Continuación de una evolución de precio: falta el activo, o falta
    # confirmar el periodo (siempre se pregunta explícitamente, nunca se
    # adivina de una fecha suelta mencionada de pasada) ---
    if pendiente and pendiente.get("tipo") == "evolucion_precio":
        if pendiente.get("ticker") is None:
            ticker_detectado = detectar_ticker(mensaje_usuario)
            if ticker_detectado is None:
                texto_mayus = mensaje_usuario.strip().upper()
                if texto_mayus in ACTIVOS_CON_EVIDENCIA:
                    ticker_detectado = texto_mayus
            pendiente["ticker"] = ticker_detectado
            if pendiente["ticker"] is None:
                conversacion["pendiente"] = pendiente
                return _pedir_ticker()
            conversacion["pendiente"] = pendiente
            return _pedir_rango_fechas()

        # Ya hay ticker: este mensaje debe responder a "¿qué periodo quieres?"
        if es_todo_el_historico(mensaje_usuario):
            conversacion["pendiente"] = None
            return _ejecutar_evolucion_precio(pendiente["ticker"], datos)

        fecha_inicio, fecha_fin = detectar_rango_fechas_explicito(mensaje_usuario)
        if fecha_inicio and fecha_fin:
            conversacion["pendiente"] = None
            return _ejecutar_evolucion_precio(pendiente["ticker"], datos, fecha_inicio, fecha_fin)

        # No se entendió como rango claro ni como "todo" — se vuelve a pedir,
        # sin adivinar nada, tal como se decidió para evitar ambigüedad.
        conversacion["pendiente"] = pendiente
        return (
            "No he entendido bien el periodo. Indica las dos fechas completas (con el año en "
            "ambas), por ejemplo \"desde 2025-01-01 hasta 2025-06-30\", o escribe "
            "\"todo el histórico\" para verlo completo."
        ), None, None

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

    if clasificacion["tipo"] == "evolucion_precio":
        ticker = clasificacion["ticker"]
        conversacion["pendiente"] = {"tipo": "evolucion_precio", "ticker": ticker}
        if ticker:
            return _pedir_rango_fechas()
        return _pedir_ticker()

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
            return "¿Cuál es el texto del comunicado que quieres analizar?", None, None
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
    texto, grafico = generar_respuesta_pregunta_datos(mensaje_usuario, clasificacion, datos)
    # Categoría para el informe exportable: el propio tema detectado (p. ej.
    # "predicciones_hoy", "shap_importancia"...) si se reconoció uno real;
    # None si fue un saludo, cortesía, pregunta fuera de ámbito o activo no
    # soportado — esos no aportan nada útil que exportar a un informe.
    categoria = clasificacion.get("tema") if not clasificacion.get("activo_no_soportado") else None
    return texto, grafico, categoria


# --------------------------------------------------------------------------
# Estado multi-conversación (estilo ChatGPT): varias conversaciones
# independientes, cada una con su propio historial y su propio estado
# "pendiente" (para las simulaciones/consultas a medio completar). Cada
# entrada del historial es (autor, texto, grafico_o_none).
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
        # Las conversaciones vacías (recién creadas, sin ningún mensaje) no se
        # listan aparte — ya se está viendo la activa, no hace falta un botón
        # duplicado para volver a algo que ya tienes delante.
        if not conv["historial"]:
            continue
        es_activa = id_conv == st.session_state.conversacion_activa
        if st.button(
            conv["titulo"], key=f"sel_{id_conv}", use_container_width=True,
            type="primary" if es_activa else "secondary",
        ):
            st.session_state.conversacion_activa = id_conv
            st.rerun()

    st.divider()

    if st.button("Exportar informe", use_container_width=True):
        conversacion_activa = _conversacion_actual()
        if not hay_contenido_exportable(conversacion_activa):
            st.session_state.informe_generado = None
            st.session_state.informe_error = (
                "Todavía no hay contenido con datos reales en esta conversación para exportar "
                "(saludos, cortesías y preguntas fuera de ámbito no cuentan)."
            )
        else:
            with st.spinner("Generando informe (puede tardar unos segundos)..."):
                try:
                    ruta = generar_informe_docx(conversacion_activa, "/tmp/informe_tfm_agente.docx")
                    with open(ruta, "rb") as f:
                        st.session_state.informe_generado = f.read()
                    st.session_state.informe_error = None
                    st.session_state.informe_diagnostico = list(informe.DIAGNOSTICO)
                except Exception as e:
                    st.session_state.informe_generado = None
                    st.session_state.informe_error = f"No se pudo generar el informe: {e}"
                    st.session_state.informe_diagnostico = list(informe.DIAGNOSTICO)

    if st.session_state.get("informe_error"):
        st.warning(st.session_state.informe_error)

    if st.session_state.get("informe_diagnostico"):
        with st.expander("Diagnóstico técnico del último informe generado"):
            for linea in st.session_state.informe_diagnostico:
                st.write(linea)

    if st.session_state.get("informe_generado"):
        st.download_button(
            "Descargar informe (.docx)",
            data=st.session_state.informe_generado,
            file_name="informe_tfm_agente.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )

    st.divider()
    st.caption("TFM — Impacto de comunicaciones públicas en mercados financieros")

with st.spinner("Cargando datos del TFM desde Drive..."):
    datos = cargar_todo()

if datos["errores"]:
    with st.expander("Algunos archivos no se pudieron cargar (el agente seguirá funcionando con lo disponible)"):
        for error in datos["errores"]:
            st.write(f"- {error}")

# Chat centrado en una columna de ancho cómodo (no a todo el ancho de la
# pantalla en modo "wide", para que se lea como un chat, no como una tabla).
_, col_chat, _ = st.columns([1, 3, 1])

with col_chat:
    tarjeta_chat = st.container(border=True)
    with tarjeta_chat:
        conversacion = _conversacion_actual()

        def _enviar_mensaje(mensaje: str):
            conversacion["historial"].append(("user", mensaje, None, None))
            # La primera vez que se manda un mensaje en una conversación nueva, se
            # usa como título en el panel lateral (recortado), igual que hace ChatGPT.
            if conversacion["titulo"] == "Nueva conversación":
                conversacion["titulo"] = mensaje[:40] + ("…" if len(mensaje) > 40 else "")
            with st.spinner("Pensando..."):
                respuesta, grafico, categoria = _procesar_mensaje(mensaje, datos, conversacion)
            conversacion["historial"].append(("assistant", respuesta, grafico, categoria))

        for autor, texto, grafico, categoria in conversacion["historial"]:
            with st.chat_message(autor):
                # unsafe_allow_html=True: necesario para que los avisos legales
                # se vean como párrafo aparte en gris (ver _aviso_html en
                # respuestas.py). Riesgo aceptado: cada sesión es privada (no
                # hay usuarios compartiendo la misma vista), y los prompts de
                # Gemini ya incluyen una defensa explícita contra instrucciones
                # escondidas que intenten hacerle generar HTML arbitrario.
                st.markdown(texto, unsafe_allow_html=True)
                if grafico is not None:
                    st.plotly_chart(grafico, use_container_width=True, config={"displayModeBar": False})

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
        fila_1, fila_2 = st.columns(2), st.columns(2)
        columnas_botones = fila_1 + fila_2
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

    with st.expander("Fuentes de datos y metodología"):
        st.markdown(
            "**Cómo funciona este agente**\n\n"
            "Este agente audita la evidencia ya generada por el TFM — no predice el mercado. "
            "El modelo predictivo (LightGBM) se entrenó sobre un horizonte histórico cerrado y "
            "estima la probabilidad de un evento de volatilidad relevante, no la dirección del precio.\n\n"
            "- **Consulta histórica**: reporta hechos ya registrados dentro del horizonte de estudio.\n"
            "- **Simulación**: aplica el modelo, ya entrenado, a condiciones de mercado actuales "
            "combinadas con el sentimiento de un comunicado nuevo.\n\n"
            "En ambos casos, el modelo en sí permanece fijo — solo cambian los datos de entrada.\n\n"
            "**Datos que usa este agente**\n\n"
            "- `predicciones_hoy.csv` — predicción diaria por activo (pipeline de producción)\n"
            "- `dataset_consolidado_05.csv` — condiciones de mercado en vivo, sin corte de fecha\n"
            "- `dataset_modelado.csv` — horizonte de entrenamiento congelado, usado como referencia\n"
            "- `informe_shap_importancia.csv` — importancia de variables (capítulo 6, sección 9.1)\n"
            "- `informe_contribucion_familias.csv` — AUC por familia de variables (sección 9.2)\n"
            "- `informe_auc_por_activo.csv` — AUC por activo (sección 9.3)\n"
            "- `informe_comparacion_modelos.csv` — comparación de modelos baseline (sección 6.1)\n"
            "- `informe_cv_temporal.csv` — validación cruzada temporal (sección 7)\n"
            "- `modelo_evento_importante.pkl` — modelo LightGBM serializado\n"
            "- `twitter_roberta_finetuned.zip` — modelo de sentimiento fine-tuned (capítulo 4)"
        )

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
