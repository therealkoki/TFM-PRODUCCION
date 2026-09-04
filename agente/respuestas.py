"""
Capa de generación de respuestas en lenguaje natural para el agente de la
sección 8.4 del TFM.

Estrategia: Gemini API (capa gratuita) como generador principal para poder
responder preguntas formuladas libremente, con una plantilla de texto como
reserva automática si Gemini falla (cuota agotada, error de red, secreto no
configurado, etc.) — así el agente nunca se queda sin respuesta en plena
defensa del TFM.

IMPORTANTE: el nombre del modelo de Gemini (GEMINI_MODEL) usa el alias
"gemini-flash-latest", que Google mantiene apuntando siempre al modelo Flash
más reciente de la capa gratuita, para no depender de una versión concreta
que puede quedar obsoleta (Google retira versiones de Gemini con relativa
frecuencia). Aun así, comprobad en Google AI Studio (aistudio.google.com)
que este alias sigue existiendo y sigue siendo gratuito en el momento del
despliegue — la información sobre modelos de Gemini cambia con frecuencia
y no se puede dar por garantizada de antemano.
"""

import os

import pandas as pd

from router_intencion import ACTIVOS_CON_EVIDENCIA, ALIAS_ACTIVOS

GEMINI_MODEL = "gemini-flash-latest"
TIMEOUT_GEMINI_SEGUNDOS = 15

AVISO_ENFOQUE = (
    "Este agente es un auditor de la evidencia ya generada por el TFM, no un predictor de "
    "mercado: el propio análisis estadístico (capítulo 6) demuestra que la relación entre "
    "comunicaciones y mercado es modesta, heterogénea entre activos y depende del modelo usado."
)


# --------------------------------------------------------------------------
# Utilidad: detectar TODOS los activos mencionados en un mensaje (a diferencia
# de router_intencion.detectar_ticker, que solo devuelve el primero — aquí
# hace falta la lista completa para poder comparar activos entre sí).
# --------------------------------------------------------------------------

def extraer_todos_los_tickers(mensaje: str) -> list:
    texto = mensaje.lower()
    encontrados = []
    for ticker in ACTIVOS_CON_EVIDENCIA:
        if ticker.lower() in texto and ticker not in encontrados:
            encontrados.append(ticker)
    for alias, ticker in ALIAS_ACTIVOS.items():
        if alias in texto and ticker not in encontrados:
            encontrados.append(ticker)
    return encontrados


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------

def _llamar_gemini(prompt: str) -> str:
    """Devuelve el texto de Gemini, o lanza una excepción si algo falla (la
    capturamos siempre en el llamador para poder caer al fallback de plantilla).

    Se fija un límite de tiempo explícito (TIMEOUT_GEMINI_SEGUNDOS) porque sin
    él, si Gemini se queda esperando por un problema de red, la app se queda
    "cargando" indefinidamente en vez de caer al fallback de plantilla como
    está diseñado — el timeout convierte una espera indefinida en un fallo
    rápido y controlado.
    """
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("No se encontró la variable de entorno GEMINI_API_KEY.")

    genai.configure(api_key=api_key)
    modelo = genai.GenerativeModel(GEMINI_MODEL)
    respuesta = modelo.generate_content(
        prompt, request_options={"timeout": TIMEOUT_GEMINI_SEGUNDOS}
    )
    texto = (respuesta.text or "").strip()
    if not texto:
        raise RuntimeError("Gemini devolvió una respuesta vacía.")
    return texto


def _prompt_pregunta_datos(mensaje_usuario: str, tema: str, datos_relevantes: str) -> str:
    return f"""Eres el asistente de un TFM de Data Science sobre el impacto de comunicaciones
públicas (Trump, Musk, Fed) en mercados financieros. Tu único papel es explicar, en
español y de forma clara y breve (máximo 3-4 frases), los datos ya calculados que te
paso a continuación. NO inventes cifras que no estén en estos datos. NO te presentes
como un predictor de mercado: el TFM demuestra que la relación es modesta y heterogénea.

Pregunta del usuario: {mensaje_usuario}

Datos relevantes ({tema}):
{datos_relevantes}

Responde directamente a la pregunta usando solo estos datos."""


def _prompt_consulta_historica(resultado: dict) -> str:
    return f"""Eres el asistente de un TFM de Data Science. Un usuario ha preguntado sobre un
día histórico concreto, ya conocido, dentro del horizonte de estudio del TFM. NO es una
predicción — es una consulta de un hecho ya registrado. Redacta la respuesta en español,
clara y breve (máximo 4-5 frases), usando SOLO estos datos, sin inventar nada:

Activo: {resultado['ticker']}
Fecha: {resultado['fecha']}
¿Hubo evento importante real ese día?: {"Sí" if resultado['evento_importante_real'] else "No"}
¿Se detectó anomalía (autoencoder, capítulo 5)?: {"Sí" if resultado['is_anomaly_real'] else "No"}
Retorno real ese día: {resultado['log_return_real']:.4f}
Volatilidad (20d) ese día: {resultado['volatility_20d_real']:.4f}
Sentimiento agregado de las comunicaciones ese día: {resultado['sentiment_real']:.4f} ({resultado['n_comunicaciones_real']} comunicaciones)
Probabilidad que el modelo asigna a este día: {resultado['probabilidad_modelo']:.3f}

Termina SIEMPRE la respuesta incluyendo, tal cual, este aviso: "{resultado['aviso']}\""""


def _prompt_simulacion(resultado: dict) -> str:
    aviso_distribucion_texto = (
        f"\nAviso de fuera de distribución: {resultado['aviso_distribucion']}"
        if resultado.get("aviso_distribucion") else ""
    )
    return f"""Eres el asistente de un TFM de Data Science. Un usuario ha pedido analizar
un comunicado (real o hipotético) sobre el activo {resultado['ticker']}. Ya se ha
calculado todo lo necesario; tu único trabajo es redactar la respuesta en español,
clara y breve (máximo 4-5 frases), usando SOLO estos datos, sin inventar nada:

Texto analizado: {resultado['texto_original']}
Sentimiento detectado: {resultado['sentimiento']['etiqueta']} (prob. positiva={resultado['sentimiento']['prob_positive']:.3f}, prob. negativa={resultado['sentimiento']['prob_negative']:.3f})
Probabilidad de evento importante ANTES (con la comunicación real del último día): {resultado['prediccion_antes']['probabilidad']:.3f}
Probabilidad de evento importante DESPUÉS (con este comunicado): {resultado['prediccion_despues']['probabilidad']:.3f}
Diferencia: {resultado['diferencia_probabilidad']:+.3f}
{aviso_distribucion_texto}
Termina SIEMPRE la respuesta incluyendo, tal cual, este aviso: "{resultado['aviso']}\""""


# --------------------------------------------------------------------------
# Plantillas de reserva (sin dependencias externas, nunca fallan)
# --------------------------------------------------------------------------

def _plantilla_predicciones_hoy(datos: dict, tickers: list) -> str:
    df = datos.get("predicciones_hoy")
    if df is None:
        return ("Todavía no hay predicciones del día disponibles — el pipeline de producción "
                "(punto 7) aún no las ha generado.")
    tickers = tickers or list(df["ticker"].unique()) if "ticker" in df.columns else []
    partes = []
    for ticker in tickers:
        fila = df[df["ticker"] == ticker] if "ticker" in df.columns else pd.DataFrame()
        if fila.empty:
            continue
        fila = fila.iloc[0]
        partes.append(
            f"Para {ticker}, el modelo estima una probabilidad de evento importante de "
            f"{fila.get('probabilidad', float('nan')):.1%}, "
            f"{'por encima' if fila.get('es_evento') else 'por debajo'} del umbral de decisión."
        )
    if not partes:
        return "No encontré la predicción de hoy para ese activo."
    return " ".join(partes) + " " + AVISO_ENFOQUE


def _plantilla_shap_importancia(datos: dict) -> str:
    df = datos["informes"].get("shap_importancia")
    if df is None:
        return "El informe de importancia SHAP todavía no está disponible."
    df = df.copy()
    if "pct_del_total" not in df.columns:
        df["pct_del_total"] = df["importancia_media"] / df["importancia_media"].sum() * 100
    top5 = df.sort_values("importancia_media", ascending=False).head(5)
    lineas = [f"{fila['variable']} ({fila['pct_del_total']:.1f}% del total)" for _, fila in top5.iterrows()]
    return "Las variables con más peso en el modelo, según SHAP, son: " + ", ".join(lineas) + "."


def _plantilla_contribucion_familias(datos: dict) -> str:
    df = datos["informes"].get("contribucion_familias")
    if df is None:
        return "El informe de contribución por familias todavía no está disponible."
    if "n_features" in df.columns:
        lineas = [f"{fila['familia']}: AUC={fila['auc']:.3f} ({fila['n_features']} variables)" for _, fila in df.iterrows()]
    else:
        lineas = [f"{fila['familia']}: AUC={fila['auc']:.3f}" for _, fila in df.iterrows()]
    return "Comparación de AUC según las variables usadas: " + "; ".join(lineas) + "."


def _plantilla_auc_por_activo(datos: dict, tickers: list) -> str:
    df = datos["informes"].get("auc_por_activo")
    if df is None:
        return "El informe de AUC por activo todavía no está disponible."
    tickers = tickers or list(df["ticker"].unique())
    lineas = []
    for ticker in tickers:
        fila = df[df["ticker"] == ticker]
        if fila.empty:
            continue
        fila = fila.iloc[0]
        lineas.append(
            f"{ticker}: solo financieras AUC={fila['auc_financieras']:.3f}, "
            f"solo comunicación AUC={fila['auc_comunicacion']:.3f}, ambas AUC={fila['auc_ambas']:.3f}"
        )
    if not lineas:
        return "No encontré AUC por activo para lo que preguntas."
    return "; ".join(lineas) + ". " + AVISO_ENFOQUE


def _plantilla_comparacion_modelos(datos: dict) -> str:
    df = datos["informes"].get("comparacion_modelos")
    if df is None:
        return "El informe de comparación de modelos todavía no está disponible."
    df = df.sort_values("auc", ascending=False)
    lineas = [f"{fila['modelo']}: AUC={fila['auc']:.3f}, F1={fila['f1']:.3f}" for _, fila in df.iterrows()]
    return "Comparación de modelos baseline (ordenados por AUC): " + "; ".join(lineas) + "."


def _plantilla_cv_temporal(datos: dict) -> str:
    df = datos["informes"].get("cv_temporal")
    if df is None:
        return "El informe de validación cruzada temporal todavía no está disponible."
    df = df.sort_values("auc_medio", ascending=False)
    lineas = [
        f"{fila['modelo']}: AUC medio={fila['auc_medio']:.3f} (±{fila['auc_std']:.3f}, {int(fila['n_splits_validos'])} splits)"
        for _, fila in df.iterrows()
    ]
    return ("Resultados de validación cruzada temporal (5 splits): " + "; ".join(lineas) +
            ". El resultado depende del modelo usado, no es igual de robusto en todos los casos.")


def _plantilla_matriz_confusion(datos: dict) -> str:
    df = datos["informes"].get("matriz_confusion_umbrales")
    if df is None:
        return "El informe de matriz de confusión por umbrales todavía no está disponible."
    lineas = [
        f"umbral {fila['umbral']}: precisión={fila['precision']:.2f}, recall={fila['recall']:.2f} "
        f"(TP={int(fila['TP'])}, FP={int(fila['FP'])}, FN={int(fila['FN'])}, TN={int(fila['TN'])})"
        for _, fila in df.iterrows()
    ]
    return "Matriz de confusión según el umbral de decisión: " + "; ".join(lineas) + "."


def _plantilla_general() -> str:
    return (
        "Puedo ayudarte con datos ya calculados del TFM: la predicción de hoy para un activo, "
        "qué variables pesan más (SHAP), cuánto aporta el sentimiento frente a las variables "
        "financieras, cómo varía el AUC entre activos, qué modelo funciona mejor, o si el "
        "resultado es robusto en el tiempo. También puedo analizar un comunicado nuevo si me "
        "lo pegas (indicando el activo). " + AVISO_ENFOQUE
    )


PLANTILLAS_POR_TEMA = {
    "shap_importancia": lambda datos, tickers: _plantilla_shap_importancia(datos),
    "contribucion_familias": lambda datos, tickers: _plantilla_contribucion_familias(datos),
    "auc_por_activo": lambda datos, tickers: _plantilla_auc_por_activo(datos, tickers),
    "comparacion_modelos": lambda datos, tickers: _plantilla_comparacion_modelos(datos),
    "cv_temporal": lambda datos, tickers: _plantilla_cv_temporal(datos),
    "matriz_confusion_umbrales": lambda datos, tickers: _plantilla_matriz_confusion(datos),
}


def _construir_texto_datos_para_gemini(datos: dict, tema: str, tickers: list) -> str:
    """Reutiliza las plantillas como forma de convertir los DataFrames en texto
    plano legible que se le pasa a Gemini en el prompt (Gemini no lee CSVs)."""
    if tema == "predicciones_hoy":
        return _plantilla_predicciones_hoy(datos, tickers)
    if tema in PLANTILLAS_POR_TEMA:
        return PLANTILLAS_POR_TEMA[tema](datos, tickers)
    return _plantilla_general()


# --------------------------------------------------------------------------
# Puntos de entrada
# --------------------------------------------------------------------------

def generar_respuesta_pregunta_datos(mensaje_usuario: str, clasificacion: dict, datos: dict) -> str:
    tema = clasificacion.get("tema")
    tickers = extraer_todos_los_tickers(mensaje_usuario) or (
        [clasificacion["ticker"]] if clasificacion.get("ticker") else []
    )

    if tema is None and not tickers:
        return _plantilla_general()

    texto_plantilla = _construir_texto_datos_para_gemini(datos, tema, tickers)

    try:
        return _llamar_gemini(_prompt_pregunta_datos(mensaje_usuario, tema or "visión general", texto_plantilla))
    except Exception:
        return texto_plantilla


def generar_respuesta_consulta_historica(resultado: dict) -> str:
    try:
        return _llamar_gemini(_prompt_consulta_historica(resultado))
    except Exception:
        evento_texto = "SÍ hubo" if resultado["evento_importante_real"] else "NO hubo"
        anomalia_texto = "sí se detectó anomalía (capítulo 5)" if resultado["is_anomaly_real"] else "no se detectó anomalía"
        return (
            f"El {resultado['fecha']}, {resultado['ticker']} — {evento_texto} un evento importante real "
            f"ese día; {anomalia_texto}. Retorno: {resultado['log_return_real']:+.2%}, "
            f"volatilidad (20d): {resultado['volatility_20d_real']:.2%}, con "
            f"{resultado['n_comunicaciones_real']} comunicaciones ese día "
            f"(sentimiento agregado: {resultado['sentiment_real']:+.3f}). "
            f"Como referencia, el modelo asigna a este día una probabilidad de "
            f"{resultado['probabilidad_modelo']:.1%}. "
            f"{resultado['aviso']}"
        )


def generar_respuesta_simulacion(resultado: dict) -> str:
    try:
        return _llamar_gemini(_prompt_simulacion(resultado))
    except Exception:
        s = resultado["sentimiento"]
        aviso_distribucion = resultado.get("aviso_distribucion")
        return (
            f"Comunicado analizado sobre {resultado['ticker']}: sentimiento {s['etiqueta']} "
            f"(prob. positiva={s['prob_positive']:.3f}, prob. negativa={s['prob_negative']:.3f}). "
            f"Probabilidad de evento importante antes: {resultado['prediccion_antes']['probabilidad']:.1%}. "
            f"Después de este comunicado: {resultado['prediccion_despues']['probabilidad']:.1%} "
            f"(diferencia: {resultado['diferencia_probabilidad']:+.1%}). "
            f"{resultado['aviso']}"
            + (f" {aviso_distribucion}" if aviso_distribucion else "")
        )
