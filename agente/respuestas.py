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
    return f"""Eres un analista de datos senior explicando, a un compañero de equipo, resultados ya
calculados de un TFM de Data Science sobre el impacto de comunicaciones públicas (Trump, Musk, Fed)
en mercados financieros. Escribe en español, con una redacción natural y fluida — como lo explicarías
en una conversación real, no como una lista de cifras leídas en voz alta.

Reglas estrictas:
- Usa SOLO los datos que te paso abajo. No inventes ni un solo número que no esté ahí.
- No listes cada elemento con la misma fórmula repetida (evita "Para X, el modelo estima... Para Y, el
  modelo estima..."). En vez de eso, agrupa, compara, señala qué destaca (el más alto, el más bajo, un
  patrón, algo sorprendente o esperable) — como haría un analista real, no una plantilla.
- Sé breve: 2-4 frases como máximo, salvo que los datos tengan varias partes claramente distintas que
  merezcan una frase cada una.
- Formato: usa Markdown. Pon en **negrita** las cifras y nombres clave. Si comparas 3 o más elementos,
  usa una lista con guiones en vez de una frase larga con comas. Empieza con un encabezado corto en
  negrita que resuma el tema (p. ej. "**Predicción de hoy**").
- No te presentes como un predictor de mercado: el TFM demuestra que la relación entre comunicaciones
  y mercado es modesta y heterogénea — encájalo con naturalidad si el tema lo pide, sin forzarlo.

Pregunta del usuario: {mensaje_usuario}

Datos reales disponibles ({tema}):
{datos_relevantes}

Responde con una explicación analítica y natural, no con una recitación de los datos."""


def _prompt_consulta_historica(resultado: dict) -> str:
    return f"""Eres un analista de datos senior explicando, a un compañero de equipo, un día concreto
ya conocido dentro del horizonte de estudio de un TFM de Data Science. NO es una predicción — es la
consulta de un hecho ya registrado. Escribe en español, con una redacción natural y fluida, no como
una lista de cifras.

Reglas estrictas:
- Usa SOLO los datos que te paso abajo, sin inventar nada.
- Cuenta primero el hecho (hubo o no evento, hubo o no anomalía) y solo después, con matiz, la
  probabilidad del modelo — nunca al revés.
- Formato: usa Markdown. Un encabezado corto en negrita, y una lista con guiones para las cifras
  clave (retorno, volatilidad, sentimiento, probabilidad del modelo).
- Máximo 4-5 frases antes del aviso final obligatorio.
- Termina SIEMPRE incluyendo, tal cual y sin parafrasear, este aviso: "{resultado['aviso']}"

Activo: {resultado['ticker']}
Fecha: {resultado['fecha']}
¿Hubo evento importante real ese día?: {"Sí" if resultado['evento_importante_real'] else "No"}
¿Se detectó anomalía (autoencoder, capítulo 5)?: {"Sí" if resultado['is_anomaly_real'] else "No"}
Retorno real ese día: {resultado['log_return_real']:.4f}
Volatilidad (20d) ese día: {resultado['volatility_20d_real']:.4f}
Sentimiento agregado de las comunicaciones ese día: {resultado['sentiment_real']:.4f} ({resultado['n_comunicaciones_real']} comunicaciones)
Probabilidad que el modelo asigna a este día: {resultado['probabilidad_modelo']:.3f}

Redacta la respuesta ahora."""


def _prompt_simulacion(resultado: dict) -> str:
    aviso_distribucion_texto = (
        f"\nAviso de fuera de distribución: {resultado['aviso_distribucion']}"
        if resultado.get("aviso_distribucion") else ""
    )
    return f"""Eres un analista de datos senior explicando, a un compañero de equipo, el resultado de
una simulación de sensibilidad de un modelo de un TFM de Data Science. Escribe en español, con una
redacción natural y fluida, como en una conversación real — no como una lista de cifras.

Reglas estrictas:
- Usa SOLO los datos que te paso abajo, sin inventar nada.
- No te limites a repetir los números: señala si el cambio es grande o pequeño en su contexto (recuerda
  que la comunicación pesa poco frente a las variables financieras en este modelo), y qué dirección tomó.
- Formato: usa Markdown. Un encabezado corto en negrita con el activo, y una lista con guiones para
  sentimiento y probabilidad antes/después.
- Máximo 3-4 frases antes del aviso final obligatorio.
- Termina SIEMPRE incluyendo, tal cual y sin parafrasear, este aviso: "{resultado['aviso']}"

Activo: {resultado['ticker']}
Texto analizado: {resultado['texto_original']}
Sentimiento detectado: {resultado['sentimiento']['etiqueta']} (prob. positiva={resultado['sentimiento']['prob_positive']:.3f}, prob. negativa={resultado['sentimiento']['prob_negative']:.3f})
Probabilidad de evento importante ANTES (con la comunicación real del último día): {resultado['prediccion_antes']['probabilidad']:.3f}
Probabilidad de evento importante DESPUÉS (con este comunicado): {resultado['prediccion_despues']['probabilidad']:.3f}
Diferencia: {resultado['diferencia_probabilidad']:+.3f}
{aviso_distribucion_texto}
Redacta la respuesta ahora."""


# --------------------------------------------------------------------------
# Plantillas de reserva (sin dependencias externas, nunca fallan)
# --------------------------------------------------------------------------

def _plantilla_predicciones_hoy(datos: dict, tickers: list) -> str:
    df = datos.get("predicciones_hoy")
    if df is None:
        return ("Todavía no hay predicciones del día disponibles — el pipeline de producción "
                "(punto 7) aún no las ha generado.")
    tickers = tickers or list(df["ticker"].unique()) if "ticker" in df.columns else []
    filas = []
    for ticker in tickers:
        fila = df[df["ticker"] == ticker] if "ticker" in df.columns else pd.DataFrame()
        if not fila.empty:
            filas.append(fila.iloc[0])
    if not filas:
        return "No encontré la predicción de hoy para ese activo."

    if len(filas) == 1:
        fila = filas[0]
        estado = "**por encima**" if fila.get("es_evento") else "por debajo"
        return (
            f"**Predicción de hoy para {fila['ticker']}**\n\n"
            f"Probabilidad de evento importante: **{fila['probabilidad']:.1%}** "
            f"({estado} del umbral de decisión).\n\n{AVISO_ENFOQUE}"
        )

    ordenadas = sorted(filas, key=lambda f: f["probabilidad"], reverse=True)
    mas_alta, mas_baja = ordenadas[0], ordenadas[-1]
    activos_con_evento = [f["ticker"] for f in filas if f.get("es_evento")]
    calificativo = "bajas en general" if mas_alta["probabilidad"] < 0.3 else "dispares"

    lineas = [f"- **{f['ticker']}**: {f['probabilidad']:.1%}" + (" ⚠️ *supera el umbral*" if f.get("es_evento") else "")
              for f in ordenadas]

    conclusion = (
        "Ninguna supera el umbral de decisión, así que no se prevé ningún evento hoy."
        if not activos_con_evento
        else f"**{', '.join(activos_con_evento)}** sí supera{'n' if len(activos_con_evento) > 1 else ''} el umbral de decisión hoy."
    )

    return (
        f"**Predicción de hoy** — probabilidades {calificativo} "
        f"(la más alta: **{mas_alta['ticker']}** {mas_alta['probabilidad']:.1%}; "
        f"la más baja: **{mas_baja['ticker']}** {mas_baja['probabilidad']:.1%})\n\n"
        + "\n".join(lineas) + f"\n\n{conclusion}\n\n{AVISO_ENFOQUE}"
    )


def _plantilla_shap_importancia(datos: dict) -> str:
    df = datos["informes"].get("shap_importancia")
    if df is None:
        return "El informe de importancia SHAP todavía no está disponible."
    df = df.copy()
    if "pct_del_total" not in df.columns:
        df["pct_del_total"] = df["importancia_media"] / df["importancia_media"].sum() * 100
    top5 = df.sort_values("importancia_media", ascending=False).head(5)
    lineas = [f"- **{fila['variable']}** — {fila['pct_del_total']:.1f}% del total" for _, fila in top5.iterrows()]
    return "**Variables con más peso en el modelo (SHAP)**\n\n" + "\n".join(lineas)


def _plantilla_contribucion_familias(datos: dict) -> str:
    df = datos["informes"].get("contribucion_familias")
    if df is None:
        return "El informe de contribución por familias todavía no está disponible."
    if "n_features" in df.columns:
        lineas = [f"- **{fila['familia']}**: AUC = {fila['auc']:.3f} ({fila['n_features']} variables)" for _, fila in df.iterrows()]
    else:
        lineas = [f"- **{fila['familia']}**: AUC = {fila['auc']:.3f}" for _, fila in df.iterrows()]
    return "**Contribución por familia de variables**\n\n" + "\n".join(lineas)


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
            f"- **{ticker}**: solo financieras {fila['auc_financieras']:.3f} · "
            f"solo comunicación {fila['auc_comunicacion']:.3f} · ambas **{fila['auc_ambas']:.3f}**"
        )
    if not lineas:
        return "No encontré AUC por activo para lo que preguntas."
    return "**AUC por activo**\n\n" + "\n".join(lineas) + f"\n\n{AVISO_ENFOQUE}"


def _plantilla_comparacion_modelos(datos: dict) -> str:
    df = datos["informes"].get("comparacion_modelos")
    if df is None:
        return "El informe de comparación de modelos todavía no está disponible."
    df = df.sort_values("auc", ascending=False)
    lineas = [f"- **{fila['modelo']}**: AUC = {fila['auc']:.3f}, F1 = {fila['f1']:.3f}" for _, fila in df.iterrows()]
    return "**Comparación de modelos baseline** (ordenados por AUC)\n\n" + "\n".join(lineas)


def _plantilla_cv_temporal(datos: dict) -> str:
    df = datos["informes"].get("cv_temporal")
    if df is None:
        return "El informe de validación cruzada temporal todavía no está disponible."
    df = df.sort_values("auc_medio", ascending=False)
    lineas = [
        f"- **{fila['modelo']}**: AUC medio = {fila['auc_medio']:.3f} (±{fila['auc_std']:.3f}, {int(fila['n_splits_validos'])} splits)"
        for _, fila in df.iterrows()
    ]
    return (
        "**Validación cruzada temporal** (5 splits)\n\n" + "\n".join(lineas) +
        "\n\nEl resultado depende del modelo usado — no es igual de robusto en todos los casos."
    )


def _plantilla_matriz_confusion(datos: dict) -> str:
    df = datos["informes"].get("matriz_confusion_umbrales")
    if df is None:
        return "El informe de matriz de confusión por umbrales todavía no está disponible."
    lineas = [
        f"- **Umbral {fila['umbral']}**: precisión = {fila['precision']:.2f}, recall = {fila['recall']:.2f} "
        f"(TP={int(fila['TP'])}, FP={int(fila['FP'])}, FN={int(fila['FN'])}, TN={int(fila['TN'])})"
        for _, fila in df.iterrows()
    ]
    return "**Matriz de confusión según el umbral de decisión**\n\n" + "\n".join(lineas)


def _plantilla_general() -> str:
    return (
        "Puedo ayudarte con esto:\n\n"
        "- **Predicción de hoy** para un activo\n"
        "- **Variables importantes** (SHAP)\n"
        "- **Financieras vs. comunicación** (cuánto aporta cada una)\n"
        "- **Comparación de modelos** y **robustez** en el tiempo\n"
        "- **Analizar un comunicado nuevo** (dime el texto y el activo)\n"
        "- **Consultar un día histórico concreto** dentro del horizonte del TFM\n\n"
        + AVISO_ENFOQUE
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
    activo_no_soportado = clasificacion.get("activo_no_soportado")
    if activo_no_soportado:
        return (
            f"**{activo_no_soportado.capitalize()}** no es uno de los 6 activos con evidencia "
            f"suficiente en este TFM. Los disponibles son: **{', '.join(ACTIVOS_CON_EVIDENCIA)}**."
        )

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
        evento_texto = "✅ **SÍ** hubo" if resultado["evento_importante_real"] else "❌ **NO** hubo"
        anomalia_texto = "**sí** se detectó anomalía (capítulo 5)" if resultado["is_anomaly_real"] else "no se detectó anomalía"
        return (
            f"**Consulta histórica — {resultado['ticker']}, {resultado['fecha']}**\n\n"
            f"{evento_texto} un evento importante real ese día; {anomalia_texto}.\n\n"
            f"- Retorno: **{resultado['log_return_real']:+.2%}**\n"
            f"- Volatilidad (20d): **{resultado['volatility_20d_real']:.2%}**\n"
            f"- Comunicaciones ese día: **{resultado['n_comunicaciones_real']}** "
            f"(sentimiento agregado: {resultado['sentiment_real']:+.3f})\n"
            f"- Probabilidad de referencia del modelo: **{resultado['probabilidad_modelo']:.1%}**\n\n"
            f"{resultado['aviso']}"
        )


def generar_respuesta_simulacion(resultado: dict) -> str:
    try:
        return _llamar_gemini(_prompt_simulacion(resultado))
    except Exception:
        s = resultado["sentimiento"]
        aviso_distribucion = resultado.get("aviso_distribucion")
        return (
            f"**Simulación — {resultado['ticker']}**\n\n"
            f"*\"{resultado['texto_original']}\"*\n\n"
            f"- Sentimiento detectado: **{s['etiqueta']}** "
            f"(positiva {s['prob_positive']:.3f} · negativa {s['prob_negative']:.3f})\n"
            f"- Probabilidad de evento importante — antes: **{resultado['prediccion_antes']['probabilidad']:.1%}** "
            f"→ después: **{resultado['prediccion_despues']['probabilidad']:.1%}** "
            f"({resultado['diferencia_probabilidad']:+.1%})\n\n"
            f"{resultado['aviso']}"
            + (f"\n\n{aviso_distribucion}" if aviso_distribucion else "")
        )
