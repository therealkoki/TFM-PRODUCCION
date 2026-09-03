"""
Router de intención para el agente de la sección 8.4 del TFM.

Decide, para cada mensaje del usuario, si se trata de:
- una SIMULACIÓN: un comunicado nuevo (real o hipotético) que hay que analizar
  con el modelo (delegado a simulacion.analizar_comunicado_nuevo), o
- una PREGUNTA_DATOS: una pregunta sobre algo que el modelo/análisis ya calculó
  (predicción de hoy, SHAP, AUC por activo, robustez, etc.), respondida
  leyendo los informes ya cargados en memoria (carga_datos.cargar_todo()).

Deliberadamente basado en palabras clave, no en un LLM: esta clasificación
tiene que funcionar siempre, sin depender de una API externa ni de su cuota.
Gemini (si se usa) entra solo después, para redactar en lenguaje natural la
respuesta ya construida con los datos correctos — nunca para decidir qué
datos consultar.
"""

import re

ACTIVOS_CON_EVIDENCIA = ["IXIC", "XLE", "TSLA", "GSPC", "ETH-USD", "BTC-USD"]

# Alias en lenguaje natural -> ticker exacto que usan los CSV/el modelo.
ALIAS_ACTIVOS = {
    "ixic": "IXIC", "nasdaq": "IXIC",
    "xle": "XLE", "energia": "XLE", "energía": "XLE", "sector energetico": "XLE", "sector energético": "XLE",
    "tsla": "TSLA", "tesla": "TSLA",
    "gspc": "GSPC", "sp500": "GSPC", "s&p500": "GSPC", "s&p 500": "GSPC", "sp 500": "GSPC",
    "eth": "ETH-USD", "eth-usd": "ETH-USD", "ethereum": "ETH-USD",
    "btc": "BTC-USD", "btc-usd": "BTC-USD", "bitcoin": "BTC-USD",
}

# Palabras que indican petición de simulación / análisis de un comunicado nuevo.
PALABRAS_SIMULACION = [
    "simula", "simular", "simulación", "simulacion",
    "qué pasaría si", "que pasaria si", "hipotético", "hipotetico",
    "analiza este tuit", "analiza este tweet", "analiza este comunicado",
    "acaba de publicar", "acaba de decir", "acaba de tuitear",
    "nuevo tuit", "nuevo tweet", "nuevo comunicado",
]

# Palabras que indican una consulta sobre un día histórico ya conocido (Modo A):
# se responde con el hecho ya calculado (evento_importante, is_anomaly reales de
# ese día), no con una predicción nueva del modelo. Se comprueba ANTES que
# PALABRAS_SIMULACION en la clasificación de abajo solo si hay fecha en el
# mensaje; si no hay fecha, no tiene sentido este tipo de consulta.
PALABRAS_CONSULTA_HISTORICA = [
    "qué volatilidad tuvo", "que volatilidad tuvo", "qué pasó el", "que paso el",
    "hubo un evento", "hubo evento", "fue un evento importante", "fue evento importante",
    "hubo anomalía", "hubo anomalia", "qué volatilidad hubo", "que volatilidad hubo",
    "cómo reaccionó el mercado", "como reacciono el mercado",
]

MESES_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

# Palabras clave -> qué informe consultar para responder una pregunta sobre datos ya existentes.
TEMAS_PREGUNTA_DATOS = {
    "predicciones_hoy": ["predicción de hoy", "prediccion de hoy", "hoy predice", "qué predice", "que predice",
                          "predicción actual", "prediccion actual", "hoy el modelo"],
    "shap_importancia": ["shap", "importancia de las variables", "qué variable pesa", "que variable pesa",
                          "variable más importante", "variable mas importante"],
    "contribucion_familias": ["cuánto pesa el sentimiento", "cuanto pesa el sentimiento",
                               "sentimiento frente a", "financieras frente a", "comunicación frente a",
                               "comunicacion frente a", "familias de variables"],
    "auc_por_activo": ["ayuda más", "ayuda mas", "difiere entre activos",
                        "auc por activo", "auc de cada activo", "comparar activos"],
    "comparacion_modelos": ["qué modelo es mejor", "que modelo es mejor", "comparación de modelos",
                             "comparacion de modelos", "random forest", "xgboost", "lightgbm frente a"],
    "cv_temporal": ["es un resultado robusto", "es robusto", "validación cruzada", "validacion cruzada",
                     "cv temporal", "se mantiene en el tiempo"],
    "matriz_confusion_umbrales": ["matriz de confusión", "matriz de confusion", "umbral de decisión",
                                   "umbral de decision", "falsos positivos", "falsos negativos"],
}


def detectar_ticker(mensaje: str) -> str | None:
    """Busca un ticker o alias conocido en el mensaje. Devuelve el ticker exacto o None."""
    texto = mensaje.lower()
    for ticker in ACTIVOS_CON_EVIDENCIA:
        if ticker.lower() in texto:
            return ticker
    for alias, ticker in ALIAS_ACTIVOS.items():
        if alias in texto:
            return ticker
    return None


def extraer_texto_comunicado(mensaje: str) -> str | None:
    """
    Intenta aislar el texto del comunicado a simular dentro del mensaje del
    usuario: primero busca contenido entre comillas dobles o comillas
    angulares, si no lo encuentra usa lo que venga después de ':'.

    IMPORTANTE: el apóstrofo (') NO se trata como delimitador de cierre, a
    diferencia de una versión anterior de esta función — un apóstrofo normal
    del inglés dentro del texto (p. ej. "World's Markets") cortaba el texto
    a mitad de frase, perdiendo el resto del comunicado sin ningún aviso.
    """
    coincidencia = re.search(r'["«]([^"»]{5,})["»]', mensaje)
    if coincidencia:
        return coincidencia.group(1).strip()

    coincidencia = re.search(r"'([^']{5,})'", mensaje)
    if coincidencia:
        return coincidencia.group(1).strip()

    if ":" in mensaje:
        posible = mensaje.split(":", 1)[1].strip()
        if len(posible) >= 5:
            return posible.strip('"\'«»')

    return None


def detectar_fecha(mensaje: str) -> str | None:
    """
    Busca una fecha en el mensaje, en formato ISO (2025-04-09) o en formato
    natural en español ("9 de abril de 2025"). Devuelve la fecha en formato
    ISO (YYYY-MM-DD) o None si no encuentra ninguna.
    """
    coincidencia_iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", mensaje)
    if coincidencia_iso:
        return coincidencia_iso.group(0)

    coincidencia_natural = re.search(
        r"\b(\d{1,2})\s+de\s+([a-záéíóú]+)\s+de\s+(\d{4})\b", mensaje.lower()
    )
    if coincidencia_natural:
        dia, mes_texto, anio = coincidencia_natural.groups()
        mes = MESES_ES.get(mes_texto)
        if mes:
            return f"{anio}-{mes:02d}-{int(dia):02d}"

    return None


def detectar_tema_pregunta_datos(mensaje: str) -> str | None:
    """Para preguntas sobre datos ya existentes, identifica a qué informe se refiere."""
    texto = mensaje.lower()
    for tema, palabras_clave in TEMAS_PREGUNTA_DATOS.items():
        if any(palabra in texto for palabra in palabras_clave):
            return tema
    return None


def clasificar_mensaje(mensaje: str) -> dict:
    """
    Punto de entrada del router. Devuelve un dict con:
    - tipo: "simulacion" | "consulta_historica" | "pregunta_datos"
    - ticker: ticker detectado o None (si es None, la app debe preguntar cuál)
    - texto_comunicado: solo si tipo == "simulacion"; puede ser None si no se
      pudo extraer, en cuyo caso la app debe pedir el texto explícitamente
    - fecha: solo si tipo == "consulta_historica" (formato ISO YYYY-MM-DD);
      puede ser None si no se detectó, en cuyo caso la app debe pedirla
    - tema: solo si tipo == "pregunta_datos"; puede ser None si no se identifica
      un informe concreto, en cuyo caso conviene responder con una visión general

    Orden de comprobación: simulación primero (un mensaje de simulación puede
    mencionar una fecha solo como metadato de cuándo se publicó originalmente
    el comunicado — eso NO debe activar consulta_historica, que usa las
    condiciones de mercado de esa fecha en vez de las de hoy). La consulta
    histórica solo se activa si además hay palabras clave específicas de
    "qué pasó ese día", no por la mera presencia de una fecha.
    """
    texto = mensaje.lower()
    ticker = detectar_ticker(mensaje)

    es_simulacion = any(palabra in texto for palabra in PALABRAS_SIMULACION)
    if es_simulacion:
        return {
            "tipo": "simulacion",
            "ticker": ticker,
            "texto_comunicado": extraer_texto_comunicado(mensaje),
        }

    fecha = detectar_fecha(mensaje)
    es_consulta_historica = fecha is not None and any(
        palabra in texto for palabra in PALABRAS_CONSULTA_HISTORICA
    )
    if es_consulta_historica:
        return {
            "tipo": "consulta_historica",
            "ticker": ticker,
            "fecha": fecha,
        }

    return {
        "tipo": "pregunta_datos",
        "ticker": ticker,
        "tema": detectar_tema_pregunta_datos(mensaje),
    }

