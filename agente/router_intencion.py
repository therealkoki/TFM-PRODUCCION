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

# Para "evolución del precio" se exige una palabra de evolución/movimiento Y
# una de precio/cotización a la vez — así "evolución de la probabilidad" o
# "cómo se ha movido el mercado en general" no disparan esto por error.
PALABRAS_EVOLUCION = [
    "evolución", "evolucion", "trayectoria", "histórico de precio", "historico de precio",
    "cómo se ha movido", "como se ha movido", "cómo ha ido", "como ha ido", "gráfico de precio",
    "grafico de precio", "serie temporal",
]
PALABRAS_PRECIO = ["precio", "cotización", "cotizacion", "valor del activo"]

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


# Nombres de activos/empresas habituales que NO están soportados — se usan solo
# para poder avisar explícitamente ("Apple no es uno de los 6 activos...") en
# vez de responder en silencio con los 6 sí soportados, como si el usuario no
# hubiera especificado ningún activo. Lista no exhaustiva, cubre los casos más
# probables en una demo o defensa.
ACTIVOS_NO_SOPORTADOS_COMUNES = [
    "apple", "google", "alphabet", "microsoft", "amazon", "meta", "facebook",
    "nvidia", "netflix", "oro", "gold", "plata", "silver", "petróleo", "petroleo",
    "oil", "brent", "wti", "euro/dólar", "eur/usd", "dólar", "dolar", "dow jones",
    "nikkei", "ibex",
]


PALABRAS_SALUDO = ["hola", "buenas", "buenos días", "buenas tardes", "buenas noches", "hey", "qué tal", "que tal"]
PALABRAS_CORTESIA_CIERRE = ["gracias", "vale", "ok", "okay", "genial", "perfecto", "de acuerdo",
                            "adiós", "adios", "chao", "hasta luego", "nos vemos"]


def es_saludo(mensaje: str) -> bool:
    texto = mensaje.lower().strip().strip("!¡.,?¿")
    return any(texto == p or texto.startswith(p + " ") for p in PALABRAS_SALUDO)


def es_cortesia_cierre(mensaje: str) -> bool:
    texto = mensaje.lower().strip().strip("!¡.,?¿")
    return len(texto.split()) <= 4 and any(p in texto for p in PALABRAS_CORTESIA_CIERRE)


def hay_senal_de_tema_nuevo(mensaje: str, tipo_pendiente: str) -> bool:
    """
    Comprueba si un mensaje nuevo, llegado mientras el agente esperaba
    completar una simulación o consulta histórica (ticker/fecha/texto que
    faltaba), en realidad señala con claridad que el usuario quiere abandonar
    eso y pedir otra cosa distinta — para no quedarse insistiendo para
    siempre con la misma pregunta si el usuario simplemente cambió de tema
    (un saludo, un cierre de cortesía, otra simulación, otra consulta
    histórica, o una pregunta sobre un informe concreto).
    """
    if es_saludo(mensaje) or es_cortesia_cierre(mensaje):
        return True
    texto = mensaje.lower()
    if tipo_pendiente != "simulacion" and any(palabra in texto for palabra in PALABRAS_SIMULACION):
        return True
    if tipo_pendiente != "consulta_historica" and any(palabra in texto for palabra in PALABRAS_CONSULTA_HISTORICA):
        return True
    if detectar_tema_pregunta_datos(mensaje) is not None:
        return True
    return False


def detectar_activo_no_soportado(mensaje: str) -> str | None:
    """Si el mensaje menciona un activo/empresa habitual que NO está entre los
    6 soportados, y no menciona ninguno de los 6 sí soportados, devuelve el
    nombre mencionado (para poder aclararlo explícitamente en la respuesta)."""
    if detectar_ticker(mensaje) is not None:
        return None
    texto = mensaje.lower()
    for nombre in ACTIVOS_NO_SOPORTADOS_COMUNES:
        if nombre in texto:
            return nombre
    return None


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


def detectar_todas_las_fechas(mensaje: str) -> list:
    """Encuentra TODAS las fechas válidas del mensaje (ISO o naturales en
    español), ordenadas cronológicamente — a diferencia de detectar_fecha(),
    que solo devuelve la primera. Se usa para detectar rangos ("del X al Y")."""
    import datetime

    fechas = []

    for anio, mes, dia in re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", mensaje):
        try:
            fechas.append(datetime.date(int(anio), int(mes), int(dia)).isoformat())
        except ValueError:
            pass

    for dia, mes_texto, anio in re.findall(
        r"\b(\d{1,2})\s+de\s+([a-záéíóú]+)\s+de\s+(\d{4})\b", mensaje.lower()
    ):
        mes = MESES_ES.get(mes_texto)
        if mes:
            try:
                fechas.append(datetime.date(int(anio), mes, int(dia)).isoformat())
            except ValueError:
                pass

    return sorted(set(fechas))


PALABRAS_TODO_HISTORICO = [
    "todo el histórico", "todo el historico", "todo el periodo", "todo el período",
    "histórico completo", "historico completo", "desde el principio",
]


def es_todo_el_historico(mensaje: str) -> bool:
    texto = mensaje.lower().strip().strip("!¡.,?¿")
    if any(p in texto for p in PALABRAS_TODO_HISTORICO):
        return True
    # "todo" a secas solo cuenta si es el mensaje completo (o casi), no si
    # aparece de paso dentro de una frase más larga sobre otra cosa.
    return texto in ("todo", "todo el", "completo", "todo completo")


def detectar_rango_fechas_explicito(mensaje: str) -> tuple:
    """
    Para cuando se le pide explícitamente al usuario el periodo (después de
    _pedir_rango_fechas()): exige DOS fechas completas y válidas en el mismo
    mensaje para aceptar un rango — nunca infiere un año compartido ni asume
    "desde esa fecha hasta hoy" con una sola fecha, para evitar ambigüedad.
    Devuelve (fecha_inicio, fecha_fin) si hay exactamente 2 fechas válidas
    (ordenadas), o (None, None) si no encuentra un rango claro (en cuyo caso
    hay que volver a pedirlo).
    """
    fechas = detectar_todas_las_fechas(mensaje)
    if len(fechas) == 2:
        return fechas[0], fechas[1]
    return None, None


def detectar_rango_fechas(mensaje: str) -> tuple:
    """
    Busca un rango de fechas en el mensaje (p. ej. "del 1 de enero al 1 de
    marzo de 2025", o simplemente dos fechas sueltas). Devuelve
    (fecha_inicio, fecha_fin), cualquiera de las dos puede ser None:
    - Si hay 2+ fechas: la más antigua y la más reciente.
    - Si hay solo 1 fecha: se interpreta como "desde esa fecha hasta hoy"
      (fecha_inicio=esa fecha, fecha_fin=None).
    - Si no hay ninguna: (None, None), se usará el histórico completo.
    """
    fechas = detectar_todas_las_fechas(mensaje)
    if len(fechas) >= 2:
        return fechas[0], fechas[-1]
    if len(fechas) == 1:
        return fechas[0], None
    return None, None


def detectar_fecha(mensaje: str) -> str | None:
    """
    Busca una fecha en el mensaje, en formato ISO (2025-04-09) o en formato
    natural en español ("9 de abril de 2025"). Devuelve la fecha en formato
    ISO (YYYY-MM-DD) si es una fecha real y válida, o None si no encuentra
    ninguna o si la que encuentra es imposible (p. ej. "32 de enero").
    """
    import datetime

    def _validar(anio: str, mes: int, dia: str) -> str | None:
        try:
            fecha = datetime.date(int(anio), mes, int(dia))
        except ValueError:
            return None
        return fecha.isoformat()

    coincidencia_iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", mensaje)
    if coincidencia_iso:
        anio, mes, dia = coincidencia_iso.groups()
        return _validar(anio, int(mes), dia)

    coincidencia_natural = re.search(
        r"\b(\d{1,2})\s+de\s+([a-záéíóú]+)\s+de\s+(\d{4})\b", mensaje.lower()
    )
    if coincidencia_natural:
        dia, mes_texto, anio = coincidencia_natural.groups()
        mes = MESES_ES.get(mes_texto)
        if mes:
            return _validar(anio, mes, dia)

    return None


def detectar_tema_pregunta_datos(mensaje: str) -> str | None:
    """Para preguntas sobre datos ya existentes, identifica a qué informe se refiere."""
    texto = mensaje.lower()
    for tema, palabras_clave in TEMAS_PREGUNTA_DATOS.items():
        if any(palabra in texto for palabra in palabras_clave):
            return tema
    return None


def es_pregunta_evolucion_precio(mensaje: str) -> bool:
    texto = mensaje.lower()
    if not any(p in texto for p in PALABRAS_EVOLUCION):
        return False
    # Basta con la palabra de evolución + un ticker reconocido (p. ej.
    # "evolución de Bitcoin"), o con la palabra de evolución + "precio"/
    # "cotización" aunque no se mencione ningún ticker en esa misma frase.
    return any(p in texto for p in PALABRAS_PRECIO) or detectar_ticker(mensaje) is not None


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
    es_consulta_historica = any(palabra in texto for palabra in PALABRAS_CONSULTA_HISTORICA)
    if es_consulta_historica:
        return {
            "tipo": "consulta_historica",
            "ticker": ticker,
            "fecha": fecha,
        }

    if es_pregunta_evolucion_precio(mensaje):
        return {
            "tipo": "evolucion_precio",
            "ticker": ticker,
        }

    return {
        "tipo": "pregunta_datos",
        "ticker": ticker,
        "tema": detectar_tema_pregunta_datos(mensaje),
        "activo_no_soportado": detectar_activo_no_soportado(mensaje),
    }
