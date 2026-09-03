"""
Análisis de un comunicado nuevo (tuit real recién publicado, o hipotético) para
la sección 8.4 del TFM.

Reutiliza, literalmente, el mismo preprocesado de texto y la misma inferencia
del modelo de sentimiento que src/04_analisis_semantico.py, y la misma
función predecir_evento_importante() de src/06_modelo_predictivo.py, para que
el resultado sea comparable al del pipeline histórico.

Flujo:
1. Limpieza de texto (idéntica al módulo 2: ftfy + quitar HTML + espacios,
   luego emoji.demojize, luego normalizar menciones/URLs).
2. Sentimiento con el modelo fine-tuned (prob_negative/neutral/positive).
3. sentimiento_continuo = prob_positive - prob_negative (misma fórmula que
   src/05_analisis_impacto_mercados.py, línea 609).
4. Como es una única comunicación nueva (n=1): sentiment = sentimiento_continuo,
   intensidad_max = intensidad_media = abs(sentimiento_continuo), n_comunicaciones = 1.
5. Se coge la última fila de dataset_consolidado_05.csv (salida del módulo 5,
   recalculada a diario SIN corte de fecha) para el ticker indicado — es decir,
   las condiciones de mercado reales más recientes disponibles, no una fecha
   fija del horizonte de entrenamiento. Los 10 lags (log_return/volatility de
   1 a 5 días atrás) no vienen en ese CSV, así que se calculan aquí mismo con
   la misma operación exacta que usa construir_dataset_modelado() en
   src/06_modelo_predictivo.py (groupby("ticker").shift(lag)), sin su corte
   de fecha (FECHA_FIN_TFM).
6. Se sustituyen SOLO las 4 columnas de comunicación por las del comunicado
   nuevo, y se llama a predecir_evento_importante() con el resto de variables
   financieras (las reales, de hoy) intactas.
7. Se compara la probabilidad antes (con las comunicaciones reales del día)
   y después (con las del comunicado nuevo).
8. Se comprueba si las condiciones de mercado de hoy caen fuera del rango
   observado durante el entrenamiento (dataset_modelado.csv, congelado en el
   horizonte del TFM) — si es así, se añade un aviso explícito de menor
   fiabilidad, en vez de dar la predicción sin matizar.

IMPORTANTE (léase antes de usar en la defensa): el MODELO sigue entrenado
únicamente con datos hasta el corte del horizonte de estudio del TFM — lo que
aquí se actualiza a diario son las variables de ENTRADA (condiciones de
mercado), no los patrones aprendidos por el modelo. Es decir, se trata de
inferencia con un modelo estático sobre datos nuevos, la forma estándar de
desplegar cualquier modelo de ML en producción; la fiabilidad de cada
predicción concreta depende de cuánto se parezcan esas condiciones nuevas,
estadísticamente, a las que el modelo vio durante el entrenamiento.
"""

import re

import emoji
import ftfy
import numpy as np
import pandas as pd
import torch

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")

FEATURES_COMUNICACION = ["sentiment", "intensidad_max", "intensidad_media", "n_comunicaciones"]

ACTIVOS_CON_EVIDENCIA = ["IXIC", "XLE", "TSLA", "GSPC", "ETH-USD", "BTC-USD"]

# Debe coincidir con VARIABLES_CHEQUEO_DISTRIBUCION de carga_datos.py.
VARIABLES_CHEQUEO_DISTRIBUCION = ["volatility_20d", "volume_zscore_20d", "log_return"]


def limpiar_texto_base(texto: str) -> str:
    """Copia literal de limpiar_texto_base() en src/04_analisis_semantico.py."""
    if pd.isna(texto):
        return ""
    texto = ftfy.fix_text(str(texto))
    texto = HTML_TAG_PATTERN.sub(" ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def preprocesar_para_transformer(texto: str) -> str:
    """Copia literal de preprocesar_para_transformer() en src/04_analisis_semantico.py."""
    texto = MENTION_PATTERN.sub("@user", texto)
    return URL_PATTERN.sub("http", texto).strip()


def preparar_texto(texto_original: str) -> str:
    """Aplica la misma cadena de limpieza que preparar_texto_modelo() del módulo 2, para un único texto."""
    texto_limpio = limpiar_texto_base(texto_original)
    texto_limpio = emoji.demojize(texto_limpio, language="es")
    return preprocesar_para_transformer(texto_limpio)


def preparar_fila_base(dataset_consolidado_05: pd.DataFrame, ticker: str) -> dict:
    """
    Coge la última fila real de dataset_consolidado_05.csv (salida del módulo 5,
    sin corte de fecha) para el ticker indicado, y le calcula los 10 lags que
    ese CSV no trae de fábrica, con la misma operación que usa
    construir_dataset_modelado() en src/06_modelo_predictivo.py — solo que
    aquí SIN el corte FECHA_FIN_TFM, para poder usar el día más reciente real.
    """
    filas_ticker = dataset_consolidado_05[dataset_consolidado_05["ticker"] == ticker].copy()
    if filas_ticker.empty:
        raise ValueError(f"No hay filas de '{ticker}' en dataset_consolidado_05.csv.")

    filas_ticker = filas_ticker.sort_values("date").reset_index(drop=True)

    for lag in range(1, 6):
        filas_ticker[f"log_return_lag{lag}"] = filas_ticker["log_return"].shift(lag)
        filas_ticker[f"volatility_lag{lag}"] = filas_ticker["volatility_20d"].shift(lag)

    for columna in FEATURES_COMUNICACION:
        if columna in filas_ticker.columns:
            filas_ticker[columna] = filas_ticker[columna].fillna(0)

    ultima_fila = filas_ticker.iloc[-1]

    # Si hay menos de 5 días de historial disponibles (caso extremo, no esperado
    # en producción), los lags más antiguos saldrían NaN — se rellenan a 0 en
    # vez de fallar, ya que el modelo espera siempre un valor numérico.
    cols_lag = [c for c in filas_ticker.columns if "_lag" in c]
    fila_dict = ultima_fila.to_dict()
    for col in cols_lag:
        if pd.isna(fila_dict.get(col)):
            fila_dict[col] = 0.0

    return fila_dict


def comprobar_fuera_de_distribucion(fila: dict, ticker: str, rangos_entrenamiento: dict) -> str | None:
    """
    Compara las condiciones de mercado de 'fila' (el día que se va a usar en la
    simulación) contra el rango [percentil 1, percentil 99] que tuvieron esas
    mismas variables, PARA ESE MISMO ACTIVO, durante el horizonte de
    entrenamiento (dataset_modelado.csv). Devuelve None si todo está dentro de
    rango, o un texto de aviso explícito si alguna variable se sale.
    """
    rangos_ticker = rangos_entrenamiento.get(ticker) if rangos_entrenamiento else None
    if not rangos_ticker:
        return None

    fuera_de_rango = []
    for variable in VARIABLES_CHEQUEO_DISTRIBUCION:
        if variable not in rangos_ticker or variable not in fila:
            continue
        p1, p99 = rangos_ticker[variable]
        valor = fila[variable]
        if pd.isna(valor):
            continue
        if valor < p1 or valor > p99:
            fuera_de_rango.append(f"{variable}={valor:.4f} (rango visto en entrenamiento: [{p1:.4f}, {p99:.4f}])")

    if not fuera_de_rango:
        return None

    return (
        "Aviso: las condiciones de mercado actuales de " + ticker + " están fuera del rango "
        "observado durante el entrenamiento del modelo — " + "; ".join(fuera_de_rango) +
        ". Esta predicción tiene menor fiabilidad de lo habitual, ya que el modelo está "
        "extrapolando fuera de lo que aprendió."
    )


def calcular_sentimiento(texto_original: str, tokenizer, modelo) -> dict:
    """
    Calcula prob_negative/neutral/positive para un único texto con el modelo
    fine-tuned, y deriva sentimiento_continuo = prob_positive - prob_negative
    (misma fórmula que el módulo 5).
    """
    texto_modelo = preparar_texto(texto_original)

    inputs = tokenizer([texto_modelo], return_tensors="pt", truncation=True, padding=True, max_length=128)
    modelo.eval()
    with torch.no_grad():
        probs = torch.softmax(modelo(**inputs).logits, dim=-1).cpu().numpy()[0]

    prob_negative = float(probs[LABEL2ID["negative"]])
    prob_neutral = float(probs[LABEL2ID["neutral"]])
    prob_positive = float(probs[LABEL2ID["positive"]])
    sentimiento_continuo = prob_positive - prob_negative

    return {
        "texto_modelo": texto_modelo,
        "etiqueta": ID2LABEL[int(np.argmax(probs))],
        "prob_negative": prob_negative,
        "prob_neutral": prob_neutral,
        "prob_positive": prob_positive,
        "sentimiento_continuo": sentimiento_continuo,
    }


def analizar_comunicado_nuevo(texto: str, ticker: str, dataset_consolidado_05: pd.DataFrame,
                               tokenizer, modelo, modelo_info: dict,
                               rangos_entrenamiento: dict = None) -> dict:
    """
    Pipeline completo: sentimiento del texto nuevo -> condiciones de mercado
    reales más recientes (dataset_consolidado_05, sin corte de fecha) ->
    sustitución de columnas de comunicación -> predicción antes/después ->
    chequeo de fuera de distribución frente al horizonte de entrenamiento.

    Devuelve un dict con toda la información necesaria para que la capa de
    lenguaje natural (Gemini o plantilla) construya la respuesta, incluyendo
    el aviso de que esto es una lectura de sensibilidad del modelo, no una
    predicción de mercado garantizada, y el aviso de fuera de distribución
    si aplica.
    """
    if ticker not in ACTIVOS_CON_EVIDENCIA:
        raise ValueError(
            f"'{ticker}' no es uno de los activos con evidencia suficiente. "
            f"Los disponibles son: {', '.join(ACTIVOS_CON_EVIDENCIA)}."
        )

    fila_base = preparar_fila_base(dataset_consolidado_05, ticker)

    resultado_sentimiento = calcular_sentimiento(texto, tokenizer, modelo)
    sentimiento_continuo = resultado_sentimiento["sentimiento_continuo"]

    valores_comunicacion_nuevos = {
        "sentiment": sentimiento_continuo,
        "intensidad_max": abs(sentimiento_continuo),
        "intensidad_media": abs(sentimiento_continuo),
        "n_comunicaciones": 1,
    }

    datos_antes = {k: v for k, v in fila_base.items() if k in modelo_info["feature_cols_originales"]}
    datos_despues = dict(datos_antes)
    datos_despues.update(valores_comunicacion_nuevos)

    prediccion_antes = predecir_evento_importante(datos_antes, modelo_info)
    prediccion_despues = predecir_evento_importante(datos_despues, modelo_info)

    aviso_distribucion = comprobar_fuera_de_distribucion(fila_base, ticker, rangos_entrenamiento or {})

    return {
        "ticker": ticker,
        "fecha_datos_mercado": fila_base.get("date"),
        "texto_original": texto,
        "sentimiento": resultado_sentimiento,
        "valores_comunicacion_reales_ultimo_dia": {k: fila_base.get(k) for k in FEATURES_COMUNICACION},
        "valores_comunicacion_nuevos": valores_comunicacion_nuevos,
        "prediccion_antes": prediccion_antes,
        "prediccion_despues": prediccion_despues,
        "diferencia_probabilidad": round(
            prediccion_despues["probabilidad"] - prediccion_antes["probabilidad"], 4
        ),
        "aviso_distribucion": aviso_distribucion,
        "aviso": (
            "Esto es una simulación de sensibilidad del modelo ante un comunicado nuevo, "
            "no una predicción real de mercado. El TFM descarta predecir la dirección del "
            "retorno (capítulo 6); esta cifra refleja únicamente cómo cambia la probabilidad "
            "de evento importante estimada por el modelo al variar las columnas de "
            "comunicación, manteniendo las condiciones de mercado reales más recientes "
            "disponibles. El modelo en sí sigue entrenado solo con datos hasta el horizonte "
            "de estudio del TFM; esta predicción aplica esos patrones aprendidos a "
            "condiciones de mercado actuales."
        ),
    }


def consultar_dia_historico(ticker: str, fecha: str, dataset_modelado: pd.DataFrame, modelo_info: dict) -> dict:
    """
    Consulta un día concreto YA CONOCIDO dentro del horizonte de estudio del
    TFM (Modo A) — NO es una predicción nueva. Devuelve el hecho real tal
    como quedó registrado en dataset_modelado.csv (evento_importante, is_anomaly),
    y, como dato secundario, la probabilidad que el modelo le asigna a esa
    misma fila.

    IMPORTANTE — por qué esa probabilidad NO es una predicción genuina: el
    modelo final (el que usa el agente) se entrena en src/06_modelo_predictivo.py
    sobre el histórico COMPLETO (entrenar_modelo_final() usa X_full, no una
    partición de test separada). Es decir, el modelo YA VIO este día durante
    su entrenamiento. Preguntarle su probabilidad para un día que ya conoce
    no demuestra capacidad predictiva — solo confirma ajuste interno. Por eso
    esta función se llama "consultar", no "predecir", y el resultado se
    etiqueta siempre con este matiz para no confundirlo con la simulación de
    un comunicado nuevo (analizar_comunicado_nuevo), que sí es inferencia
    genuina sobre condiciones que el modelo no vio durante el entrenamiento.
    """
    if ticker not in ACTIVOS_CON_EVIDENCIA:
        raise ValueError(
            f"'{ticker}' no es uno de los activos con evidencia suficiente. "
            f"Los disponibles son: {', '.join(ACTIVOS_CON_EVIDENCIA)}."
        )

    filas = dataset_modelado[
        (dataset_modelado["ticker"] == ticker) & (dataset_modelado["date"] == fecha)
    ]
    if filas.empty:
        raise ValueError(
            f"No hay datos de '{ticker}' para la fecha {fecha} en el horizonte de estudio del TFM "
            f"(dataset_modelado.csv). Puede que esa fecha esté fuera del horizonte, o que ese día "
            f"faltaran datos suficientes para calcular los lags."
        )

    fila = filas.iloc[0].to_dict()

    datos_modelo = {k: v for k, v in fila.items() if k in modelo_info["feature_cols_originales"]}
    prediccion = predecir_evento_importante(datos_modelo, modelo_info)

    return {
        "ticker": ticker,
        "fecha": fecha,
        "evento_importante_real": bool(fila.get("evento_importante")),
        "is_anomaly_real": bool(fila.get("is_anomaly")),
        "log_return_real": fila.get("log_return"),
        "volatility_20d_real": fila.get("volatility_20d"),
        "sentiment_real": fila.get("sentiment"),
        "n_comunicaciones_real": fila.get("n_comunicaciones"),
        "probabilidad_modelo": prediccion["probabilidad"],
        "aviso": (
            "Esto es una consulta de un hecho histórico ya registrado en el horizonte de estudio "
            "del TFM, no una predicción nueva. La 'probabilidad del modelo' que se muestra es solo "
            "una referencia de ajuste interno: el modelo final se entrenó sobre el histórico "
            "completo, incluido este mismo día, así que ya lo conocía — esta cifra no demuestra "
            "capacidad predictiva genuina, a diferencia de la simulación de un comunicado nuevo."
        ),
    }


def predecir_evento_importante(datos_nuevos: dict, modelo_info: dict) -> dict:
    """Copia literal de predecir_evento_importante() en src/06_modelo_predictivo.py."""
    modelo = modelo_info["modelo"]
    columnas_esperadas = modelo_info["columnas_esperadas"]
    umbral = modelo_info["umbral_decision"]

    df_input = pd.DataFrame([datos_nuevos])
    df_input_dummies = pd.get_dummies(df_input, columns=["ticker"], prefix="ticker")
    df_input_dummies = df_input_dummies.reindex(columns=columnas_esperadas, fill_value=0)

    probabilidad = modelo.predict_proba(df_input_dummies)[0, 1]
    return {
        "probabilidad": round(float(probabilidad), 4),
        "es_evento": bool(probabilidad >= umbral),
        "umbral_usado": umbral,
    }


