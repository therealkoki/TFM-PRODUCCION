"""
Gráficos nativos (Plotly) para la sección 8.4 del TFM.

Cada función recibe los mismos DataFrames que ya usa respuestas.py y devuelve
una figura de Plotly lista para pegar debajo del mensaje del chat con
st.plotly_chart(). Sustituye al dashboard de Tableau embebido: se generan
directamente en Python a partir de los mismos datos que ya tiene el agente
en memoria, así que se adaptan solos al ancho de la columna y comparten la
paleta de color oscura de toda la app, sin depender de una publicación externa.

Paleta: mismo acento morado que .streamlit/config.toml (#7C5CFC), fondo oscuro
a juego con el tema, elemento(s) relevante(s) resaltado(s) en el acento y el
resto en gris tenue, para dirigir la vista hacia lo que responde la pregunta.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go

COLOR_ACENTO = "#7C5CFC"
COLOR_TENUE = "#3A3F4B"
COLOR_FONDO = "#0E1117"
COLOR_TEXTO = "#E5E7EB"
COLOR_CUADRICULA = "#2D3340"

LAYOUT_BASE = dict(
    paper_bgcolor=COLOR_FONDO,
    plot_bgcolor=COLOR_FONDO,
    font=dict(color=COLOR_TEXTO, size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    height=320,
    showlegend=False,
)


def _colores_destacado(etiquetas: list, destacados: list) -> list:
    """Devuelve una lista de colores: acento para las etiquetas destacadas, gris para el resto."""
    destacados = destacados or []
    return [COLOR_ACENTO if e in destacados else COLOR_TENUE for e in etiquetas]


def grafico_predicciones_hoy(df: pd.DataFrame, tickers_destacados: list = None) -> go.Figure:
    df = df.sort_values("probabilidad", ascending=True)
    colores = _colores_destacado(df["ticker"].tolist(), tickers_destacados)
    fig = go.Figure(go.Bar(
        x=df["probabilidad"], y=df["ticker"], orientation="h",
        marker_color=colores, text=[f"{p:.1%}" for p in df["probabilidad"]], textposition="outside",
    ))
    fig.update_layout(**LAYOUT_BASE, title="Predicción de hoy — probabilidad de evento importante",
                       xaxis=dict(tickformat=".1%", gridcolor=COLOR_CUADRICULA), yaxis=dict(gridcolor=COLOR_CUADRICULA))
    return fig


def grafico_shap_importancia(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    if "pct_del_total" not in df.columns:
        df = df.copy()
        df["pct_del_total"] = df["importancia_media"] / df["importancia_media"].sum() * 100
    top = df.sort_values("importancia_media", ascending=True).tail(top_n)
    fig = go.Figure(go.Bar(
        x=top["pct_del_total"], y=top["variable"], orientation="h",
        marker_color=COLOR_ACENTO, text=[f"{p:.1f}%" for p in top["pct_del_total"]], textposition="outside",
    ))
    fig.update_layout(**LAYOUT_BASE, title=f"Top {top_n} variables por importancia (SHAP)",
                       xaxis=dict(title="% del total", gridcolor=COLOR_CUADRICULA), yaxis=dict(gridcolor=COLOR_CUADRICULA))
    return fig


def grafico_contribucion_familias(df: pd.DataFrame) -> go.Figure:
    df = df.sort_values("auc", ascending=True)
    fig = go.Figure(go.Bar(
        x=df["auc"], y=df["familia"], orientation="h",
        marker_color=COLOR_ACENTO, text=[f"{a:.3f}" for a in df["auc"]], textposition="outside",
    ))
    fig.update_layout(**LAYOUT_BASE, title="AUC según las variables usadas",
                       xaxis=dict(title="AUC", range=[0, 1], gridcolor=COLOR_CUADRICULA), yaxis=dict(gridcolor=COLOR_CUADRICULA))
    return fig


def grafico_auc_por_activo(df: pd.DataFrame, tickers_destacados: list = None) -> go.Figure:
    df = df.sort_values("auc_comunicacion", ascending=True)
    colores = _colores_destacado(df["ticker"].tolist(), tickers_destacados)
    fig = go.Figure(go.Bar(
        x=df["auc_comunicacion"], y=df["ticker"], orientation="h",
        marker_color=colores, text=[f"{a:.3f}" for a in df["auc_comunicacion"]], textposition="outside",
    ))
    fig.update_layout(**LAYOUT_BASE, title="AUC de \"solo comunicación\" por activo",
                       xaxis=dict(title="AUC", gridcolor=COLOR_CUADRICULA), yaxis=dict(gridcolor=COLOR_CUADRICULA))
    return fig


def grafico_comparacion_modelos(df: pd.DataFrame) -> go.Figure:
    df = df.sort_values("auc", ascending=True)
    fig = go.Figure(go.Bar(
        x=df["auc"], y=df["modelo"], orientation="h",
        marker_color=COLOR_ACENTO, text=[f"{a:.3f}" for a in df["auc"]], textposition="outside",
    ))
    fig.update_layout(**LAYOUT_BASE, title="Comparación de modelos baseline (AUC)",
                       xaxis=dict(title="AUC", gridcolor=COLOR_CUADRICULA), yaxis=dict(gridcolor=COLOR_CUADRICULA))
    return fig


def grafico_cv_temporal(df: pd.DataFrame) -> go.Figure:
    df = df.sort_values("auc_medio", ascending=True)
    fig = go.Figure(go.Bar(
        x=df["auc_medio"], y=df["modelo"], orientation="h",
        marker_color=COLOR_ACENTO,
        error_x=dict(type="data", array=df["auc_std"], color=COLOR_TEXTO, thickness=1.2),
        text=[f"{a:.3f}" for a in df["auc_medio"]], textposition="outside",
    ))
    fig.update_layout(**LAYOUT_BASE, title="Validación cruzada temporal (AUC medio ± desviación)",
                       xaxis=dict(title="AUC", gridcolor=COLOR_CUADRICULA), yaxis=dict(gridcolor=COLOR_CUADRICULA))
    return fig


def grafico_simulacion(resultado: dict) -> go.Figure:
    etiquetas = ["Antes", "Después"]
    valores = [resultado["prediccion_antes"]["probabilidad"], resultado["prediccion_despues"]["probabilidad"]]
    sube = valores[1] >= valores[0]
    colores = [COLOR_TENUE, COLOR_ACENTO if sube else "#EF4444"]
    fig = go.Figure(go.Bar(
        x=etiquetas, y=valores, marker_color=colores,
        text=[f"{v:.1%}" for v in valores], textposition="outside",
    ))
    fig.update_layout(**{**LAYOUT_BASE, "height": 280}, title=f"Probabilidad de evento importante — {resultado['ticker']}",
                       yaxis=dict(tickformat=".1%", gridcolor=COLOR_CUADRICULA))
    return fig


def calcular_serie_evolucion_precio(dataset_consolidado_05: pd.DataFrame, ticker: str,
                                     fecha_inicio: str = None, fecha_fin: str = None) -> pd.DataFrame:
    """
    Construye la serie con el índice de evolución relativa (base 100) para
    un ticker — reutilizada tanto por el gráfico como por el texto que lo
    acompaña, para no calcularla dos veces con el riesgo de que diverjan.

    Si se pasan fecha_inicio/fecha_fin, se recorta la serie a ese rango ANTES
    de calcular el índice, para que el 100 quede justo al principio de la
    ventana pedida (no arrastrando la tendencia acumulada de antes del rango).
    """
    df = dataset_consolidado_05[dataset_consolidado_05["ticker"] == ticker].sort_values("date").copy()
    if fecha_inicio:
        df = df[df["date"] >= pd.Timestamp(fecha_inicio)]
    if fecha_fin:
        df = df[df["date"] <= pd.Timestamp(fecha_fin)]
    df["indice"] = 100 * np.exp(df["log_return"].cumsum())
    return df


def grafico_evolucion_precio(dataset_consolidado_05: pd.DataFrame, ticker: str,
                              fecha_inicio: str = None, fecha_fin: str = None) -> go.Figure:
    """
    Índice de evolución relativa (base 100), construido a partir de los
    log_return diarios de dataset_consolidado_05.

    IMPORTANTE: no es el precio real en dólares/euros — ninguno de los CSV
    del pipeline guarda precios absolutos, solo retornos diarios. El índice
    sube y baja exactamente igual que el precio real (misma forma, mismos
    porcentajes de subida/bajada), solo que normalizado para empezar en 100,
    de forma que se puede leer la evolución relativa sin necesitar el precio
    real de partida.
    """
    df = calcular_serie_evolucion_precio(dataset_consolidado_05, ticker, fecha_inicio, fecha_fin)

    fig = go.Figure(go.Scatter(
        x=df["date"], y=df["indice"], mode="lines",
        line=dict(color=COLOR_ACENTO, width=2),
    ))
    fig.update_layout(
        **{**LAYOUT_BASE, "height": 350},
        title=f"Evolución relativa de {ticker} (índice, base 100)",
        xaxis=dict(gridcolor=COLOR_CUADRICULA),
        yaxis=dict(title="Índice (base 100)", gridcolor=COLOR_CUADRICULA),
    )
    return fig


def grafico_consulta_historica(resultado: dict) -> go.Figure:
    etiquetas = ["Retorno", "Volatilidad (20d)"]
    valores = [resultado["log_return_real"], resultado["volatility_20d_real"]]
    colores = [COLOR_ACENTO if resultado["evento_importante_real"] else COLOR_TENUE] * 2
    fig = go.Figure(go.Bar(
        x=etiquetas, y=valores, marker_color=colores,
        text=[f"{v:+.2%}" if i == 0 else f"{v:.2%}" for i, v in enumerate(valores)], textposition="outside",
    ))
    titulo = f"{resultado['ticker']} — {resultado['fecha']}"
    if resultado["evento_importante_real"]:
        titulo += " (evento importante real)"
    fig.update_layout(**{**LAYOUT_BASE, "height": 280}, title=titulo, yaxis=dict(tickformat=".1%", gridcolor=COLOR_CUADRICULA))
    return fig
