"""
Módulo de carga de datos para el agente de la sección 8.4 del TFM.

Se conecta a Google Drive con una cuenta de servicio (clave JSON en el secreto
GDRIVE_SERVICE_ACCOUNT_KEY de Streamlit) y descarga/cachea en memoria todos los
artefactos que el agente necesita: predicciones del día, informes de
interpretabilidad, y el modelo predictivo serializado.

Las funciones get_drive_service(), resolver_carpeta_drive(), buscar_archivo() y
descargar_archivo() son una copia (con crear_si_falta=False al resolver
carpetas, ya que este módulo solo lee) de las de src/06_modelo_predictivo.py
(raulruiz25/TFM-PRODUCCION). subir_o_actualizar_archivo() no se incluye aquí
porque el agente nunca escribe en Drive.
"""

import io
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

RUTA_BASE = ["TFM DATA SCIENCE", "data"]
CARPETA_MODELADO = RUTA_BASE + ["PROCESSED - Modelado"]
CARPETA_MODELO_SENTIMIENTO = RUTA_BASE + ["MODELS - Analisis Semantico"]

ACTIVOS_CON_EVIDENCIA = ["IXIC", "XLE", "TSLA", "GSPC", "ETH-USD", "BTC-USD"]

NOMBRES_INFORMES = {
    "predicciones_hoy": "predicciones_hoy.csv",
    "shap_importancia": "informe_shap_importancia.csv",
    "contribucion_familias": "informe_contribucion_familias.csv",
    "auc_por_activo": "informe_auc_por_activo.csv",
    "comparacion_modelos": "informe_comparacion_modelos.csv",
    "cv_temporal": "informe_cv_temporal.csv",
    "matriz_confusion_umbrales": "informe_matriz_confusion_umbrales.csv",
}


def get_drive_service():
    """Devuelve un cliente autenticado de la API de Drive (cuenta de servicio)."""
    key_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT_KEY")
    if not key_json:
        raise RuntimeError("No se encontró la variable de entorno GDRIVE_SERVICE_ACCOUNT_KEY.")
    key_info = json.loads(key_json)
    credentials = service_account.Credentials.from_service_account_info(key_info, scopes=DRIVE_SCOPES)
    return build("drive", "v3", credentials=credentials)


def resolver_carpeta_drive(drive_service, partes_ruta: list, crear_si_falta: bool = True) -> str:
    parent_id = None
    for i, nombre in enumerate(partes_ruta):
        query = f"name = '{nombre}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        resultado = drive_service.files().list(q=query, fields="files(id, name)").execute()
        encontrados = resultado.get("files", [])
        if encontrados:
            parent_id = encontrados[0]["id"]
        elif crear_si_falta and i > 0:
            metadata = {"name": nombre, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
            carpeta = drive_service.files().create(body=metadata, fields="id").execute()
            parent_id = carpeta["id"]
        else:
            raise FileNotFoundError(f"No se encontró la carpeta '{nombre}' (ruta: {' / '.join(partes_ruta[:i + 1])}).")
    return parent_id


def buscar_archivo(drive_service, carpeta_id: str, nombre_archivo: str):
    query = f"name = '{nombre_archivo}' and '{carpeta_id}' in parents and trashed = false"
    resultado = drive_service.files().list(q=query, fields="files(id, name)").execute()
    encontrados = resultado.get("files", [])
    return encontrados[0]["id"] if encontrados else None


def descargar_archivo(drive_service, file_id: str, destino: Path):
    request = drive_service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(buffer.getvalue())


def _descargar_a_local(drive_service, ruta_carpeta: list, nombre_archivo: str, local_dir: Path) -> Path:
    # crear_si_falta=False: el agente solo lee de Drive, nunca debe crear
    # carpetas nuevas silenciosamente si algo no cuadra (a diferencia de
    # los scripts de producción, que sí escriben y necesitan ese comportamiento).
    carpeta_id = resolver_carpeta_drive(drive_service, ruta_carpeta, crear_si_falta=False)
    file_id = buscar_archivo(drive_service, carpeta_id, nombre_archivo)
    if file_id is None:
        raise FileNotFoundError(f"No se encontró '{nombre_archivo}' en la carpeta {' / '.join(ruta_carpeta)}.")
    destino = local_dir / nombre_archivo
    descargar_archivo(drive_service, file_id, destino)
    return destino


@st.cache_resource(show_spinner="Conectando con Google Drive...")
def cargar_todo():
    """
    Descarga y devuelve en un único dict todos los artefactos que el agente necesita:
    - predicciones_hoy: DataFrame con la última predicción por activo (probabilidad, es_evento)
    - informes: dict de DataFrames, uno por cada informe de interpretabilidad (sección 9 del cap. 6)
    - modelo_info: dict con el modelo LightGBM, columnas esperadas y umbral (joblib.load)

    Cacheado con st.cache_resource: solo se descarga una vez por sesión de la app,
    no en cada mensaje del chat.
    """
    local_dir = Path("/tmp/agente_data")
    local_dir.mkdir(parents=True, exist_ok=True)
    drive_service = get_drive_service()

    resultado = {"informes": {}, "errores": []}

    # --- Predicciones del día (generadas por el pipeline de producción, punto 7) ---
    try:
        ruta = _descargar_a_local(drive_service, CARPETA_MODELADO, NOMBRES_INFORMES["predicciones_hoy"], local_dir)
        resultado["predicciones_hoy"] = pd.read_csv(ruta)
    except FileNotFoundError as e:
        resultado["predicciones_hoy"] = None
        resultado["errores"].append(str(e))

    # --- Dataset modelado completo (necesario para la simulación: última fila real por activo) ---
    try:
        ruta = _descargar_a_local(drive_service, CARPETA_MODELADO, "dataset_modelado.csv", local_dir)
        resultado["dataset_modelado"] = pd.read_csv(ruta)
    except FileNotFoundError as e:
        resultado["dataset_modelado"] = None
        resultado["errores"].append(str(e))

    # --- Informes de interpretabilidad (capítulo 6, sección 9) ---
    for clave, nombre_archivo in NOMBRES_INFORMES.items():
        if clave == "predicciones_hoy":
            continue
        try:
            ruta = _descargar_a_local(drive_service, CARPETA_MODELADO, nombre_archivo, local_dir)
            resultado["informes"][clave] = pd.read_csv(ruta)
        except FileNotFoundError as e:
            resultado["informes"][clave] = None
            resultado["errores"].append(str(e))

    # --- Modelo predictivo (LightGBM serializado) ---
    try:
        ruta_modelo = _descargar_a_local(drive_service, CARPETA_MODELADO, "modelo_evento_importante.pkl", local_dir)
        resultado["modelo_info"] = joblib.load(ruta_modelo)
    except FileNotFoundError as e:
        resultado["modelo_info"] = None
        resultado["errores"].append(str(e))

    return resultado


@st.cache_resource(show_spinner="Cargando modelo de sentimiento (puede tardar unos segundos)...")
def cargar_modelo_sentimiento():
    """
    Descarga y descomprime el modelo fine-tuned de sentimiento (zip) desde
    MODELS - Analisis Semantico/, y carga tokenizer + modelo con transformers.

    Se llama de forma perezosa (lazy): solo cuando llega la primera simulación
    de un tuit nuevo, no al arrancar la app, para no consumir de golpe los
    ~1GB de RAM gratuitos de Streamlit Community Cloud.
    """
    import zipfile

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    local_dir = Path("/tmp/agente_data")
    local_dir.mkdir(parents=True, exist_ok=True)
    drive_service = get_drive_service()

    ruta_zip = _descargar_a_local(
        drive_service, CARPETA_MODELO_SENTIMIENTO, "twitter_roberta_finetuned.zip", local_dir
    )

    ruta_extraido = local_dir / "modelo_sentimiento_finetuned"
    if not ruta_extraido.exists():
        with zipfile.ZipFile(ruta_zip, "r") as zf:
            zf.extractall(ruta_extraido)

    tokenizer = AutoTokenizer.from_pretrained(str(ruta_extraido))
    modelo = AutoModelForSequenceClassification.from_pretrained(str(ruta_extraido))
    modelo.eval()

    return tokenizer, modelo
