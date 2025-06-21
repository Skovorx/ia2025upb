import os
import streamlit as st
import pandas as pd
import pickle

# Archivos por defecto
DATA_FILE = 'combined.csv'
MODEL_FILE = 'final_xgb_model.pkl'
SCALER_FILE = 'scaler.pkl'

# 0. Cargar el modelo y el scaler
loaded_model = None
loaded_scaler = None
feature_names = []
try:
    with open(MODEL_FILE, 'rb') as f:
        loaded_model = pickle.load(f)
    # Extraer nombres de características del modelo (sklearn API)
    if hasattr(loaded_model, 'feature_names_in_'):
        feature_names = list(loaded_model.feature_names_in_)
    else:
        # Para XGBoost puro
        try:
            feature_names = list(loaded_model.get_booster().feature_names)
        except Exception:
            feature_names = []
    with open(SCALER_FILE, 'rb') as f:
        loaded_scaler = pickle.load(f)
    st.success("Modelo y Scaler cargados correctamente.")
except FileNotFoundError as e:
    st.error(f"Error: no se encontró {e.filename}. Asegúrate de que los archivos existen.")

st.title("Predicción de Coeficiente de Sustentación (Cl)")
st.write("Carga un archivo CSV sin procesar (o usa 'combined.csv'), preprocesamos los datos, aplicamos el modelo entrenado y mostramos las predicciones.")

# 1. Carga del archivo
st.header("1. Cargar archivo CSV")
uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

df_raw = None
if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.success("Archivo cargado exitosamente.")
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
elif os.path.exists(DATA_FILE):
    try:
        df_raw = pd.read_csv(DATA_FILE, encoding='utf-8')
        st.info(f"Cargando datos por defecto desde {DATA_FILE}.")
    except Exception as e:
        st.error(f"Error al leer {DATA_FILE}: {e}")
else:
    st.warning(f"Sube un CSV o coloca {DATA_FILE} en el directorio de la aplicación.")

if df_raw is not None:
    st.subheader("Datos cargados (sin procesar)")
    st.dataframe(df_raw)

# 2. Preprocesamiento y predicción
if df_raw is not None and loaded_model is not None and loaded_scaler is not None and feature_names:
    st.header("2. Procesar datos y predecir")
    try:
        df_processed = df_raw.copy()

        # Eliminar columnas no necesarias
        cols_to_drop = ['z_te', 'dz_te', 'airfoil']
        df_processed.drop(columns=cols_to_drop, errors='ignore', inplace=True)

        # Convertir a numérico
        for col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        # Escalar columnas que existan
        cols_for_scaling = [c for c in feature_names if c in df_processed.columns]
        missing = set(feature_names) - set(cols_for_scaling)
        if missing:
            st.warning(f"Faltan columnas para escalar o predecir: {missing}. Se usarán ceros para ellas.")

        if cols_for_scaling:
            df_processed[cols_for_scaling] = loaded_scaler.transform(df_processed[cols_for_scaling])

        st.subheader("Datos preprocesados")
        st.dataframe(df_processed)

        # Preparar DataFrame para predicción con orden y nombres correctos
        X_predict = pd.DataFrame(
            data=0,
            index=df_processed.index,
            columns=feature_names
        )
        for col in feature_names:
            if col in df_processed.columns:
                X_predict[col] = df_processed[col]

        # Predecir
        predictions = loaded_model.predict(X_predict)
        df_predictions = pd.DataFrame(predictions, columns=['Cl_Predicho'], index=df_processed.index)

        st.subheader("Predicciones")
        st.dataframe(df_predictions)

        # Unir resultados
        df_out = df_raw.copy()
        df_out['Cl_Predicho'] = df_predictions['Cl_Predicho']
        st.subheader("Datos originales con predicción")
        st.dataframe(df_out)

    except Exception as e:
        st.error(f"Error durante preprocesamiento o predicción: {e}")
else:
    if df_raw is None:
        st.info("Esperando CSV...")
    elif not feature_names:
        st.info("No se pudieron obtener nombres de características del modelo.")
    else:
        st.info("Modelo o scaler no disponibles para predecir.")
