import os
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Archivos por defecto
DATA_FILE = 'combined.csv'
MODEL_FILE = 'final_xgb_model.pkl'
SCALER_FILE = 'scaler.pkl'

# 0. Cargar el modelo y el scaler
loaded_model = None
loaded_scaler = None
try:
    with open(MODEL_FILE, 'rb') as f:
        loaded_model = pickle.load(f)
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
    # Si el usuario sube un archivo
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.success("Archivo cargado exitosamente.")
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
elif os.path.exists(DATA_FILE):
    # Si no subió, intentamos cargar combined.csv
    try:
        df_raw = pd.read_csv(DATA_FILE, encoding='utf-8')
        st.info(f"Cargando datos por defecto desde {DATA_FILE}.")
    except Exception as e:
        st.error(f"Error al leer {DATA_FILE}: {e}")
else:
    st.warning(f"Sube un CSV o coloca {DATA_FILE} en el directorio de la aplicación.")

# Mostrar datos sin procesar si se cargaron
if df_raw is not None:
    st.subheader("Datos cargados (sin procesar)")
    st.dataframe(df_raw)

# 2. Preprocesamiento y predicción
if df_raw is not None and loaded_model is not None and loaded_scaler is not None:
    st.header("2. Procesar datos y predecir")
    try:
        df_processed = df_raw.copy()

        # Eliminar columnas no necesarias sin error si faltan
        cols_to_drop = ['z_te', 'dz_te', 'airfoil']
        df_processed = df_processed.drop(columns=cols_to_drop, errors='ignore')

        # Asegurar tipo float
        if 'alpha' in df_processed.columns:
            df_processed['alpha'] = pd.to_numeric(df_processed['alpha'], errors='coerce')

        # Definir columnas para escalar
        original_scaled_cols = [
            'r_le', 'x_up_pt', 'z_up_pt', 'zxx_up_pt',
            'x_lo_pt', 'z_lo_pt', 'zxx_lo_pt',
            'alpha_te', 'beta_te', 'alpha'
        ]
        # Seleccionar las presentes
        cols_for_scaling = [c for c in original_scaled_cols if c in df_processed.columns]
        missing = set(original_scaled_cols) - set(cols_for_scaling)
        if missing:
            st.warning(f"Faltan columnas para escalar: {missing}. Se escalán solo las disponibles.")

        # Escalar solo si hay columnas
        if cols_for_scaling:
            df_processed[cols_for_scaling] = loaded_scaler.transform(df_processed[cols_for_scaling])

        st.subheader("Datos preprocesados")
        st.dataframe(df_processed)

        # Preparar para predicción: usar solo las columnas esperadas
        expected_cols = original_scaled_cols
        X_predict = df_processed.reindex(columns=expected_cols, fill_value=0)

        # Predecir
        predictions = loaded_model.predict(X_predict)
        df_predictions = pd.DataFrame(predictions, columns=['Cl_Predicho'])

        st.subheader("Predicciones")
        st.dataframe(df_predictions)

        # Unir predicciones con datos originales
        df_out = df_raw.copy()
        df_out['Cl_Predicho'] = df_predictions['Cl_Predicho'].values
        st.subheader("Datos originales con predicción")
        st.dataframe(df_out)

    except Exception as e:
        st.error(f"Error durante preprocesamiento o predicción: {e}")
else:
    if df_raw is None:
        st.info("Esperando CSV...")
    else:
        st.info("Modelo o scaler no disponibles para predecir.")
