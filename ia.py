import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Cargar el modelo y el scaler
try:
    with open('final_xgb_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
    st.success("Modelo y Scaler cargados correctamente.")
except FileNotFoundError:
    st.error("Error: Asegúrate de que los archivos 'final_xgb_model.pkl' y 'scaler.pkl' existen.")
    loaded_model = None
    loaded_scaler = None

st.title("Predicción de Coeficiente de Sustentación (Cl)")
st.write("Carga un archivo CSV sin procesar, preprocesaremos los datos, aplicaremos el modelo entrenado y mostraremos las predicciones.")

# 1. Carga del archivo
st.header("1. Cargar archivo CSV")
uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

df_raw = None
if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.success("Archivo cargado exitosamente.")
        st.subheader("Datos cargados (sin procesar)")
        st.dataframe(df_raw)
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")

# 2. Preprocesamiento y predicción
if df_raw is not None and loaded_model is not None and loaded_scaler is not None:
    st.header("2. Procesar datos y predecir")

    try:
        df_processed = df_raw.copy()

        # Eliminar columnas no necesarias
        cols_to_drop = ['z_te', 'dz_te', 'airfoil']
        df_processed = df_processed.drop(columns=[col for col in cols_to_drop if col in df_processed.columns])

        # Asegurarse que 'alpha' sea float
        if 'alpha' in df_processed.columns:
            df_processed['alpha'] = df_processed['alpha'].astype('float64')

        # Columnas usadas para escalar (sin 'airfoil')
        original_scaled_cols = ['r_le', 'x_up_pt', 'z_up_pt', 'zxx_up_pt', 
                                'x_lo_pt', 'z_lo_pt', 'zxx_lo_pt', 
                                'alpha_te', 'beta_te', 'alpha']

        # Verificar columnas disponibles para escalar
        cols_for_scaling_in_input = [col for col in original_scaled_cols if col in df_processed.columns]
        if len(cols_for_scaling_in_input) != len(original_scaled_cols):
            missing_cols = set(original_scaled_cols) - set(cols_for_scaling_in_input)
            st.warning(f"Advertencia: Faltan columnas esperadas para el escalado: {missing_cols}. Se escalarán solo las presentes.")

        # Escalar las columnas disponibles
        if cols_for_scaling_in_input:
            df_to_scale = df_processed[cols_for_scaling_in_input]
            df_processed[cols_for_scaling_in_input] = loaded_scaler.transform(df_to_scale)

        st.subheader("Datos preprocesados")
        st.dataframe(df_processed)

        # Ordenar columnas como se usaron durante el entrenamiento
        expected_model_cols = ['r_le', 'x_up_pt', 'z_up_pt', 'zxx_up_pt', 
                               'x_lo_pt', 'z_lo_pt', 'zxx_lo_pt', 
                               'alpha_te', 'beta_te', 'alpha']
        X_predict = df_processed[expected_model_cols]

        # Predicciones
        predictions = loaded_model.predict(X_predict)
        df_predictions = pd.DataFrame(predictions, columns=['Cl_Predicho'])

        st.subheader("Predicciones")
        st.dataframe(df_predictions)

        # Datos originales con predicción añadida
        df_raw_with_predictions = df_raw.copy()
        if len(df_raw_with_predictions) == len(df_predictions):
            df_raw_with_predictions['Cl_Predicho'] = df_predictions['Cl_Predicho']
            st.subheader("Datos originales con Predicciones")
            st.dataframe(df_raw_with_predictions)
        else:
            st.warning("El número de filas no coincide con el número de predicciones.")

    except Exception as e:
        st.error(f"Error durante el preprocesamiento o la predicción: {e}")
