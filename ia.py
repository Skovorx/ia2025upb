import streamlit as st
import pandas as pd
import pickle

# Título principal
st.set_page_config(page_title="Predicción de Cl", layout="centered")
st.title("🧪 Predicción del Coeficiente de Sustentación (Cl)")

# --- Cargar modelo y scaler ---
st.header("1️⃣ Cargar modelo y scaler")

model_file = 'final_xgb_model.pkl'
scaler_file = 'scaler.pkl'

loaded_model = None
loaded_scaler = None
model_features = []
scaler_features = []

try:
    with open(model_file, 'rb') as f:
        loaded_model = pickle.load(f)
    if hasattr(loaded_model, 'feature_names_in_'):
        model_features = list(loaded_model.feature_names_in_)
    else:
        model_features = list(loaded_model.get_booster().feature_names)
    st.success("✅ Modelo cargado correctamente.")

    with open(scaler_file, 'rb') as f:
        loaded_scaler = pickle.load(f)
    if hasattr(loaded_scaler, 'feature_names_in_'):
        scaler_features = list(loaded_scaler.feature_names_in_)
    else:
        scaler_features = model_features.copy()
    st.success("✅ Scaler cargado correctamente.")

except FileNotFoundError as e:
    st.error(f"❌ No se encontró el archivo: {e.filename}")
except Exception as e:
    st.error(f"❌ Error cargando modelo o scaler: {e}")

# --- Cargar CSV de usuario ---
st.header("2️⃣ Cargar archivo CSV de entrada")
uploaded_file = st.file_uploader("📁 Sube un archivo `.csv` con los datos sin procesar", type="csv")

df_raw = None
if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.success("📄 Archivo CSV cargado correctamente.")
        st.subheader("🔍 Vista previa de los datos cargados")
        st.dataframe(df_raw)
    except Exception as e:
        st.error(f"❌ Error al leer el archivo CSV: {e}")

# --- Preprocesar y predecir ---
if df_raw is not None and loaded_model and loaded_scaler and model_features:
    st.header("3️⃣ Preprocesamiento y predicción")

    df_clean = df_raw.copy()

    # Eliminar columnas irrelevantes y la real 'Cl'
    df_clean.drop(columns=['z_te', 'dz_te', 'airfoil', 'Cl'], errors='ignore', inplace=True)

    # Asegurar tipo numérico
    df_clean = df_clean.apply(pd.to_numeric, errors='coerce')

    # Escalar características necesarias
    X_to_scale = df_clean.reindex(columns=scaler_features, fill_value=0)

    try:
        X_scaled = loaded_scaler.transform(X_to_scale)
        df_scaled = pd.DataFrame(X_scaled, columns=scaler_features, index=df_clean.index)
        df_clean.update(df_scaled)
    except Exception as e:
        st.error(f"❌ Error durante el escalado: {e}")
        df_scaled = X_to_scale  # Fallback sin escalar

    # Reordenar columnas para predicción
    X_input = df_clean.reindex(columns=model_features, fill_value=0).values

    # Predicción
    try:
        preds = loaded_model.predict(X_input)
        df_output = df_raw.copy()
        df_output = df_output.drop(columns='Cl', errors='ignore')  # eliminar Cl original
        df_output['Cl_predicho'] = preds

        st.subheader("📈 Resultado: Datos originales + Cl predicho")
        st.dataframe(df_output)

        # Permitir descarga
        csv = df_output.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descargar CSV con predicciones",
            data=csv,
            file_name='predicciones_con_Cl.csv',
            mime='text/csv'
        )
    except Exception as e:
        st.error(f"❌ Error al predecir: {e}")
else:
    st.info("🔄 Esperando que cargues el archivo y se carguen modelo y scaler.")
