import os
import streamlit as st
import pandas as pd
import pickle

# Archivos por defecto
data_file = 'combined.csv'
model_file = 'final_xgb_model.pkl'
scaler_file = 'scaler.pkl'

# Cargar modelo y scaler
loaded_model = None
loaded_scaler = None
model_features = []
scaler_features = []
try:
    with open(model_file, 'rb') as f:
        loaded_model = pickle.load(f)
    # Obtener nombres de features del modelo
    if hasattr(loaded_model, 'feature_names_in_'):
        model_features = list(loaded_model.feature_names_in_)
    else:
        try:
            model_features = list(loaded_model.get_booster().feature_names)
        except Exception:
            st.error("No se pudieron extraer nombres de características del modelo.")

    with open(scaler_file, 'rb') as f:
        loaded_scaler = pickle.load(f)
    # Obtener nombres de features del scaler
    if hasattr(loaded_scaler, 'feature_names_in_'):
        scaler_features = list(loaded_scaler.feature_names_in_)
    else:
        scaler_features = model_features.copy()
    st.success("Modelo y scaler cargados correctamente.")
except FileNotFoundError as e:
    st.error(f"Error: no se encontró {e.filename}.")

st.title("Predicción de Cl")

# Cargar datos
uploaded_file = st.file_uploader("Sube un CSV", type='csv')
if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.success("CSV cargado.")
    except Exception as e:
        st.error(f"Error leyendo CSV: {e}")
elif os.path.exists(data_file):
    try:
        df_raw = pd.read_csv(data_file)
        st.info(f"Usando {data_file}.")
    except Exception as e:
        st.error(f"Error leyendo {data_file}: {e}")
else:
    st.warning(f"Sube un CSV o coloca {data_file} en la carpeta.")
    df_raw = None

if df_raw is not None:
    st.dataframe(df_raw)

# Preprocesar y predecir
if df_raw is not None and loaded_model and loaded_scaler and model_features:
    df = df_raw.copy()
    # Eliminar columnas irrelevantes
    df.drop(columns=['z_te', 'dz_te', 'airfoil'], errors='ignore', inplace=True)
    # Convertir a numérico
    df = df.apply(pd.to_numeric, errors='coerce')

    # Escalar solo las columnas esperadas por el scaler
    X_scale = df.reindex(columns=scaler_features, fill_value=0)
    try:
        X_scaled = loaded_scaler.transform(X_scale)
    except Exception as e:
        st.error(f"Error en escalado: {e}")
        X_scaled = X_scale.values

    # Reemplazar en df
    df_scaled = pd.DataFrame(X_scaled, columns=scaler_features, index=df.index)
    df.update(df_scaled)

    # Preparar datos para predicción
    X_pred = df.reindex(columns=model_features, fill_value=0)

    # Convertir a numpy para evitar mismatch de nombres
    X_input = X_pred.values

    # Predecir
    try:
        preds = loaded_model.predict(X_input)
        df_out = df_raw.copy()
        df_out['Cl_predicho'] = preds
        st.write(df_out)
    except Exception as e:
        st.error(f"Error en predict(): {e}")
else:
    st.info("Esperando archivos y modelo.")
