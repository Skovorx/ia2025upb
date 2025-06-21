# prompt: haz todo el deployment anterior con streamlit cargando el modelo y el scaler, permite cargar un csv. ten encuenta esto Error loading or applying scaler: The feature names should match those that were passed during fit. Feature names must be in the same order as they were in fit. los features esperados son airfoil,r_le,x_up_pt,z_up_pt,zxx_up_pt,x_lo_pt,z_lo_pt,zxx_lo_pt,z_te,dz_te,alpha_te,beta_te,alpha,Cl


import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Define expected feature names in the correct order
EXPECTED_FEATURES = [
    'r_le',
    'x_up_pt',
    'z_up_pt',
    'zxx_up_pt',
    'x_lo_pt',
    'z_lo_pt',
    'zxx_lo_pt',
    'alpha_te',
    'beta_te',
    'alpha'
]

@st.cache_resource
def load_model():
  """Loads the trained XGBoost model."""
  try:
    with open('final_xgb_model.pkl', 'rb') as f:
      model = pickle.load(f)
    return model
  except FileNotFoundError:
    st.error("Model file 'final_xgb_model.pkl' not found. Please ensure it's in the same directory.")
    return None
  except Exception as e:
    st.error(f"Error loading model: {e}")
    return None

@st.cache_resource
def load_scaler():
  """Loads the trained StandardScaler."""
  try:
    with open('scaler.pkl', 'rb') as f:
      scaler = pickle.load(f)
    return scaler
  except FileNotFoundError:
    st.error("Scaler file 'scaler.pkl' not found. Please ensure it's in the same directory.")
    return None
  except Exception as e:
    st.error(f"Error loading scaler: {e}")
    return None

# Load the model and scaler
model = load_model()
scaler = load_scaler()

st.title('Predicción de Coeficiente de Sustentación (Cl)')

st.write("""
Este aplicativo predice el coeficiente de sustentación (Cl) basado en las
características geométricas de un perfil aerodinámico y su ángulo de ataque.
""")

uploaded_file = st.file_uploader("Cargar archivo CSV de perfiles aerodinámicos", type="csv")

if uploaded_file is not None:
  try:
    input_df = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.subheader('Datos cargados:')
    st.write(input_df.head())

    if model is not None and scaler is not None:
      # --- Data Preprocessing (Matching the notebook steps) ---

      # 1. Drop irrelevant columns (if they exist in the input)
      input_df = input_df.drop(['z_te', 'dz_te', 'airfoil'], axis=1, errors='ignore')

      # 2. Ensure 'alpha' is float64 (if it exists)
      if 'alpha' in input_df.columns:
          input_df['alpha'] = input_df['alpha'].astype('float64')

      # 3. Handle missing columns (if any) - fill with a placeholder or raise error
      # For simplicity here, we'll check if all EXPECTED_FEATURES are present
      if not all(feature in input_df.columns for feature in EXPECTED_FEATURES):
          missing_features = [feature for feature in EXPECTED_FEATURES if feature not in input_df.columns]
          st.error(f"El archivo CSV cargado debe contener las siguientes columnas: {EXPECTED_FEATURES}. Faltan: {missing_features}")
      else:
          # 4. Reorder columns to match the training data and select only expected features
          input_df_processed = input_df[EXPECTED_FEATURES].copy()

          # 5. Apply the loaded scaler
          # Make sure the scaler was fitted on data with the same number and order of features
          try:
              input_scaled = scaler.transform(input_df_processed)
              input_scaled_df = pd.DataFrame(input_scaled, columns=EXPECTED_FEATURES) # Keep column names
          except ValueError as e:
               st.error(f"Error aplicando el scaler: {e}. Asegúrese de que las columnas y su orden en el CSV cargado coinciden con las usadas para entrenar el scaler.")
               st.info(f"Columnas esperadas por el scaler: {EXPECTED_FEATURES}")
               st.info(f"Columnas en el CSV cargado (después de limpieza): {input_df_processed.columns.tolist()}")
               st.stop() # Stop execution if scaling fails

          # 6. Make predictions
          predictions = model.predict(input_scaled_df)

          # Display predictions
          st.subheader('Predicciones de Cl:')
          results_df = input_df_processed.copy() # Display predictions alongside input features
          results_df['Predicted_Cl'] = predictions
          st.write(results_df)

  except Exception as e:
    st.error(f"Ocurrió un error durante el procesamiento del archivo: {e}")
else:
  st.info("Por favor, cargue un archivo CSV para obtener predicciones.")
