# prompt: haz todo el deployment anterior con streamlit cargando el modelo y el scaler, permite cargar un csv. ten encuenta esto Error loading or applying scaler: The feature names should match those that were passed during fit. Feature names must be in the same order as they were in fit. r_le,x_up_pt,z_up_pt,zxx_up_pt,x_lo_pt,z_lo_pt,zxx_lo_pt,alpha_te,beta_te,alpha,Cl


import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model and scaler
try:
  with open('final_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
  with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
except FileNotFoundError:
  st.error("Error: Make sure 'final_xgb_model.pkl' and 'scaler.pkl' are in the same directory.")
  st.stop()

st.title('Predicción del Coeficiente de Sustentación (Cl)')

st.write("""
Esta aplicación predice el coeficiente de sustentación (Cl) de un perfil aerodinámico
basado en sus parámetros geométricos y de flujo, utilizando un modelo XGBoost entrenado.
""")

uploaded_file = st.file_uploader("Carga tu archivo CSV con los datos de los perfiles aerodinámicos", type=["csv"])

if uploaded_file is not None:
  try:
    input_df = pd.read_csv(uploaded_file)

    # Drop irrelevant columns if they exist
    columns_to_drop_if_exist = ['airfoil', 'z_te', 'dz_te']
    input_df = input_df.drop(columns=[col for col in columns_to_drop_if_exist if col in input_df.columns])

    # Define expected features based on the training data (excluding 'Cl')
    expected_features = ['r_le', 'x_up_pt', 'z_up_pt', 'x_lo_pt', 'z_lo_pt', 'zxx_lo_pt', 'alpha_te', 'beta_te', 'alpha']

    # Check if all expected features are in the uploaded file
    if not all(feature in input_df.columns for feature in expected_features):
        missing_features = [feature for feature in expected_features if feature not in input_df.columns]
        st.error(f"Error: The uploaded CSV is missing the following required columns: {', '.join(missing_features)}")
        st.stop()

    # Ensure columns are in the same order as during training
    input_df = input_df[expected_features]

    # Display the uploaded data
    st.subheader('Datos cargados')
    st.write(input_df.head())

    # Apply the scaler
    try:
        scaled_input_data = scaler.transform(input_df)
        scaled_input_df = pd.DataFrame(scaled_input_data, columns=input_df.columns) # Keep column names
        st.subheader('Datos escalados')
        st.write(scaled_input_df.head())
    except Exception as e:
        st.error(f"Error applying scaler: {e}")
        st.stop()

    # Make predictions
    try:
      prediction = model.predict(scaled_input_df)
      input_df['Predicted Cl'] = prediction
      st.subheader('Predicciones')
      st.write(input_df[['Predicted Cl']]) # Show only the original data plus prediction
    except Exception as e:
      st.error(f"Error making predictions: {e}")

  except Exception as e:
    st.error(f"Error processing the uploaded file: {e}")

else:
  st.info("Por favor, carga un archivo CSV para realizar predicciones.")

# You can add more components like:
# - Input fields for single prediction
# - Visualizations of results
# - Download button for the predictions

