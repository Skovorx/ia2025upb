# prompt: haz todo el deployment anterior con streamlit cargando el modelo y el scaler, permite cargar un csv. ten encuenta esto Error loading or applying scaler: The feature names should match those that were passed during fit. Feature names must be in the same order as they were in fit. los features esperados son airfoil,r_le,x_up_pt,z_up_pt,zxx_up_pt,x_lo_pt,z_lo_pt,zxx_lo_pt,z_te,dz_te,alpha_te,beta_te,alpha,Cl PIDE TODAS ESAS QUE TE ESTOY PIDIENDO


import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model and scaler
try:
    with open('final_xgb_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    st.success("Modelo y escalador cargados exitosamente.")
except FileNotFoundError:
    st.error("Archivos de modelo o escalador no encontrados. Asegúrate de que 'final_xgb_model.pkl' y 'scaler.pkl' existan en la ubicación correcta.")
    st.stop() # Stop the app if files are not found
except Exception as e:
    st.error(f"Error al cargar el modelo o escalador: {e}")
    st.stop()


st.title("Predicción de Coeficiente de Sustentación (Cl)")

st.write("""
Esta aplicación predice el coeficiente de sustentación (Cl) de un perfil alar
basado en sus parámetros geométricos y el ángulo de ataque.
""")

# --- Input Options ---
input_option = st.radio(
    "Selecciona una opción para ingresar los datos:",
    ("Ingresar datos manualmente", "Cargar archivo CSV")
)

# Define the expected features based on the model training
expected_features = ['airfoil','r_le', 'x_up_pt', 'z_up_pt', 'zxx_up_pt', 'x_lo_pt', 'z_lo_pt', 'zxx_lo_pt', 'alpha_te', 'beta_te', 'alpha']

input_df = None

if input_option == "Ingresar datos manualmente":
    st.header("Ingresa los parámetros del perfil alar:")

    input_data = {}
    for feature in expected_features:
        # Add specific input types or ranges if known
        if feature in ['r_le', 'x_up_pt', 'z_up_pt', 'zxx_up_pt', 'x_lo_pt', 'z_lo_pt', 'zxx_lo_pt', 'alpha_te', 'beta_te']:
             input_data[feature] = st.number_input(f"{feature}", format="%.6f")
        elif feature == 'alpha':
             input_data[feature] = st.number_input(f"{feature}", format="%.2f") # Angle of attack

    input_df = pd.DataFrame([input_data])

elif input_option == "Cargar archivo CSV":
    st.header("Carga tu archivo CSV")
    uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.write("Vista previa del archivo cargado:")
            st.write(uploaded_df.head())

            # Validate columns in the uploaded CSV
            missing_cols = [col for col in expected_features if col not in uploaded_df.columns]
            if missing_cols:
                st.error(f"El archivo CSV debe contener las siguientes columnas: {expected_features}. Faltan: {missing_cols}")
                input_df = None # Invalidate input_df if columns are missing
            else:
                # Ensure the column order matches the training data
                input_df = uploaded_df[expected_features].copy()
                st.success("Archivo CSV cargado y validado exitosamente.")

        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {e}")
            input_df = None # Invalidate input_df on error


# --- Prediction ---
if input_df is not None and st.button("Predecir Cl"):
    try:
        # Apply the scaler
        # Check if the scaler expects feature names
        if hasattr(scaler, 'feature_names_in_'):
            # Align columns with the scaler's expected features before scaling
            # This handles potential column order issues or extra columns in uploaded CSV
            input_df_scaled = scaler.transform(input_df[scaler.feature_names_in_])
            input_df_scaled = pd.DataFrame(input_df_scaled, columns=scaler.feature_names_in_)
        else:
             # Assume scaler expects columns in the defined expected_features order
             input_df_scaled = scaler.transform(input_df[expected_features])
             input_df_scaled = pd.DataFrame(input_df_scaled, columns=expected_features)


        # Make predictions
        predictions = model.predict(input_df_scaled)

        st.header("Resultado de la Predicción:")

        if input_option == "Ingresar datos manualmente":
            st.write(f"El coeficiente de sustentación (Cl) predicho es: **{predictions[0]:.4f}**")
        elif input_option == "Cargar archivo CSV":
            results_df = input_df.copy()
            results_df['Predicted_Cl'] = predictions
            st.write("Predicciones para los datos cargados:")
            st.dataframe(results_df)

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        st.error(f"Asegúrate de que las columnas y el orden de los datos de entrada coincidan con el entrenamiento.")
