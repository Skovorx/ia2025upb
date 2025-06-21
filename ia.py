# prompt: Carga el modelo y el scaler, con streamlit permite que con una interfaz gráfica permita cargar un .csv sin procesar, luego aplica todo lo que se ha hecho en el collab, aplica el modelo y el scaler y luego predice. Crea una subseccion donde se visualizen los dato cargados, los datos preprocesados, y luego las predicciones, donde al lado de la matriz sin procesar agrega uan columna con la prediccion

import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import os

# Cargar el modelo y el scaler serializados
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

# Sección de carga de archivo
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

# Sección de preprocesamiento y predicción
if df_raw is not None and loaded_model is not None and loaded_scaler is not None:
    st.header("2. Procesar datos y predecir")

    # Aplicar preprocesamiento
    try:
        df_processed = df_raw.copy()

        # Eliminar columnas irrelevantes (si existen)
        cols_to_drop = ['z_te', 'dz_te', 'airfoil']
        for col in cols_to_drop:
            if col in df_processed.columns:
                df_processed = df_processed.drop(col, axis=1)

        # Convertir 'alpha' a float si existe
        if 'alpha' in df_processed.columns:
             df_processed['alpha'] = df_processed['alpha'].astype('float64')

        # Asumimos que el scaler fue entrenado con todas las columnas excepto 'Cl'.
        # Identificamos las columnas del df_processed (sin Cl, si estuviera)
        # que se usaron para entrenar el scaler.
        # Las columnas escaladas en el notebook original fueron:
        # 'r_le', 'x_up_pt', 'z_up_pt', 'x_lo_pt', 'z_lo_pt', 'zxx_lo_pt', 'alpha_te', 'beta_te', 'alpha'

        # Asegurarnos de que las columnas a escalar existan en el df_processed
        # y que tengan el mismo orden que las columnas usadas para entrenar el scaler.
        # Una forma robusta es obtener las columnas del scaler si fuera posible,
        # pero como no guardamos el order, lo definimos explicitamente basado en el notebook.
        original_scaled_cols = ['airfoil','r_le', 'x_up_pt', 'z_up_pt', 'zxx_up_pt', 'x_lo_pt', 'z_lo_pt', 'zxx_lo_pt', 'alpha_te', 'beta_te', 'alpha']
        cols_for_scaling_in_input = [col for col in original_scaled_cols if col in df_processed.columns]

        # Verificar si las columnas necesarias para el scaler están presentes
        if len(cols_for_scaling_in_input) != len(original_scaled_cols):
             missing_cols = set(original_scaled_cols) - set(cols_for_scaling_in_input)
             st.warning(f"Advertencia: Faltan algunas columnas esperadas para el escalado en el archivo cargado: {missing_cols}. El escalado se aplicará a las columnas presentes.")

        # Aplicar escalado
        if cols_for_scaling_in_input:
            # Asegurarnos de que el orden de las columnas para escalar sea el mismo que el scaler espera
            # Nota: StandardScaler no guarda el nombre de las columnas.
            # Necesitamos confiar en que el orden de las columnas en el df_processed para escalar
            # sea el mismo que el orden de las columnas que se usaron para entrenar el scaler.
            # Si el df_processed tiene columnas extra o en diferente orden (excepto Cl),
            # el escalado podría ser incorrecto. Asumimos que el input tiene las columnas
            # necesarias (menos las eliminadas) y las ordena para el scaler.
            # Un enfoque más robusto guardaría los nombres de las columnas junto con el scaler.

            # Crear un dataframe solo con las columnas a escalar para el scaler
            df_to_scale = df_processed[cols_for_scaling_in_input]

            # Aplicar transform
            df_processed[cols_for_scaling_in_input] = loaded_scaler.transform(df_to_scale)

        st.subheader("Datos preprocesados (escalados y columnas eliminadas)")
        st.dataframe(df_processed)

        # Asegurarse de que las columnas del dataframe preprocesado coincidan con las columnas de entrenamiento
        # Es crucial que las columnas del dataframe de entrada coincidan exactamente con las que el modelo espera.
        # El modelo XGBoost fue entrenado con X = df.drop(columns="Cl"). Necesitamos obtener esas columnas exactas.
        # Como no guardamos explicitamente la lista de X.columns en el notebook,
        # la reconstruimos basándonos en el código original de preprocesamiento.

        # Columnas originales después de eliminar 'z_te', 'dz_te', 'airfoil' y antes de escalar
        # Esto asume que el archivo de entrada original ('combined.csv') tenía todas estas columnas
        # y que solo se eliminaron esas 3.
        original_training_cols_before_scaling = [col for col in df_raw.columns if col not in ['z_te', 'dz_te', 'airfoil', 'Cl']]

        # Columnas que el modelo final espera DEBEN estar en el mismo orden que X_final.columns
        # en el notebook. X_final era df.drop(columns="Cl") DESPUÉS de todo el preprocesamiento (eliminación y escalado).
        # Las columnas escaladas fueron ['r_le', 'x_up_pt', 'z_up_pt', 'x_lo_pt', 'z_lo_pt', 'zxx_lo_pt', 'alpha_te', 'beta_te', 'alpha']
        # Las no escaladas (y no eliminadas) serían las demás. En el código original, solo 'Cl' se excluía de la lista de escalado.
        # Esto implica que TODAS las columnas restantes (después de eliminar irrelevantes) fueron escaladas.
        # Verificamos las columnas del df_processed después de la eliminación.
        expected_model_cols = list(df_processed.columns) # Esto asume que df_processed ya tiene solo las columnas correctas y escaladas.

        # Asegurarse de que el dataframe de entrada para la predicción tenga las columnas en el orden correcto
        # XGBoost espera que las características de entrada estén en el mismo orden que durante el entrenamiento.
        # La forma más segura es reordenar las columnas del df_processed según el orden de X_final.columns.
        # Como X_final.columns no fue guardado, usamos las columnas de df_processed asumiendo que el preprocesamiento
        # en Streamlit replicó el orden final del notebook.

        X_predict = df_processed[expected_model_cols] # Usamos las columnas en el orden de df_processed


        # Realizar predicciones
        predictions = loaded_model.predict(X_predict)

        st.subheader("Predicciones")
        # Convertir predicciones a DataFrame para mejor visualización
        df_predictions = pd.DataFrame(predictions, columns=['Cl_Predicho'])
        st.dataframe(df_predictions)

        # Mostrar tabla original con predicciones añadidas
        df_raw_with_predictions = df_raw.copy()
        # Asegurarse de que el número de filas coincide
        if len(df_raw_with_predictions) == len(df_predictions):
            df_raw_with_predictions['Cl_Predicho'] = df_predictions['Cl_Predicho']
            st.subheader("Datos originales con Predicciones")
            st.dataframe(df_raw_with_predictions)
        else:
            st.warning("El número de filas en el archivo cargado no coincide con el número de predicciones.")


    except Exception as e:
        st.error(f"Error durante el preprocesamiento o la predicción: {e}")

