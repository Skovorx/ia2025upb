# prompt: haz todo el deployment anterior con streamlit, permite cargar un csv. ten encuenta esto Error loading or applying scaler: The feature names should match those that were passed during fit. Feature names must be in the same order as they were in fit.


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler # Ensure StandardScaler is imported

st.title("Airfoil Cl Prediction")

st.write("Upload your CSV file to get predictions for Cl.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df_pred = pd.read_csv(uploaded_file)

        # Load the scaler and model
        try:
            with open('scaler.pkl', 'rb') as f:
                loaded_scaler = pickle.load(f)
            with open('final_xgb_model.pkl', 'rb') as f:
                loaded_model = pickle.load(f)
        except FileNotFoundError:
            st.error("Error: Model or scaler file not found. Please ensure 'scaler.pkl' and 'final_xgb_model.pkl' are in the same directory.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading model or scaler: {e}")
            st.stop()

        # --- Data Preprocessing Steps (Matching the Colab notebook) ---

        # Ensure columns exist before dropping
        cols_to_drop_initial = ['z_te', 'dz_te']
        cols_to_drop_initial_exist = [col for col in cols_to_drop_initial if col in df_pred.columns]
        if cols_to_drop_initial_exist:
            df_pred = df_pred.drop(cols_to_drop_initial_exist, axis=1)
            st.write(f"Dropped columns: {cols_to_drop_initial_exist}")
        else:
            st.write("Initial columns 'z_te', 'dz_te' not found in the uploaded file.")


        # Handle 'alpha' column (assuming it exists and needs conversion)
        if 'alpha' in df_pred.columns:
             # Attempt conversion, handle errors
            try:
                df_pred['alpha'] = df_pred['alpha'].astype('float64')
            except ValueError:
                st.error("Error: 'alpha' column could not be converted to numeric type. Please check its format.")
                st.stop()
        else:
             st.warning("Warning: 'alpha' column not found in the uploaded file. This may affect predictions if it's a required feature.")


        # Outlier detection and removal based on 'airfoil' (if 'airfoil' exists)
        if 'airfoil' in df_pred.columns and 'Cl' in df_pred.columns:
             # This part assumes you want to remove rows based on the same logic
             # as in the original notebook. You might need to adjust if 'Cl'
             # isn't present in the prediction data or if the logic changes
             # for prediction vs training data.
             # For prediction, we typically don't remove rows based on target value outliers.
             # Let's skip this step for prediction data to avoid losing rows.
             st.write("Skipping outlier removal based on 'Cl' for prediction data.")

        # Drop 'airfoil' column if it exists
        if 'airfoil' in df_pred.columns:
            df_pred = df_pred.drop(['airfoil'], axis=1)
            st.write("Dropped 'airfoil' column.")
        else:
            st.write("Column 'airfoil' not found in the uploaded file.")


        # Outlier detection/handling on 'Cl' (if 'Cl' exists)
        if 'Cl' in df_pred.columns:
             st.write("Skipping outlier detection on 'Cl' as it's the target variable for prediction.")


        # Scaling the independent variables
        # Ensure the columns to scale match the columns used during scaler fitting.
        # The loaded_scaler has `feature_names_in_` attribute which stores the
        # column names it was fitted on.
        try:
            # Get the column names the scaler was fitted on
            scaled_columns_fitted = loaded_scaler.feature_names_in_

            # Check if all columns the scaler was fitted on are present in the uploaded dataframe
            missing_columns = [col for col in scaled_columns_fitted if col not in df_pred.columns]

            if missing_columns:
                st.error(f"Error: The uploaded CSV is missing the following columns that the model was trained on: {missing_columns}")
                st.stop()

            # Ensure the columns are in the same order as the scaler was fitted on
            df_pred_scaled = df_pred[scaled_columns_fitted]

            # Apply scaling
            df_pred_scaled[scaled_columns_fitted] = loaded_scaler.transform(df_pred_scaled[scaled_columns_fitted])
            st.write("Independent variables scaled successfully.")

        except AttributeError:
            st.error("Error: Could not get feature names from the loaded scaler. Ensure the scaler was fitted with feature names.")
            st.stop()
        except KeyError as e:
             st.error(f"Error accessing column {e}. Ensure your uploaded CSV has the correct column names.")
             st.stop()
        except Exception as e:
            st.error(f"Error during scaling: {e}")
            st.stop()


        # Make predictions
        try:
            predictions = loaded_model.predict(df_pred_scaled)
            df_pred['Predicted_Cl'] = predictions

            st.subheader("Predictions")
            st.write(df_pred)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
