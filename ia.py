# prompt: haz todo el deployment anterior con streamlit, permite cargar un csv

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler # Need to import if using scaler

st.title('Cl Prediction App')

st.write("""
This app predicts the lift coefficient (Cl) of an airfoil based on geometric and flow parameters.
""")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        # --- Data Preprocessing (Mirroring the notebook steps) ---

        # 1. Drop irrelevant columns (ensure these columns exist in the input CSV)
        cols_to_drop_initial = ['z_te', 'dz_te']
        input_df = input_df.drop(columns=[col for col in cols_to_drop_initial if col in input_df.columns], errors='ignore')

        # 2. Convert 'alpha' to float64 (ensure 'alpha' exists)
        if 'alpha' in input_df.columns:
             input_df['alpha'] = input_df['alpha'].astype('float64')
        else:
            st.warning("Column 'alpha' not found in the uploaded CSV. Skipping type conversion.")


        # 3. Remove rows/profiles with extreme Cl (this step is based on the *original* data analysis)
        # In a real-world deployment for prediction, you wouldn't remove rows based on the target
        # variable of the *new* input data. However, since the original notebook removed
        # entire 'airfoil' profiles based on this condition, and the 'airfoil' column
        # is dropped later, we'll skip this step for prediction on *new* data.
        # If 'airfoil' was an input feature used for prediction, this would be more complex.
        if 'airfoil' in input_df.columns:
             st.write("Skipping 'airfoil' outlier removal as it's typically not done on new prediction data.")
             input_df = input_df.drop(['airfoil'], axis=1, errors='ignore') # Drop the column if it exists
        else:
             st.write("Column 'airfoil' not found in the uploaded CSV.")


        # 4. Scaling (Use the pre-trained scaler)
        try:
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)

            # Identify columns to scale (all except 'Cl' if it exists, or all features)
            # If the input CSV *might* contain 'Cl' (e.g., for comparison), exclude it.
            # Otherwise, scale all columns of the input data.
            columns_to_scale = input_df.columns.tolist()
            if 'Cl' in columns_to_scale:
                 columns_to_scale.remove('Cl') # Assuming 'Cl' might be present but is the target

            # Apply the scaling
            if columns_to_scale: # Check if there are columns left to scale
                 input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])
            else:
                 st.warning("No columns left to scale after considering 'Cl'. Check input data.")

        except FileNotFoundError:
            st.error("Scaler file 'scaler.pkl' not found. Please ensure it's in the same directory.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading or applying scaler: {e}")
            st.stop()


        st.subheader('Processed Input Data')
        st.write(input_df.head())

        # --- Load the trained model ---
        try:
            with open('final_xgb_model.pkl', 'rb') as f:
                model = pickle.load(f)
        except FileNotFoundError:
            st.error("Model file 'final_xgb_model.pkl' not found. Please ensure it's in the same directory.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading the model: {e}")
            st.stop()

        # --- Make Predictions ---
        try:
            prediction = model.predict(input_df)

            st.subheader('Prediction')
            # Display predictions, perhaps alongside the original data if helpful
            result_df = input_df.copy() # Use the processed input data
            result_df['Predicted_Cl'] = prediction
            st.write(result_df[['Predicted_Cl']].head()) # Show only the prediction initially
            st.write(result_df) # Or show the full processed data + prediction


        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()


    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")

else:
    st.info('Awaiting for CSV file to be uploaded.')
