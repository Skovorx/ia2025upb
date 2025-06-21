# prompt: haz todo el deployment anterior con streamlit cargando el modelo y el scaler, permite cargar un csv. ten encuenta esto Error loading or applying scaler: The feature names should match those that were passed during fit. Feature names must be in the same order as they were in fit.

import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


# Function to load the model and scaler
@st.cache_resource  # Cache the loading of model and scaler
def load_resources():
  """Loads the trained XGBoost model and the StandardScaler."""
  try:
    with open('final_xgb_model.pkl', 'rb') as model_file:
      model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
      scaler = pickle.load(scaler_file)
    return model, scaler
  except FileNotFoundError:
    st.error(
        "Error: Model or scaler files not found. Please ensure 'final_xgb_model.pkl' and 'scaler.pkl' are in the same directory as the app."
    )
    return None, None


# Load the model and scaler
model, scaler = load_resources()

# Streamlit app title
st.title("Airfoil Cl Prediction App")
st.write(
    "Upload a CSV file containing airfoil geometry and flow conditions to predict the Coefficient of Lift (Cl)."
)

if model is not None and scaler is not None:
  # File uploader
  uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

  if uploaded_file is not None:
    try:
      # Read the uploaded CSV file
      input_df = pd.read_csv(uploaded_file)

      # Display the original uploaded data
      st.subheader("Original Uploaded Data")
      st.dataframe(input_df.head())

      # --- Data Preprocessing (Matching the training process) ---
      # IMPORTANT: Ensure the columns match exactly and are in the same order as the training data
      # Identify expected columns (excluding 'Cl') based on the training code
      # We need to know the exact feature names and their order from the training phase.
      # Based on the provided training code, the features used after preprocessing were:
      # r_le, x_up_pt, z_up_pt, x_lo_pt, z_lo_pt, zxx_lo_pt, alpha_te, beta_te, alpha
      expected_features = [
          'r_le',
          'x_up_pt',
          'z_up_pt',
          'x_lo_pt',
          'z_lo_pt',
          'zxx_lo_pt',
          'alpha_te',
          'beta_te',
          'alpha',
      ]  # Adjust this list based on your final features used for training

      # Check if all expected features are in the uploaded file
      missing_cols = [col for col in expected_features if col not in input_df.columns]
      if missing_cols:
        st.error(
            f"Error: The uploaded CSV is missing the following required columns: {missing_cols}"
        )
      else:
        # Select and reorder columns to match the training data
        input_df_processed = input_df[expected_features].copy()

        # Apply the loaded scaler
        # The scaler expects the features in the exact order it was fitted on.
        try:
          input_scaled = scaler.transform(input_df_processed)
          input_scaled_df = pd.DataFrame(
              input_scaled, columns=input_df_processed.columns
          )
          st.subheader("Scaled Input Data (first 5 rows)")
          st.dataframe(input_scaled_df.head())

          # Make predictions
          predictions = model.predict(input_scaled_df)

          # Add predictions to the original dataframe
          input_df['Predicted_Cl'] = predictions

          # Display the results
          st.subheader("Predictions")
          st.dataframe(input_df)

          # Optionally, allow downloading the results
          @st.cache_data
          def convert_df_to_csv(df):
            # Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

          csv = convert_df_to_csv(input_df)

          st.download_button(
              label="Download results as CSV",
              data=csv,
              file_name='predicted_cl_results.csv',
              mime='text/csv',
          )

        except Exception as e:
          st.error(f"Error applying scaler or making predictions: {e}")
          st.write(
              "This might be due to a mismatch in feature names or their order between the uploaded data and the data used for training the scaler."
          )
          # You could inspect the scaler's feature names if possible (depends on scaler version/implementation)
          # if hasattr(scaler, 'feature_names_in_'):
          #     st.write(f"Scaler expects features: {scaler.feature_names_in_}")
          # elif hasattr(scaler, 'n_features_in_'):
          #      st.write(f"Scaler expects {scaler.n_features_in_} features.")
          st.write(f"Uploaded data columns: {input_df_processed.columns.tolist()}")

    except Exception as e:
      st.error(f"Error reading the CSV file: {e}")

