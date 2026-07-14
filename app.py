import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load the trained model and OneHotEncoder ---
try:
    model = joblib.load("car_price_decision_tree_model.pkl")
    ohe = joblib.load("encoder.pkl")   # Change this if your encoder has a different name
    st.success("Model and encoder loaded successfully!")
except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}")
    st.info("Make sure both the model and encoder files are uploaded to the GitHub repository.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or encoder: {e}")
    st.stop()
# --- 2. Streamlit App Layout ---
st.set_page_config(page_title="Used Car Price Predictor", layout="centered")
st.title("🚗 Used Car Price Prediction")
st.write("Enter the car's features to get a predicted selling price.")
st.markdown("---")

# --- 3. Input Fields for User Data ---
st.header("Car Details")

# Numeric Inputs
vehicle_age = st.slider("Vehicle Age (years)", 0, 30, 5)
km_driven = st.number_input("Kilometers Driven", 0, 500000, 50000, step=1000)
mileage = st.number_input("Mileage (kmpl)", 0.0, 50.0, 15.0, step=0.1)
engine = st.number_input("Engine Capacity (CC)", 500, 6000, 1500, step=50)
max_power = st.number_input("Max Power (BHP)", 10.0, 600.0, 80.0, step=0.5)
seats = st.slider("Number of Seats", 2, 10, 5)

st.markdown("--- Expansion for Categorical Features ---")

# Categorical Inputs (using categories from the loaded encoder)
# Access categories from the OneHotEncoder object
# The order of categories_ corresponds to the order of original_categorical_cols
# ['brand', 'seller_type', 'fuel_type', 'transmission_type']

# Ensure these match the order the encoder was fitted with
original_categorical_cols = ['brand', 'seller_type', 'fuel_type', 'transmission_type']

if len(ohe.categories_) != len(original_categorical_cols):
    st.error("Error: Mismatch in categories count between encoder and expected columns.")
    st.stop()

brand = st.selectbox("Car Brand", options=ohe.categories_[0])
seller_type = st.selectbox("Seller Type", options=ohe.categories_[1])
fuel_type = st.selectbox("Fuel Type", options=ohe.categories_[2])
transmission_type = st.selectbox("Transmission Type", options=ohe.categories_[3])

st.info("Note: The 'Model' feature was removed during preprocessing due to high cardinality and is not used for prediction in this app.")

st.markdown("---")

# --- 4. Prediction Button and Logic ---
if st.button("Predict Selling Price"): # This button triggers the prediction
    try:
        # Create a dictionary for the new input data
        input_data = {
            'vehicle_age': vehicle_age,
            'km_driven': km_driven,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'seats': seats,
            'brand': brand,
            'seller_type': seller_type,
            'fuel_type': fuel_type,
            'transmission_type': transmission_type,
        }
        
        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Separate numerical and categorical parts of the input_df
        input_df_numerical = input_df[['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']]
        input_df_categorical = input_df[original_categorical_cols]

        # Transform categorical features using the loaded OneHotEncoder
        # Need to ensure that the input to transform is a DataFrame with the same column names as fitted
        encoded_features = ohe.transform(input_df_categorical)
        encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(original_categorical_cols))

        # Get the full list of columns that the model expects (from X_train.columns during training)
        # It's crucial that the columns in the final_input_df match X_train in order and presence
        # Retrieve the original training columns to ensure consistent input structure
        # The `model.feature_names_in_` attribute can give us this if it's available, otherwise we need to rely on the context.
        # From the context, the feature names were derived from X_train.columns.

               # Get the expected feature columns
        if hasattr(model, "feature_names_in_"):
            expected_columns = list(model.feature_names_in_)
        else:
            st.error("The model does not contain feature names.")
            st.stop()

        # Initialize an empty DataFrame
        processed_input_df = pd.DataFrame(
            0,
            index=[0],
            columns=expected_columns
        )

        # Fill in numerical values
        for col in input_df_numerical.columns:
            if col in processed_input_df.columns:
                processed_input_df[col] = input_df_numerical[col].values[0]

        # Fill in one-hot encoded categorical values
        for col in encoded_df.columns:
            if col in processed_input_df.columns:
                processed_input_df[col] = encoded_df[col].values[0]

        # Make prediction
        prediction = model.predict(processed_input_df)
        predicted_price = prediction[0]

        st.subheader("Predicted Selling Price:")
        st.success(f"₹ {predicted_price:,.2f}") # Display formatted price

    except ValueError as ve:
        st.error(f"Input Error: {ve}. Please check your numerical entries.")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
        st.exception(e) # Display full traceback for debugging

st.markdown("---")
st.caption("Developed with Streamlit and Scikit-learn.")
