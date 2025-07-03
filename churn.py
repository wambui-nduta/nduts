import streamlit as st
import pandas as pd
import joblib
import numpy as np

#load data set
import pandas as pd
df = pd.read_csv("C:/Users/USER/Downloads/Expresso_churn_dataset (2).csv")


# Set page configuration
st.set_page_config(page_title="Expresso Churn Prediction", page_icon="📊", layout="wide")

# Title and description
st.title("Expresso Churn Prediction App")
st.markdown("""
This app predicts whether a customer will churn based on their usage data. 
Enter the customer details below and click 'Predict' to see the result.
""")

# Load the trained model
try:
    model = joblib.load("C:/Users/USER/Downloads/churn_model.pkl")

except FileNotFoundError:
    st.error("Model file 'churn_model.pkl' not found. Please ensure it is in the same directory as this app.")
    st.stop()

# Load the cleaned dataset to get feature ranges and categories
try:
    df = pd.read_csv('cleaned_churn_data.csv')
except FileNotFoundError:
    st.error("Dataset file 'cleaned_churn_data.csv' not found. Please ensure it is in the same directory as this app.")
    st.stop()

# Define categorical and numerical features
categorical_features = ['REGION', 'TENURE', 'MRG', 'TOP_PACK']
numerical_features = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 
                     'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 
                     'ZONE1', 'ZONE2', 'REGULARITY', 'FREQ_TOP_PACK']

# Create input fields
st.header("Customer Data Input")

# Initialize a dictionary to store user inputs
input_data = {}

# Create two columns for better layout
col1, col2 = st.columns(2)

# Categorical feature inputs
with col1:
    st.subheader("Categorical Features")
    for feature in categorical_features:
        # Get unique values from the dataset, excluding NaN
        options = df[feature].dropna().unique().tolist()
        # Add an empty option for user to select
        options.insert(0, "Select")
        input_data[feature] = st.selectbox(f"{feature}", options, index=0)







## Numerical feature inputs
with col2:
    st.subheader("Numerical Features")
    for feature in numerical_features:
        if feature in df.columns:
            # Get min and max values from the dataset for sliders
            min_val = float(df[feature].min()) if not df[feature].isna().all() else 0.0
            max_val = float(df[feature].max()) if not df[feature].isna().all() else 1000.0
            input_data[feature] = st.slider(f"{feature}", 
                                            min_value=min_val, 
                                            max_value=max_val, 
                                            value=min_val, 
                                            step=0.1 if feature != 'REGULARITY' else 1.0)
        else:
            st.warning(f"'{feature}' not found in data. Skipping...")
            input_data[feature] = 0.0








# Prediction button
if st.button("Predict Churn"):
    # Validate inputs
    missing_fields = [feature for feature, value in input_data.items() if value == "Select" or pd.isna(value)]
    if missing_fields:
        st.warning(f"Please fill in all fields. Missing: {', '.join(missing_fields)}")
    else:
        # Prepare input data for prediction
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables (using the same encoding as in training)
        for feature in categorical_features:
            # Get the unique categories from the training data
            categories = df[feature].dropna().unique()
            # Map the input to a one-hot encoded format
            if input_df[feature].iloc[0] in categories:
                for category in categories:
                    input_df[f"{feature}_{category}"] = (input_df[feature] == category).astype(int)
                input_df = input_df.drop(feature, axis=1)
            else:
                st.error(f"Invalid value for {feature}. Please select a valid option.")
                st.stop()

        # Ensure all expected columns are present
        expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else numerical_features + [
            f"{cat}_{val}" for cat in categorical_features for val in df[cat].dropna().unique()]
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match training data
        input_df = input_df[expected_columns]

        # Make prediction
        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]  # Probability of churn (class 1)
            
            # Display results
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"The customer is likely to churn with a probability of {probability:.2%}.")
            else:
                st.success(f"The customer is unlikely to churn with a probability of {1 - probability:.2%}.")
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

# Add some styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)