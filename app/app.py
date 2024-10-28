import streamlit as st
import pandas as pd
import joblib

# Load the saved model artifacts
model_artifacts = joblib.load('../model/model_with_scaler_and_encoder.pkl')
scaler = model_artifacts['scaler']
label_encoder = model_artifacts['encoder']
model = model_artifacts['model']
feature_names = model_artifacts['feature_names']  # Ensure this is stored as a list

# Ensure feature_names is a list
if isinstance(feature_names, str):
    feature_names = [feature_names]  # Convert to list if necessary

# App title
st.title("Startup Success Predictor")
st.write("Predict the likelihood of startup success based on funding and company characteristics.")

# User input function
def user_input_features():
    # Collecting user input for all features used during training
    age_first_funding_year = st.number_input("Age at first funding (years)", min_value=0.0, max_value=50.0, step=0.1)
    age_last_funding_year = st.number_input("Age at last funding (years)", min_value=0.0, max_value=50.0, step=0.1)
    funding_rounds_df1 = st.number_input("Number of funding rounds", min_value=1, max_value=20, step=1)
    funding_total_usd_df1 = st.number_input("Total funding (USD)", min_value=0, max_value=1000000000, step=10000)
    valuation = st.number_input("Valuation ($B)", min_value=0.0, max_value=100.0, step=0.1)
    
    # Collect binary input as "Yes" or "No"
    has_vc = st.selectbox("Has VC funding?", ["No", "Yes"])
    has_angel = st.selectbox("Has Angel funding?", ["No", "Yes"])
    has_rounda = st.selectbox("Has Series A?", ["No", "Yes"])
    has_roundb = st.selectbox("Has Series B?", ["No", "Yes"])
    has_roundc = st.selectbox("Has Series C?", ["No", "Yes"])
    has_roundd = st.selectbox("Has Series D?", ["No", "Yes"])
    is_software = st.selectbox("Is Software category?", ["No", "Yes"])
    is_web = st.selectbox("Is Web category?", ["No", "Yes"])
    is_mobile = st.selectbox("Is Mobile category?", ["No", "Yes"])
    is_enterprise = st.selectbox("Is Enterprise category?", ["No", "Yes"])
    is_advertising = st.selectbox("Is Advertising category?", ["No", "Yes"])
    is_gamesvideo = st.selectbox("Is Games/Video category?", ["No", "Yes"])
    is_ecommerce = st.selectbox("Is E-commerce category?", ["No", "Yes"])
    is_biotech = st.selectbox("Is Biotech category?", ["No", "Yes"])
    is_consulting = st.selectbox("Is Consulting category?", ["No", "Yes"])
    is_othercategory = st.selectbox("Is Other category?", ["No", "Yes"])
    
    # Convert "Yes" to 1 and "No" to 0 for model compatibility
    user_data = {
        'age_first_funding_year': age_first_funding_year,
        'age_last_funding_year': age_last_funding_year,
        'funding_rounds_df1': funding_rounds_df1,
        'funding_total_usd_df1': funding_total_usd_df1,
        'valuation ($b)': valuation,
        'has_vc': 1 if has_vc == "Yes" else 0,
        'has_angel': 1 if has_angel == "Yes" else 0,
        'has_rounda': 1 if has_rounda == "Yes" else 0,
        'has_roundb': 1 if has_roundb == "Yes" else 0,
        'has_roundc': 1 if has_roundc == "Yes" else 0,
        'has_roundd': 1 if has_roundd == "Yes" else 0,
        'is_software': 1 if is_software == "Yes" else 0,
        'is_web': 1 if is_web == "Yes" else 0,
        'is_mobile': 1 if is_mobile == "Yes" else 0,
        'is_enterprise': 1 if is_enterprise == "Yes" else 0,
        'is_advertising': 1 if is_advertising == "Yes" else 0,
        'is_gamesvideo': 1 if is_gamesvideo == "Yes" else 0,
        'is_ecommerce': 1 if is_ecommerce == "Yes" else 0,
        'is_biotech': 1 if is_biotech == "Yes" else 0,
        'is_consulting': 1 if is_consulting == "Yes" else 0,
        'is_othercategory': 1 if is_othercategory == "Yes" else 0
    }
    
    # Convert the dictionary to a DataFrame with the same feature names as training
    input_df = pd.DataFrame([user_data], columns=feature_names)
    return input_df

# Collecting user input
input_df = user_input_features()

# Scale the input data
input_scaled = scaler.transform(input_df)

# Predict with the model
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Display the prediction
st.subheader("Prediction")
if prediction[0] == 1:
    st.write("The model predicts **Success** for this startup!")
else:
    st.write("The model predicts **Failure** for this startup.")

# Display the prediction probabilities
st.subheader("Prediction Probability")
st.write(f"Success probability: {prediction_proba[0][1]:.2f}")
st.write(f"Failure probability: {prediction_proba[0][0]:.2f}")