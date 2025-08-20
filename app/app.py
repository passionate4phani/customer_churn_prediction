import streamlit as st
import pandas as pd
import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
from src.predict import load_artifacts, predict_single, predict_batch

# Paths to save artifacts
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

# Load model and preprocessor
model, preprocessor = load_artifacts(MODEL_PATH, PREPROCESSOR_PATH)

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Prediction App")

st.markdown("This app predicts **whether a customer will churn** based based on input data. "
            "You can either upload a CSV file for batch predictions or manually enter details below.")

# Sidebar selection
option = st.sidebar.radio("Choose Prediction Mode", (" Upload CSV (Batch) ", " Manual Input (Single) "))

# --------------------------------- Batch Predictions ---------------------------------
if option == " Upload CSV (Batch) ":
    st.subheader("Upload a csv File for Predictions")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(" Upload Data Preview:", df.head())

        predictions, probabilities = predict_batch(df, model, preprocessor)

        df["Churn_Predicton"] = predictions
        df["Churn_Probability"] = probabilities

        st.subheader("Prediction Results")
        st.write(df.head())

        # Download option
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=" Download Predictions as CSV",
            data = csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )
# --------------------------------- Single Predictions ---------------------------------
elif option == " Manual Input (Single) ":
    st.subheader("Enter Customer Details")

    st.subheader("Manual Input for Customer Details")

    # Create 4 columns
    col1, col2, col3, col4 = st.columns(4)

    # ---- Demographics ----
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=1)

    # ---- Contract ----
    with col2:
        
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=0.0, step=0.1)

    # ---- Billing ----
    with col3:
        
        total_charges = st.number_input("Total Charges", min_value=0.0, value=0.0, step=0.1)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    # ---- Services ----
    with col4:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    if st.button(" Predict Churn "):
        input_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        pred, prob = predict_single(input_data, model, preprocessor)
        churn_label = "Yes ( Likely to Churn) " if pred == 1 else "No (Not Likely)"

        st.metric("Prediction", churn_label)
        st.metric("Churn Probability", f"{prob:.2%}")
