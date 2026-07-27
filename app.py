
import streamlit as st
import pandas as pd
import joblib

model = joblib.load('telco_churn_best_model.pkl')
st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📊 Telco Customer Churn Predictor")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 120.0, 70.0)
total_charges = tenure * monthly_charges
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

tenure_group = pd.cut([tenure], bins=[0,12,24,48,72,100], labels=['0-12','12-24','24-48','48-72','72+'])[0]
input_data = pd.DataFrame({
    'SeniorCitizen': [senior_citizen], 'Partner': ['Yes'], 'Dependents': ['No'],
    'tenure': [tenure], 'PhoneService': ['Yes'], 'PaperlessBilling': ['Yes'],
    'MonthlyCharges': [monthly_charges], 'TotalCharges': [total_charges],
    'Service_Count': [3], 'Avg_Monthly_Spend': [total_charges/tenure if tenure>0 else 0],
    'gender': ['Male'], 'MultipleLines': ['No'],
    'InternetService': [internet_service], 'OnlineSecurity': ['No'],
    'OnlineBackup': ['No'], 'DeviceProtection': ['No'],
    'TechSupport': ['No'], 'StreamingTV': ['No'],
    'StreamingMovies': ['No'], 'Contract': [contract],
    'PaymentMethod': [payment_method], 'Tenure_Group': [tenure_group]
})

if st.button("Predict Churn Risk"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    if pred == 1:
        st.error(f"⚠️ High Churn Risk: {prob:.1%}")
    else:
        st.success(f"✅ Low Churn Risk: {prob:.1%}")
