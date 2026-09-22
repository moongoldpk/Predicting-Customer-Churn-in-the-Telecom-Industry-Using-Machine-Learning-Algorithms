import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

@st.cache_resource
def load_and_train():
    URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(URL)
    df = df.copy().drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df[c] = df[c].map({"Yes": 1, "No": 0})

    cat_cols = ["gender", "MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                "StreamingMovies", "Contract", "PaymentMethod"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    df[df.select_dtypes(include="bool").columns] = df.select_dtypes(include="bool").astype(int)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train_sc, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, class_weight="balanced"),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric="logloss"),
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        trained_models[name] = model

    results = {n: {"model": m, "roc_auc": roc_auc_score(y_test, m.predict_proba(X_test_sc)[:, 1])}
               for n, m in trained_models.items()}
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    return results, best_name, scaler, X.columns

results, best_name, scaler, FEATS = load_and_train()

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Predictor")
st.caption(f"Best model: **{best_name}** (ROC-AUC: {results[best_name]['roc_auc']:.3f})")
st.markdown("---")

NO_INT, NO_PH = "No internet service", "No phone service"

def encode(gender, senior, partner, dependents, tenure, phone, mlines, inet,
           osec, obak, dprot, tsupp, stv, smov, contract, paper, pay, mc, tc):
    if phone == "No":
        mlines = NO_PH
    if inet == "No":
        osec = obak = dprot = tsupp = stv = smov = NO_INT

    row = {"gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
           "Partner": partner, "Dependents": dependents, "tenure": int(tenure),
           "PhoneService": phone, "MultipleLines": mlines, "InternetService": inet,
           "OnlineSecurity": osec, "OnlineBackup": obak, "DeviceProtection": dprot,
           "TechSupport": tsupp, "StreamingTV": stv, "StreamingMovies": smov,
           "Contract": contract, "PaperlessBilling": paper, "PaymentMethod": pay,
           "MonthlyCharges": float(mc), "TotalCharges": float(tc)}
    d = pd.DataFrame([row])
    for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        d[c] = d[c].map({"Yes": 1, "No": 0})
    d = pd.get_dummies(d, columns=["gender", "MultipleLines", "InternetService",
                                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                                    "TechSupport", "StreamingTV", "StreamingMovies",
                                    "Contract", "PaymentMethod"], drop_first=False)
    d = d.reindex(columns=FEATS, fill_value=0).astype(float)
    return scaler.transform(d)

with st.form("customer_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Customer")
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        senior = st.radio("Senior Citizen", ["No", "Yes"], horizontal=True)
        partner = st.radio("Partner", ["No", "Yes"], horizontal=True)
        dependents = st.radio("Dependents", ["No", "Yes"], horizontal=True)
        tenure = st.slider("Tenure (months)", 0, 72, 3)
    with col2:
        st.subheader("📄 Account")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paper = st.radio("Paperless Billing", ["No", "Yes"], horizontal=True)
        pay = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                               "Bank transfer (automatic)", "Credit card (automatic)"])
        mc = st.number_input("Monthly Charges ($)", 0.0, 200.0, 85.0)
        tc = st.number_input("Total Charges ($)", 0.0, 10000.0, 250.0)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("📞 Phone")
        phone = st.radio("Phone Service", ["Yes", "No"], horizontal=True)
        mlines = st.selectbox("Multiple Lines", ["No", "Yes"])
    with col4:
        st.subheader("🌐 Internet")
        inet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        osec = st.selectbox("Online Security", ["No", "Yes"])
        obak = st.selectbox("Online Backup", ["No", "Yes"])
        dprot = st.selectbox("Device Protection", ["No", "Yes"])
        tsupp = st.selectbox("Tech Support", ["No", "Yes"])
        stv = st.selectbox("Streaming TV", ["No", "Yes"])
        smov = st.selectbox("Streaming Movies", ["No", "Yes"])

    submitted = st.form_submit_button("🔮 Predict Churn", type="primary")

if submitted:
    x = encode(gender, senior, partner, dependents, tenure, phone, mlines, inet,
               osec, obak, dprot, tsupp, stv, smov, contract, paper, pay, mc, tc)
    st.subheader("📊 Prediction Results")
    table_data = []
    votes = 0
    for n, r in results.items():
        p = float(r["model"].predict_proba(x)[0, 1])
        label = "Churn" if p >= 0.5 else "Stay"
        votes += label == "Churn"
        table_data.append({"Model": n, "ROC-AUC": f"{r['roc_auc']:.4f}",
                            "Churn Probability": f"{p:.1%}", "Prediction": label})
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    k = len(results)
    risk = "🔴 High Risk" if votes == k else "🟡 Medium Risk" if votes >= k / 2 else "🟢 Low Risk"
    st.success(f"**Consensus: {votes}/{k} models predict churn → {risk}**")
