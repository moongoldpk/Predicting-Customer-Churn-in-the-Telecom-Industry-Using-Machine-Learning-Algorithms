from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "WA_Fn-UseC_-Telco-Customer-Churn (1).csv"
MODEL_PATH = ROOT / "telco_churn_best_model.pkl"

MODEL = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "MonthlyCharges",
    "TotalCharges",
    "PaymentMethod",
    "Service_Count",
    "Avg_Monthly_Spend",
    "Tenure_Group",
]


def build_feature_frame(inputs: dict) -> pd.DataFrame:
    data = {
        "gender": [inputs["gender"]],
        "SeniorCitizen": [1 if inputs["senior_citizen"] else 0],
        "Partner": ["Yes" if inputs["partner"] else "No"],
        "Dependents": ["Yes" if inputs["dependents"] else "No"],
        "tenure": [inputs["tenure"]],
        "PhoneService": ["Yes" if inputs["phone_service"] else "No"],
        "MultipleLines": ["Yes" if inputs["multiple_lines"] else "No"],
        "InternetService": [inputs["internet_service"]],
        "OnlineSecurity": ["Yes" if inputs["online_security"] else "No"],
        "OnlineBackup": ["Yes" if inputs["online_backup"] else "No"],
        "DeviceProtection": ["Yes" if inputs["device_protection"] else "No"],
        "TechSupport": ["Yes" if inputs["tech_support"] else "No"],
        "StreamingTV": ["Yes" if inputs["streaming_tv"] else "No"],
        "StreamingMovies": ["Yes" if inputs["streaming_movies"] else "No"],
        "Contract": [inputs["contract"]],
        "PaperlessBilling": ["Yes" if inputs["paperless_billing"] else "No"],
        "MonthlyCharges": [inputs["monthly_charges"]],
        "TotalCharges": [inputs["total_charges"]],
        "PaymentMethod": [inputs["payment_method"]],
    }
    frame = pd.DataFrame(data)
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame["MonthlyCharges"] = pd.to_numeric(frame["MonthlyCharges"], errors="coerce")
    frame["Service_Count"] = (
        frame[["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]]
        .apply(lambda x: (x == "Yes").sum(), axis=1)
    )
    frame["Avg_Monthly_Spend"] = frame["TotalCharges"] / frame["tenure"].replace(0, np.nan)
    frame["Avg_Monthly_Spend"] = frame["Avg_Monthly_Spend"].fillna(0)

    def tenure_bucket(months: int) -> str:
        if months < 12:
            return "New"
        if months < 48:
            return "Mid"
        return "Loyal"

    frame["Tenure_Group"] = frame["tenure"].apply(tenure_bucket)

    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Service_Count", "Avg_Monthly_Spend"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)

    for col in [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "Tenure_Group",
    ]:
        frame[col] = frame[col].astype(str)

    return frame[FEATURE_COLUMNS]


def predict_churn(frame: pd.DataFrame) -> tuple[float, str]:
    probability = float(MODEL.predict_proba(frame)[0, 1])
    if probability < 0.3:
        label = "Low risk"
    elif probability < 0.6:
        label = "Moderate risk"
    else:
        label = "High risk"
    return probability, label


def build_visualisations() -> tuple[str, str]:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    churn_rate = df["Churn"].mean() * 100
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="Contract", y="Churn", estimator="mean", ax=ax1)
    ax1.set_title("Churn rate by contract")
    ax1.set_ylabel("Churn rate")
    ax1.set_ylim(0, 1)
    plt.tight_layout()
    fig1_path = ROOT / "contract_chart.png"
    fig1.savefig(fig1_path, dpi=150)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax2)
    ax2.set_title("Monthly charges vs churn")
    ax2.set_xlabel("Churn")
    ax2.set_ylabel("Monthly charges")
    plt.tight_layout()
    fig2_path = ROOT / "charges_chart.png"
    fig2.savefig(fig2_path, dpi=150)
    plt.close(fig2)

    return str(fig1_path), str(fig2_path)


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Telco Churn Predictor") as demo:
        gr.Markdown("# Telco Customer Churn Predictor")
        gr.Markdown("Use the calculator below to estimate churn risk for a customer profile.")

        with gr.Row():
            with gr.Column():
                gender = gr.Dropdown(["Female", "Male"], label="Gender")
                senior_citizen = gr.Checkbox(label="Senior citizen")
                partner = gr.Checkbox(label="Has partner")
                dependents = gr.Checkbox(label="Has dependents")
                tenure = gr.Slider(minimum=1, maximum=72, step=1, value=12, label="Tenure (months)")
                phone_service = gr.Checkbox(label="Phone service")
                multiple_lines = gr.Checkbox(label="Multiple lines")
                internet_service = gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet service")
                online_security = gr.Checkbox(label="Online security")
                online_backup = gr.Checkbox(label="Online backup")
                device_protection = gr.Checkbox(label="Device protection")
                tech_support = gr.Checkbox(label="Tech support")
                streaming_tv = gr.Checkbox(label="Streaming TV")
                streaming_movies = gr.Checkbox(label="Streaming movies")
                contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract")
                paperless_billing = gr.Checkbox(label="Paperless billing")
                monthly_charges = gr.Number(label="Monthly charges", value=70)
                total_charges = gr.Number(label="Total charges", value=500)
                payment_method = gr.Dropdown(["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"], label="Payment method")

                submit = gr.Button("Predict churn risk")

            with gr.Column():
                output_text = gr.Textbox(label="Risk summary")
                output_prob = gr.Textbox(label="Probability of churn")
                output_label = gr.Textbox(label="Risk band")
                image1 = gr.Image(type="filepath", label="Churn rate by contract")
                image2 = gr.Image(type="filepath", label="Monthly charges vs churn")

        submit.click(
            fn=predict_from_inputs,
            inputs=[
                gender,
                senior_citizen,
                partner,
                dependents,
                tenure,
                phone_service,
                multiple_lines,
                internet_service,
                online_security,
                online_backup,
                device_protection,
                tech_support,
                streaming_tv,
                streaming_movies,
                contract,
                paperless_billing,
                monthly_charges,
                total_charges,
                payment_method,
            ],
            outputs=[output_text, output_prob, output_label, image1, image2],
        )
    return demo


def predict_from_inputs(
    gender: str,
    senior_citizen: bool,
    partner: bool,
    dependents: bool,
    tenure: int,
    phone_service: bool,
    multiple_lines: bool,
    internet_service: str,
    online_security: bool,
    online_backup: bool,
    device_protection: bool,
    tech_support: bool,
    streaming_tv: bool,
    streaming_movies: bool,
    contract: str,
    paperless_billing: bool,
    monthly_charges: float,
    total_charges: float,
    payment_method: str,
) -> tuple[str, str, str, str, str]:
    inputs = {
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": int(tenure),
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "monthly_charges": float(monthly_charges),
        "total_charges": float(total_charges),
        "payment_method": payment_method,
    }
    frame = build_feature_frame(inputs)
    probability, label = predict_churn(frame)
    summary = (
        f"This customer has an estimated churn probability of {probability:.1%}."
        f" The profile falls into the {label.lower()} category."
    )
    chart1, chart2 = build_visualisations()
    return summary, f"{probability:.1%}", label, chart1, chart2


if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=False, server_name="0.0.0.0")
