
# Telco Customer Churn Prediction

## 📊 Project Overview
This project builds and compares four machine learning algorithms to predict customer churn in the telecommunications industry. The best model (Logistic Regression) achieved an **ROC-AUC of 0.8420**.

## 🧠 Algorithms Tested
- Logistic Regression (Best Model)
- Decision Tree
- Random Forest
- XGBoost

## 📈 Key Findings
- **Month-to-month contracts** are the strongest predictor of churn
- **Low tenure** (< 12 months) indicates high churn risk
- **Electronic check** payment method is a red flag

## 🚀 How to Run
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open the notebook: `Customer Churn Prediction Project.ipynb`

## 📂 Files
- `Customer Churn Prediction Project.ipynb` - Full analysis and modelling pipeline
- `telco_churn_best_model.pkl` - Saved model
- `model_comparison_results.csv` - Model evaluation comparison
- `requirements.txt` - Python dependencies

## 🔧 Technologies Used
- Python, Pandas, NumPy
- Scikit-Learn, XGBoost
- Matplotlib, Seaborn
- Gradio (for UI)
