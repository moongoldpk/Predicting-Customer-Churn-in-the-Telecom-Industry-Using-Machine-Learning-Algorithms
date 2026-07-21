# telco-customer-churn-prediction
# 📊 Telco Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)](https://xgboost.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Predicting customer churn in the telecommunications industry using machine learning.

---

## 🎯 Project Overview

Customer churn is a critical challenge in the telecom industry. This project builds and compares **four machine learning models** to predict which customers are likely to churn, enabling proactive retention strategies.

| Metric | Score |
| :--- | :--- |
| **Best Model** | Logistic Regression |
| **ROC-AUC** | **0.8420** |
| **Accuracy** | ~80% |
| **Recall (Churn)** | ~72% |

---

## 🧠 Models Evaluated

| Algorithm | ROC-AUC | Status |
| :--- | :--- | :--- |
| Logistic Regression | 0.8420 | 🏆 **Best** |
| Random Forest | 0.8381 | ✅ Strong |
| XGBoost | 0.8348 | ✅ Strong |
| Decision Tree | 0.6975 | ⚠️ Weak |

---

## 🔥 Top 5 Churn Drivers

1. **Contract: Month-to-month** – Customers on flexible contracts churn 4x more
2. **Tenure (Low)** – New customers (< 12 months) are high-risk
3. **Monthly Charges** – Higher bills increase churn likelihood
4. **Payment Method: Electronic Check** – Manual payers churn more
5. **Paperless Billing** – Correlated with higher churn (often new customers)

---

## 📂 Repository Structure
