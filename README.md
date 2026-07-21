# telco-customer-churn-prediction
# Telco Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Predicting customer churn in the telecommunications industry using machine learning.

## 🎯 Project Overview

Customer churn is a critical challenge in the telecom industry. This project builds and compares **four machine learning models** to predict which customers are likely to churn, enabling proactive retention strategies.

| Metric | Score |
| :--- | :--- |
| **Best Model** | Logistic Regression |
| **ROC-AUC** | **0.8420** |
| **Accuracy** | ~80% |

## 🧠 Models Evaluated

| Algorithm | ROC-AUC |
| :--- | :--- |
| Logistic Regression | 0.8420 🏆 |
| Random Forest | 0.8381 |
| XGBoost | 0.8348 |
| Decision Tree | 0.6975 |

## 🔥 Top Churn Drivers

1. **Contract: Month-to-month** — highest risk
2. **Low Tenure** (< 12 months) — new customers are volatile
3. **High Monthly Charges** — price sensitivity
4. **Electronic Check** — manual payers churn more
5. **Paperless Billing** — correlated with new customers

## 📂 Repository Structure
