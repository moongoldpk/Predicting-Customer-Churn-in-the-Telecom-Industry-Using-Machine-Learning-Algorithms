import joblib

m = joblib.load('telco_churn_best_model.pkl')
print(type(m).__name__)
print(list(m.named_steps.keys()))
