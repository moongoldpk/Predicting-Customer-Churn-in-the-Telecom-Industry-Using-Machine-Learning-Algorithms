import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = Path(__file__).resolve().parents[1] / 'WA_Fn-UseC_-Telco-Customer-Churn (1).csv'
MODEL_PATH = Path(__file__).resolve().parents[1] / 'telco_churn_best_model.pkl'


def prepare_data():
    df = pd.read_csv(DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)

    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['Service_Count'] = df[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
    df['Avg_Monthly_Spend'] = df['TotalCharges'] / df['tenure'].replace(0, np.nan)
    df['Avg_Monthly_Spend'].fillna(0, inplace=True)

    def tenure_bucket(months):
        if months < 12:
            return 'New'
        elif months < 48:
            return 'Mid'
        else:
            return 'Loyal'

    df['Tenure_Group'] = df['tenure'].apply(tenure_bucket)

    target_col = 'Churn'
    y = df[target_col].map({'Yes': 1, 'No': 0})
    y = y.dropna()
    X = df.drop(columns=[target_col, 'customerID']).loc[y.index]

    for col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Service_Count', 'Avg_Monthly_Spend']:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Tenure_Group']:
        X[col] = X[col].astype(str)

    numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Service_Count', 'Avg_Monthly_Spend']
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Tenure_Group']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f'Accuracy: {acc:.4f}')
    joblib.dump(pipeline, MODEL_PATH)
    print(f'Saved model to {MODEL_PATH}')


if __name__ == '__main__':
    prepare_data()
