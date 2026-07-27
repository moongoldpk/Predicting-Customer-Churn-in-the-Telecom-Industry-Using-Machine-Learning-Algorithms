from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


def test_feature_frame_contains_expected_columns_and_engineered_values():
    inputs = {
        "gender": "Female",
        "senior_citizen": False,
        "partner": True,
        "dependents": True,
        "tenure": 6,
        "phone_service": True,
        "multiple_lines": False,
        "internet_service": "DSL",
        "online_security": True,
        "online_backup": True,
        "device_protection": False,
        "tech_support": False,
        "streaming_tv": False,
        "streaming_movies": False,
        "contract": "Month-to-month",
        "paperless_billing": True,
        "monthly_charges": 45.0,
        "total_charges": 250.0,
        "payment_method": "Electronic check",
    }

    frame = app.build_feature_frame(inputs)

    assert set(frame.columns) == set(app.FEATURE_COLUMNS)
    assert frame.loc[0, "Service_Count"] == 2
    assert frame.loc[0, "Tenure_Group"] == "New"


def test_predict_churn_returns_probability_and_label():
    inputs = {
        "gender": "Male",
        "senior_citizen": False,
        "partner": False,
        "dependents": False,
        "tenure": 11,
        "phone_service": True,
        "multiple_lines": True,
        "internet_service": "Fiber optic",
        "online_security": False,
        "online_backup": False,
        "device_protection": False,
        "tech_support": False,
        "streaming_tv": True,
        "streaming_movies": True,
        "contract": "Month-to-month",
        "paperless_billing": True,
        "monthly_charges": 85.5,
        "total_charges": 900.0,
        "payment_method": "Electronic check",
    }

    frame = app.build_feature_frame(inputs)
    probability, label = app.predict_churn(frame)

    assert 0.0 <= probability <= 1.0
    assert label in {"Low risk", "Moderate risk", "High risk"}
