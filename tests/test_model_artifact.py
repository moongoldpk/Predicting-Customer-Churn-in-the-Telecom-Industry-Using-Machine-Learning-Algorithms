from pathlib import Path
import joblib


def test_model_artifact_loads():
    model_path = Path(__file__).resolve().parents[1] / "telco_churn_best_model.pkl"
    assert model_path.exists(), "Model artifact is missing"

    model = joblib.load(model_path)

    assert hasattr(model, "named_steps"), "Loaded object is not a sklearn pipeline"
    assert "preprocessor" in model.named_steps, "Pipeline is missing preprocessor step"
    assert "classifier" in model.named_steps, "Pipeline is missing classifier step"


if __name__ == "__main__":
    test_model_artifact_loads()
    print("Model artifact loads successfully")
