import joblib
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models/random_forest_classifier.joblib"
model = joblib.load(MODEL_PATH)
print(type(model))

print(MODEL_PATH)

print(model.get_params())

mlflow.set_experiment("Fraud-Model-Comparison")



with mlflow.start_run(run_name="RandomForest-Final"):
    mlflow.log_param("model","RandomForestClassifier")
    mlflow.log_param("threshold",0.5)
    mlflow.log_params(model.get_params())
    
    mlflow.log_artifact(
        str(MODEL_PATH),
        artifact_path="model"
    )
    mlflow.log_metric("test_average_precision", 0.6744)
    mlflow.log_metric("test_precision", 67 / 74)
    mlflow.log_metric("test_recall", 67 / 90)
    mlflow.log_metric("test_f1", 134 / 164)
    
print("Mlflow successfully updated")

# print("Tracking URI:", mlflow.get_tracking_uri())

# with mlflow.start_run(run_id="12d4ddf09e4843ae9ee004990a6e35c1") as run:
#     mlflow.log_metric("connection_check", 1.0)

#     print("Updated Run ID:", run.info.run_id)