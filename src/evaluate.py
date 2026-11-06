import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "data/processed/test.csv")

X_test = df.drop(columns=["Class"])
y_test = df["Class"]

rf_model = joblib.load(ROOT / "models/random_forest_classifier.joblib")

THRESHOLD = 0.5

y_pred = rf_model.predict_proba(X_test)[:,1]

y_pred = (
    y_pred >= THRESHOLD
).astype(int)

print(
    classification_report(y_test,y_pred)
)


print("Confusion Matrix: ",
      confusion_matrix(y_test, y_pred)
)


print("PR-AUC: ", round(average_precision_score(y_test, y_pred),4))

