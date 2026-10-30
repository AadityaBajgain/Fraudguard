import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import sklearn.pipeline as pipeline
from preprocessing import (create_preprocessor, create_smote_pipeline)


ROOT = Path(__file__).resolve().parents[1]

train = pd.read_csv("data/processed/train.csv")
X_train = train.drop(columns=["Class"])
y_train = train["Class"]

logistic_regression_model = pipeline.Pipeline(
    [
        ("preprocessing",create_preprocessor()),
        ("model",LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ]
)


logistic_regression_model.fit(X_train, y_train)

print("*** Logistic Regression Training Complete ***")



smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

logistic_regression_smote = create_smote_pipeline()

logistic_regression_smote.fit(X_train_smote, y_train_smote)

print("*** SMOTE Logistic Regression Training Complete ***")


rf_model = RandomForestClassifier(
    n_estimators=5,
    max_depth=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

print("*** Random Forest Classifier Training Complete ***")

model_dir = ROOT / "models"


model_dir.mkdir(exist_ok= True)

joblib.dump(logistic_regression_model, model_dir / "logistic_regression.joblib")
joblib.dump(logistic_regression_smote, model_dir / "smote.joblib")
joblib.dump(rf_model, model_dir / "random_forest_classifier.joblib")

print(" **** Model Trained and Saved Successfully. **** ")