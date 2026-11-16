"""Prepare transactions for the saved FraudGuard random forest."""

from functools import lru_cache
from pathlib import Path
import math
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest_classifier.joblib"
RAW_FEATURES = ("Time", *(f"V{number}" for number in range(1, 29)), "Amount")
MODEL_FEATURES = (*RAW_FEATURES, "Hours", "LogAmount")
THRESHOLD = 0.5
MAX_FEATURE_VALUE = float(np.finfo(np.float32).max)


class InputError(ValueError):
    def __init__(self, message, fields=()):
        super().__init__(message)
        self.fields = list(fields)


class ModelUnavailableError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_model():
    """Load the repository's trusted artifact once per server process."""
    if not MODEL_PATH.is_file():
        raise ModelUnavailableError(
            "Place the trained model at models/random_forest_classifier.joblib "
            "and restart the app."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(MODEL_PATH)
    except InconsistentVersionWarning as error:
        raise ModelUnavailableError(
            "The saved model requires scikit-learn "
            f"{error.original_sklearn_version}. Install requirements.txt "
            "in your virtual environment and restart the app."
        ) from error
    except Exception as error:
        raise ModelUnavailableError(
            "The saved model could not be loaded. Check the model artifact "
            "and install the dependencies in requirements.txt."
        ) from error

    if not isinstance(model, RandomForestClassifier):
        raise ModelUnavailableError("The model artifact must be a Random Forest classifier.")
    if tuple(getattr(model, "feature_names_in_", ())) != MODEL_FEATURES:
        raise ModelUnavailableError("The saved model's input features do not match this app.")
    if set(model.classes_) != {0, 1}:
        raise ModelUnavailableError("The saved model must use classes 0 and 1.")
    return model


def prepare_transaction(payload):
    if not isinstance(payload, dict):
        raise InputError("Send a transaction as a JSON object.")

    values = {}
    invalid_fields = []
    for feature in RAW_FEATURES:
        value = payload.get(feature)
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            invalid_fields.append(feature)
            continue
        try:
            number = float(value)
        except (ValueError, OverflowError):
            invalid_fields.append(feature)
            continue
        if not math.isfinite(number) or abs(number) > MAX_FEATURE_VALUE:
            invalid_fields.append(feature)
            continue
        values[feature] = number

    if invalid_fields:
        raise InputError(
            "Enter a finite numeric value within the model's supported range "
            "for every required field.",
            invalid_fields,
        )

    negative_fields = [feature for feature in ("Time", "Amount") if values[feature] < 0]
    if negative_fields:
        raise InputError("Amount and elapsed time must be zero or greater.", negative_fields)

    values["Hours"] = int((values["Time"] // 3600) % 24)
    values["LogAmount"] = math.log1p(values["Amount"])
    return pd.DataFrame([values], columns=MODEL_FEATURES)


def predict_transaction(payload):
    transaction = prepare_transaction(payload)
    model = load_model()
    fraud_index = list(model.classes_).index(1)
    probability = float(model.predict_proba(transaction)[0, fraud_index])
    prediction = int(probability >= THRESHOLD)
    return {
        "prediction": prediction,
        "label": "Fraudulent" if prediction else "Legitimate",
        "fraud_probability": probability,
        "threshold": THRESHOLD,
        "features": {
            "Hours": int(transaction.iloc[0]["Hours"]),
            "LogAmount": float(transaction.iloc[0]["LogAmount"]),
        },
        "transaction": {
            "Time": float(transaction.iloc[0]["Time"]),
            "Amount": float(transaction.iloc[0]["Amount"]),
        },
    }


@lru_cache(maxsize=1)
def load_examples():
    """Read one held-out example of each class without depending on training data."""
    examples = {}
    with pd.read_csv(ROOT / "data" / "processed" / "test.csv", chunksize=1000) as chunks:
        for chunk in chunks:
            for label, name in ((0, "Legitimate sample"), (1, "Fraud sample")):
                matching_rows = chunk.loc[chunk["Class"] == label]
                if label not in examples and not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    examples[label] = {
                        "name": name,
                        "values": {feature: float(row[feature]) for feature in RAW_FEATURES},
                    }
            if len(examples) == 2:
                break
    return [examples[label] for label in sorted(examples)]
