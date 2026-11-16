"""Regression checks for the saved model's input contract and HTTP interface."""

import math
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import app as application
from src import prediction


def transaction_values():
    values = {feature: 0.25 for feature in prediction.RAW_FEATURES}
    values.update(Time=90061.5, Amount=19.73)
    return values


class TransactionInputTests(unittest.TestCase):
    def test_feature_order_and_derived_values_ignore_supplied_metadata(self):
        payload = dict(reversed(list(transaction_values().items())))
        payload.update(Class=1, Hours=99, LogAmount=-100)

        prepared = prediction.prepare_transaction(payload)

        self.assertEqual(
            list(prepared.columns),
            ["Time", *[f"V{number}" for number in range(1, 29)], "Amount", "Hours", "LogAmount"],
        )
        self.assertEqual(prepared.shape, (1, 32))
        self.assertEqual(prepared.iloc[0]["Hours"], 1)
        self.assertAlmostEqual(prepared.iloc[0]["LogAmount"], math.log1p(19.73))
        self.assertEqual(payload["Hours"], 99)

    def test_zero_time_and_amount_are_valid(self):
        payload = transaction_values()
        payload.update(Time=0, Amount=0)
        prepared = prediction.prepare_transaction(payload)
        self.assertEqual(prepared.iloc[0]["Hours"], 0)
        self.assertEqual(prepared.iloc[0]["LogAmount"], 0)

    def test_numeric_strings_and_negative_pca_values_are_supported(self):
        payload = transaction_values()
        payload["V1"] = "-1.23456789012345"
        payload["Amount"] = "1e2"
        prepared = prediction.prepare_transaction(payload)
        self.assertEqual(prepared.iloc[0]["V1"], -1.23456789012345)
        self.assertEqual(prepared.iloc[0]["Amount"], 100)

    def test_all_raw_features_are_required(self):
        for feature in prediction.RAW_FEATURES:
            with self.subTest(feature=feature):
                payload = transaction_values()
                del payload[feature]
                with self.assertRaises(prediction.InputError) as raised:
                    prediction.prepare_transaction(payload)
                self.assertEqual(raised.exception.fields, [feature])

    def test_invalid_numeric_values_identify_the_field(self):
        invalid_values = (
            True, False, None, "", " ", "invalid", [], {},
            float("nan"), float("inf"), float("-inf"), "NaN", "Infinity",
            prediction.MAX_FEATURE_VALUE * 2, -prediction.MAX_FEATURE_VALUE * 2,
            10 ** 400,
        )
        for value in invalid_values:
            with self.subTest(value=value):
                payload = transaction_values()
                payload["V7"] = value
                with self.assertRaises(prediction.InputError) as raised:
                    prediction.prepare_transaction(payload)
                self.assertEqual(raised.exception.fields, ["V7"])

    def test_negative_amount_and_elapsed_time_are_rejected(self):
        for feature in ("Time", "Amount"):
            with self.subTest(feature=feature):
                payload = transaction_values()
                payload[feature] = -0.01
                with self.assertRaises(prediction.InputError) as raised:
                    prediction.prepare_transaction(payload)
                self.assertEqual(raised.exception.fields, [feature])

    def test_non_object_payloads_are_rejected(self):
        for payload in (None, [], 1, "transaction"):
            with self.subTest(payload=payload):
                with self.assertRaises(prediction.InputError):
                    prediction.prepare_transaction(payload)

    def test_threshold_is_inclusive_and_class_order_is_respected(self):
        model = Mock()
        model.classes_ = np.array([1, 0])
        for probability, expected in ((0.499999, 0), (0.5, 1), (0.500001, 1)):
            with self.subTest(probability=probability):
                model.predict_proba.return_value = np.array([[probability, 1 - probability]])
                with patch.object(prediction, "load_model", return_value=model):
                    result = prediction.predict_transaction(transaction_values())
                self.assertEqual(result["prediction"], expected)
                self.assertEqual(result["fraud_probability"], probability)
                self.assertEqual(result["label"], "Fraudulent" if expected else "Legitimate")


class PredictionApiTests(unittest.TestCase):
    def setUp(self):
        self.client = application.create_app().test_client()

    def test_invalid_json_returns_json_error(self):
        response = self.client.post("/api/predict", data="{", content_type="application/json")
        self.assertEqual(response.status_code, 400)
        self.assertIn("invalid JSON", response.get_json()["error"])
        self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_non_json_content_returns_415(self):
        response = self.client.post("/api/predict", data=transaction_values())
        self.assertEqual(response.status_code, 415)
        self.assertIn("error", response.get_json())

    def test_oversized_json_returns_413(self):
        response = self.client.post("/api/predict", json={"padding": "x" * (64 * 1024)})
        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.get_json())

    def test_invalid_input_returns_field_errors_before_loading_model(self):
        with patch.object(prediction, "load_model") as mocked_load:
            response = self.client.post("/api/predict", json={"Time": 0, "Amount": 0})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["fields"], [f"V{number}" for number in range(1, 29)])
        mocked_load.assert_not_called()

    def test_unavailable_model_has_actionable_503_responses(self):
        error = prediction.ModelUnavailableError("Restore the saved model.")
        with patch.object(application, "load_model", side_effect=error):
            response = self.client.get("/api/model")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.get_json()["ready"])
        self.assertEqual(response.get_json()["error"], str(error))

        with patch.object(prediction, "load_model", side_effect=error):
            response = self.client.post("/api/predict", json=transaction_values())
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error"], str(error))

    def test_unavailable_or_empty_examples_return_503(self):
        for failure in (FileNotFoundError(), ValueError(), KeyError("Class")):
            with self.subTest(failure=type(failure).__name__):
                with patch.object(application, "load_examples", side_effect=failure):
                    response = self.client.get("/api/examples")
                self.assertEqual(response.status_code, 503)
                self.assertIn("error", response.get_json())
        with patch.object(application, "load_examples", return_value=[]):
            response = self.client.get("/api/examples")
        self.assertEqual(response.status_code, 503)


@unittest.skipUnless(
    prediction.MODEL_PATH.is_file() and (prediction.ROOT / "data/processed/test.csv").is_file(),
    "The saved Random Forest and test dataset are required for integration checks.",
)
class SavedModelIntegrationTests(unittest.TestCase):
    def test_examples_match_direct_saved_model_predictions(self):
        model = prediction.load_model()
        client = application.create_app().test_client()
        examples_response = client.get("/api/examples")
        self.assertEqual(examples_response.status_code, 200)
        examples = examples_response.get_json()["examples"]
        self.assertEqual(len(examples), 2)
        test_data = pd.read_csv(prediction.ROOT / "data/processed/test.csv")
        fraud_index = list(model.classes_).index(1)

        for label, example in enumerate(examples):
            with self.subTest(example=example["name"]):
                self.assertEqual(set(example["values"]), set(prediction.RAW_FEATURES))
                source_row = test_data.loc[test_data["Class"] == label].iloc[[0]]
                expected = float(model.predict_proba(source_row.loc[:, model.feature_names_in_])[0, fraud_index])
                response = client.post("/api/predict", json=example["values"])
                self.assertEqual(response.status_code, 200)
                result = response.get_json()
                self.assertAlmostEqual(result["fraud_probability"], expected, places=12)
                self.assertEqual(result["prediction"], int(expected >= 0.5))
                self.assertEqual(result["prediction"], label)
                self.assertEqual(response.headers["Cache-Control"], "no-store")

        info = client.get("/api/model")
        self.assertEqual(info.status_code, 200)
        self.assertTrue(info.get_json()["ready"])
        self.assertEqual(info.get_json()["trees"], model.n_estimators)


if __name__ == "__main__":
    unittest.main()
