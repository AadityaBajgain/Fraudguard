"""Run the local FraudGuard prediction interface with `python app.py`."""

import argparse

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from src.prediction import (
    InputError,
    MODEL_FEATURES,
    ModelUnavailableError,
    THRESHOLD,
    load_examples,
    load_model,
    predict_transaction,
)


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/model")
    def model_info():
        info = {
            "name": "Random Forest",
            "threshold": THRESHOLD,
            "features": len(MODEL_FEATURES),
        }
        try:
            model = load_model()
        except ModelUnavailableError as error:
            return jsonify(**info, ready=False, error=str(error)), 503
        return jsonify(
            **info,
            ready=True,
            trees=model.n_estimators,
            max_depth=model.max_depth,
        )

    @app.get("/api/examples")
    def examples():
        try:
            samples = load_examples()
        except (OSError, ValueError, KeyError):
            return jsonify(
                error="Sample transactions are unavailable. Enter your own values "
                "or restore data/processed/test.csv."
            ), 503
        if not samples:
            return jsonify(error="No sample transactions were found in the test dataset."), 503
        return jsonify(examples=samples)

    @app.post("/api/predict")
    def predict():
        if not request.is_json:
            return jsonify(error="Send the transaction with Content-Type application/json."), 415
        try:
            return jsonify(predict_transaction(request.get_json()))
        except BadRequest:
            return jsonify(error="The transaction contains invalid JSON."), 400
        except InputError as error:
            return jsonify(error=str(error), fields=error.fields), 400
        except ModelUnavailableError as error:
            return jsonify(error=str(error)), 503

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify(error="The transaction exceeds the 64 KB request limit."), 413

    @app.after_request
    def prevent_cached_predictions(response):
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    return app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FraudGuard transaction interface.")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app.run(host="127.0.0.1", port=args.port, debug=False)
