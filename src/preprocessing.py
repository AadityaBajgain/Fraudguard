from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def create_preprocessor():
    """Scale all numeric predictors; callers must exclude the Class target."""
    return StandardScaler()


def create_smote_pipeline(*, sampling_strategy="auto", k_neighbors=5, random_state=42):
    """Build an unfitted pipeline for the prepared numeric features.

    Fit on X_train and y_train only, with Class excluded from X_train.
    Scaling learns from the original training rows before SMOTE resamples.
    Prediction reuses that scaler and skips SMOTE. The default sampling
    strategy balances the classes; k_neighbors requires at least k + 1
    minority examples in each training fold.
    """
    return Pipeline(
        steps=[
            ("preprocessing", create_preprocessor()),
            (
                "smote",
                SMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k_neighbors,
                    random_state=random_state,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
