from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "scale",
                StandardScaler(),
                ["Time", "Amount", "LogAmount"]
            )
        ],
        remainder="passthrough"
    )