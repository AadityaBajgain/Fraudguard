from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def create_preprocessor():
    return StandardScaler()


def create_smote_pipeline(*, sampling_strategy="auto", k_neighbors=5, random_state=42):
    
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
