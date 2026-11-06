# FraudGuard

FraudGuard compares classifiers for detecting fraudulent credit-card transactions. The target is `Class`: `0` means legitimate and `1` means fraud. The selected model is a balanced random forest, with a fraud decision threshold of **0.5** chosen on the validation set.

## Selected model

The configuration below is present in both [the experimentation notebook](notebooks/02_experimentation.ipynb) and [the training script](src/train.py), and was checked against the saved model at `models/random_forest_classifier.joblib`.

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
```

Other relevant saved settings are `criterion="gini"`, `max_features="sqrt"`, `bootstrap=True`, `min_samples_split=2`, `min_samples_leaf=1`, `max_samples=None`, `max_leaf_nodes=None`, and `ccp_alpha=0.0`. Out-of-bag scoring is disabled.

The random forest trains directly on all 32 features from `data/processed/train.csv`, with no scaling or SMOTE. Its saved class order is `[0, 1]`. The notebook also compares balanced logistic regression and logistic regression with SMOTE; the random forest has the highest recorded validation F1.

## Decision threshold

Predict fraud when the model's class-1 probability is **greater than or equal to 0.5**:

```python
y_scores = rf_model.predict_proba(X)[:, 1]
y_pred = (y_scores >= 0.5).astype(int)
```

The notebook tests thresholds from `0.1` to `0.9` in steps of `0.1` on validation data and selects the threshold with the highest fraud-class F1.

| Threshold | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| 0.1 | 0.2385 | 0.9022 | 0.3773 |
| 0.2 | 0.6220 | 0.8587 | 0.7215 |
| 0.3 | 0.8041 | 0.8478 | 0.8254 |
| 0.4 | 0.8370 | 0.8370 | 0.8370 |
| **0.5** | **0.8750** | **0.8370** | **0.8556** |
| 0.6 | 0.8824 | 0.8152 | 0.8475 |
| 0.7 | 0.9054 | 0.7283 | 0.8072 |
| 0.8 | 0.9118 | 0.6739 | 0.7750 |
| 0.9 | 0.9388 | 0.5000 | 0.6525 |

Thus, `0.5` is best among the tested thresholds, rather than a proven optimum over every possible threshold. The model and threshold are held fixed for test evaluation.

## Dataset and splits

[The EDA notebook](notebooks/01_eda.ipynb) loads `data/raw/creditcard.csv`, containing 284,807 rows. Removing 1,081 exact duplicate rows leaves **283,726 transactions**, including **473 frauds** and **283,253 legitimate transactions**. Two derived features, `Hours` and `LogAmount`, are added before splitting. The split script reads the resulting data from `data/added_feature/feature.csv`.

[The split script](src/split_data.py) uses two shuffled `train_test_split` calls, both with `random_state=42`:

1. Reserve 20% of the data for testing.
2. Reserve 25% of the remaining 80% for validation, producing approximately 60% training, 20% validation, and 20% test data overall.

Neither split specifies stratification, chronological ordering, or grouping. Counts below were checked directly against the processed CSV files.

| Split | File | Transactions | Legitimate | Fraud | Fraud rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | `data/processed/train.csv` | 170,235 | 169,944 | 291 | 0.1709% |
| Validation | `data/processed/val.csv` | 56,745 | 56,653 | 92 | 0.1621% |
| Test | `data/processed/test.csv` | 56,746 | 56,656 | 90 | 0.1586% |

Training data fits the model, validation data supports model and threshold selection, and test data measures the fixed selection's performance. Validation and test data are not resampled.

## Input features

The saved model and all three processed datasets agree on the following **32 features, in this order**. Exclude the target `Class` from prediction inputs.

```text
Time,
V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
V21, V22, V23, V24, V25, V26, V27, V28,
Amount, Hours, LogAmount
```

`V1` through `V28` are the dataset's anonymized PCA features. `Time` and `Amount` are retained alongside the derived features:

```python
Hours = ((Time // 3600) % 24).astype(int)
LogAmount = np.log1p(Amount)
```

`Hours` is derived from elapsed dataset time; it should not be interpreted as a verified local clock hour. Preserve feature names, order, and these transformations when preparing new transactions.

## Evaluation results

The following metrics were recomputed from the existing saved random forest and processed CSV files using threshold `0.5`, without retraining or changing the threshold. Precision, recall, and F1 refer to fraud (`Class=1`); accuracy covers both classes. Average precision (AP) uses continuous fraud probabilities.

**Reproducibility caveat:** the artifact reports scikit-learn **1.6.1**, while this verification used scikit-learn **1.8.0**, Python **3.13.7**, pandas **2.3.3**, and joblib **1.5.3**. Loading produced a version mismatch warning. Validation classification metrics match the saved notebook results, but the recomputed metrics should be confirmed in the original training environment before being treated as fully reproducible final results.

| Metric | Validation | Test |
| --- | ---: | ---: |
| Accuracy | 0.999542 | 0.999471 |
| Fraud precision | 0.875000 | 0.905405 |
| Fraud recall | 0.836957 | 0.744444 |
| Fraud F1 | 0.855556 | 0.817073 |
| Average precision | 0.837793 | 0.797788 |

Confusion matrices use actual classes as rows and predicted classes as columns, ordered `[legitimate, fraud]`:

| Split | True negatives | False positives | False negatives | True positives |
| --- | ---: | ---: | ---: | ---: |
| Validation | 56,642 | 11 | 15 | 77 |
| Test | 56,649 | 7 | 23 | 67 |

On the test set, the model detects **67 of 90 frauds**, misses **23**, and flags **7 legitimate transactions** as fraud.

### Average precision calculation

The current [evaluation script](src/evaluate.py) overwrites its probability array with binary predictions and passes those labels to `average_precision_score`. Its printed `PR-AUC` value of approximately **0.6744** on this test set therefore does not represent AP across the model's probability ranking. The probability-based test AP is **0.7978**. AP is also distinct from trapezoidal area under a precision-recall curve.

To reproduce the intended calculation, keep probabilities and predicted labels separate:

```python
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)

# Run from the repository root, using the training environment.
root = Path.cwd()
model = joblib.load(root / "models/random_forest_classifier.joblib")
test = pd.read_csv(root / "data/processed/test.csv")
X_test = test.loc[:, model.feature_names_in_]
y_test = test["Class"]

fraud_index = list(model.classes_).index(1)
y_scores = model.predict_proba(X_test)[:, fraud_index]
y_pred = (y_scores >= 0.5).astype(int)

print(classification_report(y_test, y_pred, digits=6, zero_division=0))
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
print("Average precision:", average_precision_score(y_test, y_scores))
```

## Limitations

- **Rare fraud cases:** validation contains only 92 frauds and test contains 90. A few changed predictions can materially affect recall and F1; confidence intervals and repeated validation have not been recorded. High accuracy alone is insufficient evidence of fraud detection quality.
- **Random split:** shuffled, unstratified splitting does not measure performance on future transactions or guarantee separation of related transactions. A chronological or appropriately grouped evaluation would better assess those settings.
- **Exploration before splitting:** EDA examines labels and feature relationships across the full dataset before the holdout is created. The test set was held out from model fitting and threshold selection, but not from all exploratory analysis.
- **Threshold objective:** selection maximizes F1 on a coarse grid. It does not explicitly optimize financial loss, investigation capacity, or separate costs of false alarms and missed fraud. Test results must not be used to retune this threshold.
- **Probability calibration:** no probability calibration assessment is recorded. Model scores should not automatically be interpreted as calibrated real-world fraud risks.
- **Interpretability:** anonymized PCA features limit explanations in business terms. The notebook records random forest feature importance; no completed SHAP analysis is saved there. Feature importance and SHAP contributions do not establish causation.
- **SMOTE comparison:** the logistic regression experiment and training script resample before invoking a pipeline that already contains SMOTE. The initial resampling happens before scaling; that experiment needs cleanup and fresh validation for a reliable comparison. The selected random forest does not use this path.
- **Reproducibility:** dependency versions are not pinned, the model-loading version mismatch remains unresolved, and `evaluate.py` still contains the AP calculation issue described above. The EDA notebook writes `feature.csv` to its working directory, whereas the split script expects `data/added_feature/feature.csv`; rebuilding requires placing that generated file at the expected path.
- **Generalization:** this evaluation covers the available dataset only. Performance under changes in fraud patterns, transaction populations, or fraud prevalence has not been established.
