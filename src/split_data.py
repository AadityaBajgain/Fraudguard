from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

df = pd.read_csv("data/added_feature/feature.csv")
X = df.drop(columns=["Class"])
y = df["Class"]

print(X.shape)
print(y.shape)
# print(df.head())


X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    random_state=42,
    test_size=0.25
)


output_dir = DATA_DIR / "processed"
output_dir.mkdir(parents = True, exist_ok = True)



for name, features, labels in [
    ("train", X_train, y_train),
    ("val", X_val, y_val),
    ("test",X_test, y_test)
]:
    result = features.copy()
    result["Class"] = labels
    result.to_csv(output_dir / f"{name}.csv", index = False)

print("Training Set: ", X_train.shape)
print("Validation Set: ", X_val.shape)
print("Test Set: ", X_test.shape)


print("\nTraining target: ", y_train.value_counts())
print("Validation Target: ", y_val.value_counts())
print("Test Target: ", y_test.value_counts())