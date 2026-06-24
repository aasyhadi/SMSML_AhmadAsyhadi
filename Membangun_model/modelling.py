import json
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATA_PATH = Path(__file__).resolve().parent / "namadataset_preprocessing" / "breast_cancer_preprocessing.csv"
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("SMSML Breast Cancer - Ahmad Asyhadi")
mlflow.sklearn.autolog()

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    with mlflow.start_run(run_name="baseline_random_forest_autolog"):
        model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }
        print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
