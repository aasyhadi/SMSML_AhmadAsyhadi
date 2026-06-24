import json
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import dagshub

DATA_PATH = Path(__file__).resolve().parent / "namadataset_preprocessing" / "breast_cancer_preprocessing.csv"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

dagshub.init(
    repo_owner="aasyhadi",
    repo_name="smsml-breast-cancer-ahmadasyhadi",
    mlflow=True
)

mlflow.set_experiment("SMSML Breast Cancer Tuning - AhmadAsyhadi")


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    params = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=3, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    cm_path = ARTIFACT_DIR / "training_confusion_matrix.png"
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    report_path = ARTIFACT_DIR / "classification_report.json"
    report_path.write_text(json.dumps(classification_report(y_test, y_pred, output_dict=True), indent=2))
    metric_path = ARTIFACT_DIR / "metric_info.json"
    metric_path.write_text(json.dumps(metrics, indent=2))

    with mlflow.start_run(run_name="tuned_random_forest_manual_logging"):
        mlflow.log_params(grid.best_params_)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(metric_path))

        input_example = X_test.head(5)

        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            input_example=input_example,
            registered_model_name="smsml_breast_cancer_model"
        )

        print(json.dumps({"best_params": grid.best_params_, "metrics": metrics}, indent=2))

if __name__ == "__main__":
    main()
