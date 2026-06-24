import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "namadataset_preprocessing" / "breast_cancer_preprocessing.csv"

mlflow.sklearn.autolog()

def main():

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Training selesai")

if __name__ == "__main__":
    main()