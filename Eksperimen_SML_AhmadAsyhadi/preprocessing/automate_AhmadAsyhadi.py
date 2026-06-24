from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data():
    current_dir = Path(__file__).resolve().parent
    base_dir = current_dir.parent

    raw_path = base_dir / "namadataset_raw" / "breast_cancer_raw.csv"

    output_dir = current_dir / "namadataset_preprocessing"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "breast_cancer_preprocessing.csv"

    df = pd.read_csv(raw_path)

    drop_columns = ["id", "Unnamed: 32"]
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    df["diagnosis"] = df["diagnosis"].replace({
        "B": 0,
        "M": 1,
        "benign": 0,
        "malignant": 1
    })

    df["diagnosis"] = pd.to_numeric(df["diagnosis"], errors="coerce")
    df = df.dropna(subset=["diagnosis"]).copy()
    df["diagnosis"] = df["diagnosis"].astype(int)

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    preprocessed_df = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    preprocessed_df["diagnosis"] = y.reset_index(drop=True)

    preprocessed_df.to_csv(output_path, index=False)

    print("Preprocessing selesai.")
    print(f"Input file : {raw_path}")
    print(f"Output file: {output_path}")
    print(f"Shape      : {preprocessed_df.shape}")


if __name__ == "__main__":
    preprocess_data()