"""
Automasi preprocessing dataset Breast Cancer.
Output: breast_cancer_preprocessing.csv siap dipakai untuk modelling.
"""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

RAW_PATH = Path(__file__).resolve().parents[1] / "namadataset_raw" / "breast_cancer_raw.csv"
OUT_DIR = Path(__file__).resolve().parent / "namadataset_preprocessing"
OUT_PATH = OUT_DIR / "breast_cancer_preprocessing.csv"


def preprocess(input_path: str | Path = RAW_PATH, output_path: str | Path = OUT_PATH) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)
    df = pd.read_csv(input_path)
    df = df.drop_duplicates().copy()

    # Handle missing value numerik dengan median.
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Encoding target.
    encoder = LabelEncoder()
    df["diagnosis"] = encoder.fit_transform(df["diagnosis"])  # benign=0, malignant=1 tergantung urutan encoder

    # Scaling feature.
    feature_cols = [c for c in df.columns if c != "diagnosis"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved to: {output_path}")
    return df


if __name__ == "__main__":
    preprocess()
