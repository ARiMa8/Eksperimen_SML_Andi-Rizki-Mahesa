import os
import re
import json
import argparse
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def clean_column_name(name):
    name = str(name).strip().lower()
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def preprocess_data(input_path, output_dir, test_size=0.2, random_state=42):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    df.columns = [clean_column_name(col) for col in df.columns]

    if "class" in df.columns:
        df = df.rename(columns={"class": "credit_risk"})

    target_col = "credit_risk"

    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan pada dataset.")

    df = df.drop_duplicates()
    df[target_col] = df[target_col].astype(str).str.lower()

    target_mapping = {
        "good": 1,
        "bad": 0,
        "1": 1,
        "2": 0
    }

    df[target_col] = df[target_col].map(target_mapping)

    if df[target_col].isnull().sum() > 0:
        raise ValueError("Target masih memiliki nilai yang tidak berhasil dipetakan.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    try:
        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot_encoder)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    feature_names = [clean_column_name(col) for col in feature_names]

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    y_train_df = pd.DataFrame({"credit_risk": y_train.reset_index(drop=True)})
    y_test_df = pd.DataFrame({"credit_risk": y_test.reset_index(drop=True)})

    train_combined = pd.concat([X_train_df, y_train_df], axis=1)
    test_combined = pd.concat([X_test_df, y_test_df], axis=1)
    full_preprocessed = pd.concat([train_combined, test_combined], axis=0, ignore_index=True)

    X_train_df.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test_df.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train_df.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test_df.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    full_preprocessed.to_csv(os.path.join(output_dir, "credit_scoring_preprocessed.csv"), index=False)

    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))

    metadata = {
        "dataset_name": "credit-g / Statlog German Credit Data",
        "source": "OpenML data_id=31, originally from UCI Machine Learning Repository",
        "raw_shape": list(df.shape),
        "processed_shape": list(full_preprocessed.shape),
        "train_shape": list(X_train_df.shape),
        "test_shape": list(X_test_df.shape),
        "target_column": target_col,
        "target_mapping": target_mapping,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "random_state": random_state,
        "test_size": test_size
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("Preprocessing selesai.")
    print(f"Input path: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Train shape: {X_train_df.shape}")
    print(f"Test shape: {X_test_df.shape}")

    return full_preprocessed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="credit_scoring_raw/credit_g.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="preprocessing/credit_scoring_preprocessing"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42
    )

    args = parser.parse_args()

    preprocess_data(
        input_path=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.random_state
    )
