
import os
import pickle
import logging

import pandas as pd
import yaml
from sklearn.feature_extraction.text import CountVectorizer


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# Load Parameters
# ============================================================

def load_params(path="params.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)


# ============================================================
# Load Data
# ============================================================

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if "content" not in df.columns or "sentiment" not in df.columns:
        raise ValueError(
            f"Expected 'content' and 'sentiment' columns in {path}"
        )

    return df


# ============================================================
# Feature Engineering
# ============================================================

def apply_bow(train_data, test_data, max_features):

    # Clean text
    train_text = train_data["content"].fillna("").astype(str)
    test_text = test_data["content"].fillna("").astype(str)

    # Clean labels
    train_labels = (
        train_data["sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    test_labels = (
        test_data["sentiment"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Convert labels
    label_mapping = {
        "sadness": 0,
        "happiness": 1
    }

    y_train = train_labels.map(label_mapping)
    y_test = test_labels.map(label_mapping)

    # Check for unknown labels
    if y_train.isna().any():
        unknown = train_labels[y_train.isna()].unique()
        raise ValueError(f"Unknown training labels: {unknown}")

    if y_test.isna().any():
        unknown = test_labels[y_test.isna()].unique()
        raise ValueError(f"Unknown test labels: {unknown}")

    # Convert to integers
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # ========================================================
    # Bag of Words
    # ========================================================

    vectorizer = CountVectorizer(
        max_features=max_features
    )

    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    logger.info(
        "BoW created | Train: %s | Test: %s",
        X_train.shape,
        X_test.shape
    )

    # ========================================================
    # Convert to DataFrame
    # ========================================================

    train_df = pd.DataFrame(
        X_train.toarray()
    )

    test_df = pd.DataFrame(
        X_test.toarray()
    )

    train_df["label"] = y_train.values
    test_df["label"] = y_test.values

    # ========================================================
    # Save Vectorizer
    # ========================================================

    os.makedirs("models", exist_ok=True)

    with open("mlops_project/src/model/vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)

    logger.info("Vectorizer saved successfully.")

    return train_df, test_df


# ============================================================
# Save Data
# ============================================================

def save_data(df, path):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    df.to_csv(
        path,
        index=False
    )

    logger.info(
        "Saved: %s | Shape: %s",
        path,
        df.shape
    )


# ============================================================
# Main
# ============================================================

def main():

    logger.info("Starting feature engineering...")

    # Load parameters
    params = load_params()

    max_features = params[
        "feature_engineering"
    ][
        "max_features"
    ]

    logger.info(
        "max_features = %s",
        max_features
    )

    # Input paths
    train_path = "data/interim/train_processed.csv"
    test_path = "data/interim/test_processed.csv"

    # Output paths
    train_output = "data/processed/train_bow.csv"
    test_output = "data/processed/test_bow.csv"

    # Load data
    train_data = load_data(train_path)
    test_data = load_data(test_path)

    logger.info(
        "Training data: %s",
        train_data.shape
    )

    logger.info(
        "Testing data: %s",
        test_data.shape
    )

    # Feature engineering
    train_df, test_df = apply_bow(
        train_data,
        test_data,
        max_features
    )

    # Save data
    save_data(
        train_df,
        train_output
    )

    save_data(
        test_df,
        test_output
    )

    logger.info(
        "Feature engineering completed successfully."
    )


if __name__ == "__main__":
    main()
