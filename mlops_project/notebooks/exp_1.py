
# ============================================
# BoW vs TF-IDF
# ============================================

# Import libraries
import os
import re
import string
import numpy as np
import pandas as pd
import nltk
import mlflow
import mlflow.sklearn
import dagshub

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier


# ============================================
# NLTK downloads
# ============================================

nltk.download("stopwords")
nltk.download("wordnet")


# ============================================
# MLflow + DagsHub
# ============================================

mlflow.set_tracking_uri(
    "https://dagshub.com/mayankkumar202001-svg/mylap_mlops_new.mlflow"
)

dagshub.init(
    repo_owner="mayankkumar202001-svg",
    repo_name="mylap_mlops_new",
    mlflow=True
)

mlflow.set_experiment("Bow vs TfIdf")


# ============================================
# Load dataset
# ============================================

df = pd.read_csv(
    "https://raw.githubusercontent.com/campusx-official/"
    "jupyter-masterclass/main/tweet_emotions.csv"
)

df = df.drop(columns=["tweet_id"])

print("Original shape:", df.shape)
print(df.head())


# ============================================
# Text preprocessing
# ============================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):

    # Convert to string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = re.sub(
        f"[{re.escape(string.punctuation)}]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = [
        word
        for word in text.split()
        if word not in stop_words
    ]

    # Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


# Apply preprocessing
df["content"] = df["content"].apply(preprocess_text)


# ============================================
# Select only happiness and sadness
# ============================================

df = df[
    df["sentiment"].isin(["happiness", "sadness"])
].copy()


# ============================================
# Convert target into binary labels
# ============================================

df["sentiment"] = df["sentiment"].map({
    "sadness": 0,
    "happiness": 1
})


# Remove missing values
df = df.dropna(
    subset=["content", "sentiment"]
).copy()


# Make sure target is a normal integer type
df["sentiment"] = df["sentiment"].astype(int)


print("\nFinal dataset shape:", df.shape)
print("\nTarget values:")
print(df["sentiment"].value_counts())


# ============================================
# Define X and y
# ============================================

X_text = df["content"].astype(str)

y = df["sentiment"].to_numpy(dtype=int)


# ============================================
# Feature extraction methods
# ============================================

vectorizers = {

    "BoW": CountVectorizer(),

    "TF-IDF": TfidfVectorizer()

}


# ============================================
# Machine Learning algorithms
# ============================================

algorithms = {

    "LogisticRegression":
        LogisticRegression(max_iter=1000),

    "MultinomialNB":
        MultinomialNB(),

    "XGBoost":
        XGBClassifier(
            eval_metric="logloss",
            random_state=42
        ),

    "RandomForest":
        RandomForestClassifier(
            random_state=42
        ),

    "GradientBoosting":
        GradientBoostingClassifier(
            random_state=42
        )
}


# ============================================
# Parent MLflow Run
# ============================================

with mlflow.start_run(
    run_name="All Experiments"
):

    # ========================================
    # Loop through algorithms
    # ========================================

    for algo_name, algorithm in algorithms.items():

        # ====================================
        # Loop through vectorizers
        # ====================================

        for vec_name, vectorizer in vectorizers.items():

            with mlflow.start_run(
                run_name=f"{algo_name} with {vec_name}",
                nested=True
            ):

                print("\n" + "=" * 50)
                print(
                    f"Algorithm: {algo_name}"
                )
                print(
                    f"Feature Engineering: {vec_name}"
                )
                print("=" * 50)


                # ====================================
                # Vectorization
                # ====================================

                X = vectorizer.fit_transform(
                    X_text
                )


                # ====================================
                # Train-test split
                # ====================================

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y
                )


                # ====================================
                # Log experiment parameters
                # ====================================

                mlflow.log_param(
                    "vectorizer",
                    vec_name
                )

                mlflow.log_param(
                    "algorithm",
                    algo_name
                )

                mlflow.log_param(
                    "test_size",
                    0.2
                )

                mlflow.log_param(
                    "random_state",
                    42
                )


                # ====================================
                # Train model
                # ====================================

                model = algorithm

                model.fit(
                    X_train,
                    y_train
                )


                # ====================================
                # Log model parameters
                # ====================================

                if algo_name == "LogisticRegression":

                    mlflow.log_param(
                        "C",
                        model.C
                    )

                elif algo_name == "MultinomialNB":

                    mlflow.log_param(
                        "alpha",
                        model.alpha
                    )

                elif algo_name == "XGBoost":

                    mlflow.log_param(
                        "n_estimators",
                        model.n_estimators
                    )

                    mlflow.log_param(
                        "learning_rate",
                        model.learning_rate
                    )

                elif algo_name == "RandomForest":

                    mlflow.log_param(
                        "n_estimators",
                        model.n_estimators
                    )

                    mlflow.log_param(
                        "max_depth",
                        model.max_depth
                    )

                elif algo_name == "GradientBoosting":

                    mlflow.log_param(
                        "n_estimators",
                        model.n_estimators
                    )

                    mlflow.log_param(
                        "learning_rate",
                        model.learning_rate
                    )

                    mlflow.log_param(
                        "max_depth",
                        model.max_depth
                    )


                # ====================================
                # Predictions
                # ====================================

                y_pred = model.predict(
                    X_test
                )


                # ====================================
                # Evaluation
                # ====================================

                accuracy = accuracy_score(
                    y_test,
                    y_pred
                )

                precision = precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )


                # ====================================
                # Log metrics
                # ====================================

                mlflow.log_metric(
                    "accuracy",
                    accuracy
                )

                mlflow.log_metric(
                    "precision",
                    precision
                )

                mlflow.log_metric(
                    "recall",
                    recall
                )

                mlflow.log_metric(
                    "f1_score",
                    f1
                )


                # ====================================
                # Log model
                # ====================================

                if algo_name == "XGBoost":
                 mlflow.sklearn.log_model(
                   model,
                     "model",
                skops_trusted_types=[
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier"
                    ]
                    )
                else:
                 mlflow.sklearn.log_model(
                 model,
                 "model"
                      )


                # ====================================
                # Log notebook/script
                # ====================================

                if "__file__" in globals():
                    mlflow.log_artifact(
                        __file__
                    )


                # ====================================
                # Print results
                # ====================================

                print(
                    f"Accuracy  : {accuracy:.4f}"
                )

                print(
                    f"Precision : {precision:.4f}"
                )

                print(
                    f"Recall    : {recall:.4f}"
                )

                print(
                    f"F1 Score  : {f1:.4f}"
                )

