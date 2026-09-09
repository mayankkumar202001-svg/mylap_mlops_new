
# ============================================================
# HYPERPARAMETER TUNING - LOGISTIC REGRESSION
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================

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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# NLTK RESOURCES
# ============================================================

nltk.download("stopwords")
nltk.download("wordnet")


# ============================================================
# MLFLOW + DAGSHUB
# ============================================================

mlflow.set_tracking_uri(
    "https://dagshub.com/mayankkumar202001-svg/mylap_mlops_new.mlflow"
)

dagshub.init(
    repo_owner="mayankkumar202001-svg",
    repo_name="mylap_mlops_new",
    mlflow=True
)

mlflow.set_experiment(
    "LoR Hyperparameter Tuning"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "https://raw.githubusercontent.com/campusx-official/"
    "jupyter-masterclass/main/tweet_emotions.csv"
)

df = df.drop(
    columns=["tweet_id"]
)


# ============================================================
# TEXT PREPROCESSING
# ============================================================

lemmatizer = WordNetLemmatizer()

stop_words = set(
    stopwords.words("english")
)


def preprocess_text(text):

    # Convert to string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Remove punctuation
    text = re.sub(
        f"[{re.escape(string.punctuation)}]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

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
df["content"] = df["content"].apply(
    preprocess_text
)


# ============================================================
# FILTER SENTIMENTS
# ============================================================

df = df[
    df["sentiment"].isin(
        ["happiness", "sadness"]
    )
].copy()


# ============================================================
# CONVERT TARGET TO 0 AND 1
# ============================================================

df["sentiment"] = df["sentiment"].map({
    "sadness": 0,
    "happiness": 1
})


# Remove missing values
df = df.dropna(
    subset=["content", "sentiment"]
).copy()


# ============================================================
# VERY IMPORTANT:
# FORCE TARGET TO NORMAL NUMPY INTEGER ARRAY
# ============================================================

y = np.asarray(
    df["sentiment"].to_numpy(),
    dtype=np.int64
)


# ============================================================
# TEXT DATA
# ============================================================

X_text = df["content"].astype(str).to_numpy()


# ============================================================
# CHECK TARGET
# ============================================================

print("\n================ TARGET CHECK ================")

print(
    "Target dtype:",
    y.dtype
)

print(
    "Target classes:",
    np.unique(y)
)

print(
    "Target shape:",
    y.shape
)

print(
    "Class counts:",
    np.bincount(y)
)

print("==============================================\n")


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# CHECK TRAINING TARGET
# ============================================================

print(
    "y_train dtype:",
    y_train.dtype
)

print(
    "y_train classes:",
    np.unique(y_train)
)

print(
    "y_test dtype:",
    y_test.dtype
)

print(
    "y_test classes:",
    np.unique(y_test)
)


# ============================================================
# COUNT VECTORIZER
# ============================================================

vectorizer = CountVectorizer()


# IMPORTANT:
# Fit ONLY on training data
X_train = vectorizer.fit_transform(
    X_train_text
)


# Transform test data using the same vocabulary
X_test = vectorizer.transform(
    X_test_text
)


print(
    "\nX_train shape:",
    X_train.shape
)

print(
    "X_test shape:",
    X_test.shape
)


# ============================================================
# HYPERPARAMETER GRID
# ============================================================

param_grid = {

    "C": [
        0.1,
        1,
        10
    ],

    "penalty": [
        "l1",
        "l2"
    ],

    "solver": [
        "liblinear"
    ]
}


# ============================================================
# START PARENT MLFLOW RUN
# ============================================================

with mlflow.start_run(
    run_name="LR Hyperparameter Tuning"
):

    # ========================================================
    # GRID SEARCH
    # ========================================================

    grid_search = GridSearchCV(

        estimator=LogisticRegression(
            max_iter=1000
        ),

        param_grid=param_grid,

        cv=5,

        scoring="f1",

        n_jobs=-1,

        # If something fails, show the REAL error
        error_score="raise"
    )


    print(
        "\nStarting GridSearchCV..."
    )


    # ========================================================
    # FIT GRID SEARCH
    # ========================================================

    grid_search.fit(
        X_train,
        y_train
    )


    print(
        "\nGridSearchCV completed successfully."
    )


    # ========================================================
    # LOG EACH PARAMETER COMBINATION
    # ========================================================

    for params, mean_score, std_score in zip(

        grid_search.cv_results_["params"],

        grid_search.cv_results_["mean_test_score"],

        grid_search.cv_results_["std_test_score"]

    ):

        # ----------------------------------------------------
        # CHILD RUN
        # ----------------------------------------------------

        with mlflow.start_run(

            run_name=f"LR with params: {params}",

            nested=True

        ):

            # ------------------------------------------------
            # CREATE MODEL
            # ------------------------------------------------

            model = LogisticRegression(

                max_iter=1000,

                **params
            )


            # ------------------------------------------------
            # TRAIN MODEL
            # ------------------------------------------------

            model.fit(
                X_train,
                y_train
            )


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            y_pred = model.predict(
                X_test
            )


            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

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


            # ------------------------------------------------
            # LOG PARAMETERS
            # ------------------------------------------------

            mlflow.log_params(
                params
            )


            # ------------------------------------------------
            # LOG CV METRICS
            # ------------------------------------------------

            mlflow.log_metric(
                "mean_cv_score",
                float(mean_score)
            )

            mlflow.log_metric(
                "std_cv_score",
                float(std_score)
            )


            # ------------------------------------------------
            # LOG TEST METRICS
            # ------------------------------------------------

            mlflow.log_metric(
                "accuracy",
                float(accuracy)
            )

            mlflow.log_metric(
                "precision",
                float(precision)
            )

            mlflow.log_metric(
                "recall",
                float(recall)
            )

            mlflow.log_metric(
                "f1_score",
                float(f1)
            )


            # ------------------------------------------------
            # PRINT RESULTS
            # ------------------------------------------------

            print("\n--------------------------------------------")

            print(
                "Parameters:",
                params
            )

            print(
                f"Mean CV F1: {mean_score:.4f}"
            )

            print(
                f"Std CV F1: {std_score:.4f}"
            )

            print(
                f"Test Accuracy: {accuracy:.4f}"
            )

            print(
                f"Test Precision: {precision:.4f}"
            )

            print(
                f"Test Recall: {recall:.4f}"
            )

            print(
                f"Test F1: {f1:.4f}"
            )

            print("--------------------------------------------")


    # ========================================================
    # BEST MODEL
    # ========================================================

    best_params = grid_search.best_params_

    best_score = grid_search.best_score_

    best_model = grid_search.best_estimator_


    # ========================================================
    # LOG BEST PARAMETERS
    # ========================================================

    for key, value in best_params.items():

        mlflow.log_param(
            f"best_{key}",
            value
        )


    # ========================================================
    # LOG BEST SCORE
    # ========================================================

    mlflow.log_metric(
        "best_f1_score",
        float(best_score)
    )


    # ========================================================
    # EVALUATE BEST MODEL ON TEST DATA
    # ========================================================

    best_predictions = best_model.predict(
        X_test
    )


    best_accuracy = accuracy_score(
        y_test,
        best_predictions
    )

    best_precision = precision_score(
        y_test,
        best_predictions,
        zero_division=0
    )

    best_recall = recall_score(
        y_test,
        best_predictions,
        zero_division=0
    )

    best_f1 = f1_score(
        y_test,
        best_predictions,
        zero_division=0
    )


    # ========================================================
    # LOG BEST MODEL TEST METRICS
    # ========================================================

    mlflow.log_metric(
        "best_test_accuracy",
        float(best_accuracy)
    )

    mlflow.log_metric(
        "best_test_precision",
        float(best_precision)
    )

    mlflow.log_metric(
        "best_test_recall",
        float(best_recall)
    )

    mlflow.log_metric(
        "best_test_f1",
        float(best_f1)
    )


    # ========================================================
    # PRINT BEST MODEL
    # ========================================================

    print("\n============================================")

    print(
        "BEST MODEL"
    )

    print(
        "============================================"
    )

    print(
        "Best Parameters:",
        best_params
    )

    print(
        f"Best CV F1 Score: {best_score:.4f}"
    )

    print(
        f"Test Accuracy: {best_accuracy:.4f}"
    )

    print(
        f"Test Precision: {best_precision:.4f}"
    )

    print(
        f"Test Recall: {best_recall:.4f}"
    )

    print(
        f"Test F1 Score: {best_f1:.4f}"
    )


    # ========================================================
    # LOG BEST MODEL
    # ========================================================

    mlflow.sklearn.log_model(
        best_model,
        "model"
    )


    # ========================================================
    # LOG SCRIPT
    # ========================================================

    if "__file__" in globals():

        mlflow.log_artifact(
            __file__
        )


# ============================================================
# FINISHED
# ============================================================

print(
    "\nHyperparameter tuning completed successfully!"
)
