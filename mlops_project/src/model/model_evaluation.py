
import os
import json
import pickle
import logging

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
import dagshub

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# ============================================================
# DagsHub + MLflow configuration
# ============================================================

REPO_OWNER = "mayankkumar202001-svg"
REPO_NAME = "mylap_mlops_new"

dagshub.init(
    repo_owner=REPO_OWNER,
    repo_name=REPO_NAME,
    mlflow=True
)

mlflow.set_tracking_uri(
    f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"
)


# ============================================================
# Logging configuration
# ============================================================

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(
        "model_evaluation_errors.log"
    )
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ============================================================
# Load model
# ============================================================

def load_model(file_path: str):

    try:

        with open(file_path, "rb") as file:
            model = pickle.load(file)

        logger.info(
            "Model loaded from %s",
            file_path
        )

        return model

    except FileNotFoundError:

        logger.error(
            "Model file not found: %s",
            file_path
        )

        raise

    except Exception as e:

        logger.error(
            "Error loading model: %s",
            e
        )

        raise


# ============================================================
# Load data
# ============================================================

def load_data(file_path: str) -> pd.DataFrame:

    try:

        df = pd.read_csv(file_path)

        logger.info(
            "Data loaded from %s",
            file_path
        )

        return df

    except FileNotFoundError:

        logger.error(
            "Data file not found: %s",
            file_path
        )

        raise

    except pd.errors.ParserError as e:

        logger.error(
            "Failed to parse CSV: %s",
            e
        )

        raise

    except Exception as e:

        logger.error(
            "Error loading data: %s",
            e
        )

        raise


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:

    try:

        # Predictions
        y_pred = clf.predict(X_test)

        # Basic classification metrics
        metrics = {
            "accuracy": accuracy_score(
                y_test,
                y_pred
            ),

            "precision": precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "recall": recall_score(
                y_test,
                y_pred,
                zero_division=0
            )
        }

        # AUC
        if hasattr(clf, "predict_proba"):

            y_pred_proba = clf.predict_proba(
                X_test
            )[:, 1]

            metrics["auc"] = roc_auc_score(
                y_test,
                y_pred_proba
            )

        logger.info(
            "Evaluation metrics: %s",
            metrics
        )

        return metrics

    except Exception as e:

        logger.error(
            "Error during model evaluation: %s",
            e
        )

        raise


# ============================================================
# Save metrics
# ============================================================

def save_metrics(
    metrics: dict,
    file_path: str
) -> None:

    try:

        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        with open(file_path, "w") as file:

            json.dump(
                metrics,
                file,
                indent=4
            )

        logger.info(
            "Metrics saved to %s",
            file_path
        )

    except Exception as e:

        logger.error(
            "Error saving metrics: %s",
            e
        )

        raise


# ============================================================
# Save experiment information
# ============================================================

def save_experiment_info(
    run_id: str,
    model_uri: str,
    file_path: str
) -> None:

    try:

        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        experiment_info = {
            "run_id": run_id,
            "model_uri": model_uri
        }

        with open(file_path, "w") as file:

            json.dump(
                experiment_info,
                file,
                indent=4
            )

        logger.info(
            "Experiment information saved to %s",
            file_path
        )

    except Exception as e:

        logger.error(
            "Error saving experiment information: %s",
            e
        )

        raise


# ============================================================
# Main
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # Set MLflow experiment
        # ----------------------------------------------------

        mlflow.set_experiment(
            "dvc-pipeline"
        )

        # ----------------------------------------------------
        # Start MLflow run
        # ----------------------------------------------------

        with mlflow.start_run() as run:

            logger.info(
                "MLflow run started: %s",
                run.info.run_id
            )

            # ------------------------------------------------
            # Load model
            # ------------------------------------------------

            clf = load_model(
                "mlops_project/src/model/model.pkl"
            )

            # ------------------------------------------------
            # Load test data
            # ------------------------------------------------

            test_data = load_data(
                "data/processed/test_bow.csv"
            )

            # Features
            X_test = test_data.iloc[:, :-1].values

            # Target
            y_test = (
                test_data.iloc[:, -1]
                .astype(int)
                .values
            )

            # ------------------------------------------------
            # Evaluate model
            # ------------------------------------------------

            metrics = evaluate_model(
                clf,
                X_test,
                y_test
            )

            # ------------------------------------------------
            # Save metrics locally
            # ------------------------------------------------

            metrics_path = (
                "mlops_project/reports/metrics.json"
            )

            save_metrics(
                metrics,
                metrics_path
            )

            # ------------------------------------------------
            # Log metrics to MLflow
            # ------------------------------------------------

            mlflow.log_metrics(
                metrics
            )

            # ------------------------------------------------
            # Log model parameters
            # ------------------------------------------------

            if hasattr(clf, "get_params"):

                params = clf.get_params()

                mlflow.log_params(
                    {
                        key: str(value)
                        for key, value in params.items()
                    }
                )

            # ------------------------------------------------
            # Log model to MLflow
            #
            # MLflow 3.x:
            # Use ModelInfo.model_uri instead of manually
            # constructing runs:/<run_id>/model
            # ------------------------------------------------

            model_info = mlflow.sklearn.log_model(
                sk_model=clf,
                name="model"
            )

            logger.info(
                "Model logged successfully."
            )

            logger.info(
                "Model URI: %s",
                model_info.model_uri
            )

            # ------------------------------------------------
            # Save experiment information
            # ------------------------------------------------

            experiment_info_path = (
                "mlops_project/reports/experiment_info.json"
            )

            save_experiment_info(
                run_id=run.info.run_id,
                model_uri=model_info.model_uri,
                file_path=experiment_info_path
            )

            # ------------------------------------------------
            # Log artifacts
            # ------------------------------------------------

            mlflow.log_artifact(
                metrics_path
            )

            mlflow.log_artifact(
                experiment_info_path
            )

            # ------------------------------------------------
            # Log error log if it exists
            # ------------------------------------------------

            if os.path.exists(
                "model_evaluation_errors.log"
            ):

                mlflow.log_artifact(
                    "model_evaluation_errors.log"
                )

            logger.info(
                "Model evaluation completed successfully."
            )


    except Exception as e:

        logger.exception(
            "Failed to complete model evaluation: %s",
            e
        )

        raise


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
