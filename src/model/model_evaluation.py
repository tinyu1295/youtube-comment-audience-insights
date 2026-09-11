import json
import os
import sys

import matplotlib
matplotlib.use('Agg')  # non-interactive backend, required for headless runs

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pickle  # noqa: E402
import seaborn as sns  # noqa: E402
import yaml  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

from src.config import (MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI,  # noqa: E402
                        path)
from src.logging_utils import get_logger  # noqa: E402

logger = get_logger('model_evaluation', 'model_evaluation_errors.log')


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s',
                     vectorizer_path, e)
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and log classification metrics and confusion matrix."""
    try:
        # Predict and calculate classification metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        logger.debug('Model evaluation completed')

        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save confusion matrix plot as a file and log it to MLflow
    os.makedirs(path('data', 'evaluation'), exist_ok=True)
    cm_file_path = path(
        'data', 'evaluation', f'confusion_matrix_{dataset_name}.png')
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        # Create a dictionary with the info you want to save
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        # Save the dictionary as a JSON file
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    os.makedirs(path('data', 'evaluation'), exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        try:
            params = load_params(path('params.yaml'))

            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)

            # Load model and vectorizer
            model = load_model(path('models', 'trained_model', 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(path('models', 'vectorizer_model', 'tfidf_vectorizer.pkl'))

            # Load test data for signature inference
            test_data = load_data(path('data', 'preprocessed', 'test_processed.csv'))

            # Prepare test data
            X_test_tfidf = vectorizer.transform(
                test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Create a DataFrame for signature inference (using first few rows as an example)
            input_example = pd.DataFrame(X_test_tfidf.toarray(
            )[:5], columns=vectorizer.get_feature_names_out())

            # Infer the signature
            signature = infer_signature(
                input_example, model.predict(X_test_tfidf[:5]))

            # Log model with signature
            mlflow.sklearn.log_model(
                model,
                "lgbm_model",
                signature=signature,
                input_example=input_example,
            )

            artifact_uri = mlflow.get_artifact_uri()
            save_model_info(
                run.info.run_id,
                f"{artifact_uri}/lgbm_model",
                path('data', 'evaluation', 'experiment_info.json'),
            )

            # Log the vectorizer as an artifact
            mlflow.log_artifact(path('models', 'vectorizer_model', 'tfidf_vectorizer.pkl'))

            # Evaluate model and get metrics
            report, cm = evaluate_model(model, X_test_tfidf, y_test)

            # Log classification report metrics for the test data
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            # Headline metrics under stable, queryable names. The keys above are
            # derived from classification_report labels and contain spaces, which
            # makes them awkward to filter on; registration selects on these.
            mlflow.log_metrics({
                'test_accuracy': report['accuracy'],
                'test_macro_f1': report['macro avg']['f1-score'],
                'test_weighted_f1': report['weighted avg']['f1-score'],
            })

            # Log confusion matrix
            log_confusion_matrix(cm, "Test Data")

            # Add important tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

        except Exception as e:
            logger.error('Failed to complete model evaluation: %s', e)
            sys.exit(1)


if __name__ == '__main__':
    main()
