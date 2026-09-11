"""Register the evaluated model in the MLflow Model Registry."""

import json
import sys

import mlflow

from src.config import MLFLOW_TRACKING_URI, MODEL_NAME, path
from src.logging_utils import get_logger

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logger = get_logger('model_registration', 'model_registration_errors.log')

# Stage the newly registered version under this alias. Aliases replace the
# deprecated stage transitions removed in MLflow 3.
STAGING_ALIAS = 'staging'


def load_model_info(file_path: str) -> dict:
    """Load the model info written by the evaluation stage."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error(
            'Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model and point the staging alias at the new version."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/lgbm_model"
        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias=STAGING_ALIAS,
            version=model_version.version,
        )

        logger.debug(
            'Model %s version %s registered and aliased as %s.',
            model_name, model_version.version, STAGING_ALIAS)
        return model_version
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise


def main():
    try:
        model_info = load_model_info(
            path('data', 'evaluation', 'experiment_info.json'))
        register_model(MODEL_NAME, model_info)
    except Exception as e:
        logger.error(
            'Failed to complete the model registration process: %s', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
