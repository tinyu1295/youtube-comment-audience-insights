"""Register the best evaluated model in the MLflow Model Registry.

Registration is gated on a metric rather than on recency. The evaluation stage
writes the run it just produced, but promoting that run unconditionally means a
worse model silently replaces a better one whenever the pipeline is re-run. This
compares the candidate against every previous run in the experiment and only
promotes it if it genuinely wins.
"""

import json
import os
import sys

import mlflow

from src.config import (MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_NAME,
                        path)
from src.logging_utils import get_logger

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logger = get_logger('model_registration', 'model_registration_errors.log')

# Alias applied to the winning version. Aliases replace the stage transitions
# deprecated in MLflow 2.9 and removed in 3.
STAGING_ALIAS = 'staging'

# Metric that decides which run is "best". Macro F1 weights all three sentiment
# classes equally, so a model that ignores the minority negative class cannot
# win on overall accuracy alone.
SELECTION_METRIC = os.getenv('SELECTION_METRIC', 'test_macro_f1')


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


def find_best_run(experiment_name: str, metric: str):
    """Return the run in `experiment_name` with the highest `metric`.

    Returns None if the experiment or the metric cannot be found, so the caller
    can fall back rather than fail the pipeline.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning('Experiment %s not found', experiment_name)
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f'metrics.`{metric}` > 0',
        order_by=[f'metrics.`{metric}` DESC'],
        max_results=1,
    )
    if not runs:
        logger.warning('No runs in %s carry metric %s', experiment_name, metric)
        return None

    best = runs[0]
    logger.debug('Best run %s with %s=%.4f', best.info.run_id, metric,
                 best.data.metrics[metric])
    return best


def select_run(candidate: dict, experiment_name: str, metric: str) -> str:
    """Choose between the just-evaluated run and the best historical run."""
    best = find_best_run(experiment_name, metric)
    if best is None:
        logger.warning(
            'Falling back to the candidate run; no metric comparison possible')
        return candidate['run_id']

    if best.info.run_id == candidate['run_id']:
        logger.debug('Candidate run is the best run')
    else:
        logger.debug(
            'Candidate %s did not win; promoting %s instead',
            candidate['run_id'], best.info.run_id)
    return best.info.run_id


def register_model(model_name: str, run_id: str):
    """Register the run and point the staging alias at the new version."""
    try:
        model_version = mlflow.register_model(
            f'runs:/{run_id}/lgbm_model', model_name)

        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias=STAGING_ALIAS,
            version=model_version.version,
        )

        logger.debug('Model %s version %s registered and aliased as %s.',
                     model_name, model_version.version, STAGING_ALIAS)
        return model_version
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise


def main():
    try:
        candidate = load_model_info(
            path('data', 'evaluation', 'experiment_info.json'))
        run_id = select_run(candidate, MLFLOW_EXPERIMENT_NAME, SELECTION_METRIC)
        register_model(MODEL_NAME, run_id)
    except Exception as e:
        logger.error(
            'Failed to complete the model registration process: %s', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
