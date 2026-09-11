"""Central configuration, resolved from environment variables with local defaults.

Keeping deployment-specific values (tracking server, artifact locations) out of
the source tree means the same code runs locally and on a remote host without
edits, and no infrastructure addresses are baked into version control.
"""

import os

# Repository root, resolved from this file so scripts work from any directory.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# MLflow tracking server. Defaults to a local ./mlruns store so a fresh clone
# runs end to end without access to any hosted infrastructure.
MLFLOW_TRACKING_URI = os.getenv(
    'MLFLOW_TRACKING_URI',
    f'file://{os.path.join(ROOT_DIR, "mlruns")}',
)

MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'dvc-pipeline-runs')

MODEL_NAME = os.getenv('MODEL_NAME', 'yt_comment_judge_plugin_model')

# Source dataset. A local path by default; override to pull from elsewhere.
RAW_DATA_PATH = os.getenv('RAW_DATA_PATH', os.path.join(ROOT_DIR, 'reddit.csv'))


def path(*parts: str) -> str:
    """Build an absolute path relative to the repository root."""
    return os.path.join(ROOT_DIR, *parts)
