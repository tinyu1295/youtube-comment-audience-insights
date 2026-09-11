"""Class-imbalance and feature-size sweep, tracked in MLflow.

Sentiment classes in this dataset are unbalanced (22% negative, 35% neutral,
43% positive), so the question is which imbalance strategy and vocabulary size
actually help. This runs every combination as a tracked MLflow run and writes a
ranked summary.

Selection is done on a validation split carved out of the training data. The
test set is never touched here — it is reserved for the final evaluation stage,
so the reported test metrics stay an honest estimate rather than the best of
fifty attempts.

Usage:
    python -m src.model.run_experiments
    python -m src.model.run_experiments --quick    # 2 strategies x 3 sizes
"""

import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.config import MLFLOW_TRACKING_URI, path
from src.logging_utils import get_logger

logger = get_logger('run_experiments', 'experiments_errors.log')

EXPERIMENT_NAME = 'imbalance-feature-sweep'

# Vocabulary sizes to sweep.
FEATURE_SIZES = [100, 250, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000]

# Class-imbalance strategies.
STRATEGIES = [
    'none',
    'class_weight_balanced',
    'is_unbalance',
    'random_oversample',
    'random_undersample',
]

QUICK_FEATURE_SIZES = [500, 1000, 2000]
QUICK_STRATEGIES = ['none', 'class_weight_balanced']


def resample_indices(y: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Return row indices that over- or under-sample classes to equal size."""
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max() if mode == 'over' else counts.min()

    picked = []
    for cls in classes:
        cls_idx = np.flatnonzero(y == cls)
        replace = mode == 'over' and len(cls_idx) < target
        picked.append(rng.choice(cls_idx, size=target, replace=replace))

    out = np.concatenate(picked)
    rng.shuffle(out)
    return out


def build_model(strategy: str, seed: int) -> lgb.LGBMClassifier:
    """Construct the classifier for one imbalance strategy.

    Hyperparameters other than the imbalance handling are held constant, so any
    difference in score is attributable to the strategy and vocabulary size.
    """
    kwargs = dict(
        objective='multiclass',
        num_class=3,
        metric='multi_logloss',
        learning_rate=0.09,
        max_depth=20,
        n_estimators=367,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        verbose=-1,
    )
    if strategy == 'class_weight_balanced':
        kwargs['class_weight'] = 'balanced'
    elif strategy == 'is_unbalance':
        kwargs['is_unbalance'] = True
    return lgb.LGBMClassifier(**kwargs)


def evaluate(model, X_val, y_val) -> dict:
    pred = model.predict(X_val)
    per_class = f1_score(y_val, pred, average=None, labels=[-1, 0, 1])
    return {
        'val_accuracy': accuracy_score(y_val, pred),
        'val_macro_f1': f1_score(y_val, pred, average='macro'),
        'val_weighted_f1': f1_score(y_val, pred, average='weighted'),
        'val_f1_negative': per_class[0],
        'val_f1_neutral': per_class[1],
        'val_f1_positive': per_class[2],
    }


def run_sweep(feature_sizes, strategies, seed: int = 42) -> pd.DataFrame:
    train = pd.read_csv(path('data', 'preprocessed', 'train_processed.csv')).fillna('')

    # Hold out a validation split; the test set stays untouched.
    text_train, text_val, y_train, y_val = train_test_split(
        train['clean_comment'].values,
        train['category'].values,
        test_size=0.2,
        random_state=seed,
        stratify=train['category'].values,
    )
    logger.debug('sweep on %d train / %d val rows', len(y_train), len(y_val))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    rows = []
    total = len(feature_sizes) * len(strategies)
    done = 0

    for max_features in feature_sizes:
        # Vectorise once per vocabulary size and reuse across strategies, so the
        # comparison is like-for-like and the sweep stays affordable.
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 3))
        X_train = vectorizer.fit_transform(text_train)
        X_val = vectorizer.transform(text_val)

        for strategy in strategies:
            done += 1
            started = time.time()

            Xs, ys = X_train, y_train
            if strategy in ('random_oversample', 'random_undersample'):
                mode = 'over' if strategy == 'random_oversample' else 'under'
                idx = resample_indices(y_train, mode, np.random.default_rng(seed))
                Xs, ys = X_train[idx], y_train[idx]

            model = build_model(strategy, seed)
            model.fit(Xs, ys)
            metrics = evaluate(model, X_val, y_val)
            elapsed = time.time() - started

            with mlflow.start_run(run_name=f'{strategy}_{max_features}'):
                mlflow.log_params({
                    'strategy': strategy,
                    'max_features': max_features,
                    'ngram_range': '1-3',
                    'n_train_rows': Xs.shape[0],
                })
                mlflow.log_metrics({**metrics, 'fit_seconds': elapsed})
                mlflow.set_tag('sweep', 'imbalance-x-features')

            rows.append({'strategy': strategy, 'max_features': max_features,
                         'fit_seconds': round(elapsed, 1), **metrics})
            logger.debug('[%d/%d] %-22s feat=%-6d macroF1=%.4f (%.0fs)',
                         done, total, strategy, max_features,
                         metrics['val_macro_f1'], elapsed)

    return pd.DataFrame(rows).sort_values('val_macro_f1', ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true',
                        help='small grid for a smoke test')
    args = parser.parse_args()

    feature_sizes = QUICK_FEATURE_SIZES if args.quick else FEATURE_SIZES
    strategies = QUICK_STRATEGIES if args.quick else STRATEGIES

    try:
        results = run_sweep(feature_sizes, strategies)
    except Exception as e:
        logger.error('Sweep failed: %s', e)
        sys.exit(1)

    out_dir = path('data', 'experiments')
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'sweep_results.csv')
    results.to_csv(csv_path, index=False)

    best = results.iloc[0]
    summary = {
        'best_strategy': best['strategy'],
        'best_max_features': int(best['max_features']),
        'best_val_macro_f1': round(float(best['val_macro_f1']), 4),
        'configurations_run': len(results),
    }
    with open(os.path.join(out_dir, 'sweep_best.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print('\n=== top 10 by validation macro F1 ===')
    print(results.head(10).to_string(index=False))
    print('\nbest:', json.dumps(summary))
    print('results written to', csv_path)


if __name__ == '__main__':
    main()
