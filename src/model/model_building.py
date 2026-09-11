import os
import pickle
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import path
from src.logging_utils import get_logger

logger = get_logger('model_building', 'model_building_errors.log')


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """Apply TF-IDF with ngrams to the data."""
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range)

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        # Perform TF-IDF transformation
        X_train_tfidf = vectorizer.fit_transform(X_train)

        logger.debug(
            f"TF-IDF transformation complete. Train shape: {X_train_tfidf.shape}")

        vectorizer_path = path(
            'models', 'vectorizer_model', 'tfidf_vectorizer.pkl')
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        # Save the vectorizer in the root directory
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.debug('TF-IDF applied with trigrams and data transformed')
        return X_train_tfidf, y_train
    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise


def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:
    """Train a LightGBM model."""
    try:
        best_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric="multi_logloss",
            # class_weight alone: the sweep showed is_unbalance is a no-op for
            # multiclass objectives (identical scores at all 10 feature sizes).
            class_weight="balanced",
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        best_model.fit(X_train, y_train)
        logger.debug('LightGBM model training completed')
        return best_model
    except Exception as e:
        logger.error('Error during LightGBM model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def main():
    try:
        params = load_params(path('params.yaml'))
        max_features = params['model_building']['max_features']
        ngram_range = tuple(params['model_building']['ngram_range'])

        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']

        # Load the preprocessed training data from the interim directory
        train_data = load_data(
            path('data', 'preprocessed', 'train_processed.csv'))

        # Apply TF-IDF feature engineering on training data
        X_train_tfidf, y_train = apply_tfidf(
            train_data, max_features, ngram_range)

        # Train the LightGBM model using hyperparameters from params.yaml
        best_model = train_lgbm(X_train_tfidf, y_train,
                                learning_rate, max_depth, n_estimators)

        # Save the trained model in the root directory
        save_model(best_model,
                   path('models', 'trained_model', 'lgbm_model.pkl'))

    except Exception as e:
        logger.error(
            'Failed to complete the feature engineering and model building process: %s', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
