import os
import sys

import pandas as pd

from src.config import path
from src.logging_utils import get_logger
from src.preprocessing import ensure_nltk_data, preprocess_comment

logger = get_logger('data_preprocessing', 'preprocessing_errors.log')


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to the text column of the dataframe."""
    try:
        df = df.copy()
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'preprocessed')
        os.makedirs(interim_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(
            interim_data_path, 'train_processed.csv'), index=False)
        test_data.to_csv(os.path.join(
            interim_data_path, 'test_processed.csv'), index=False)

        logger.debug('Processed data saved to %s', interim_data_path)
    except Exception as e:
        logger.error('Error occurred while saving data: %s', e)
        raise


def main():
    try:
        logger.debug('Starting data preprocessing...')
        ensure_nltk_data()

        train_data = pd.read_csv(path('data', 'raw', 'train.csv'))
        test_data = pd.read_csv(path('data', 'raw', 'test.csv'))
        logger.debug('Data loaded successfully')

        save_data(normalize_text(train_data),
                  normalize_text(test_data),
                  data_path=path('data'))
    except Exception as e:
        logger.error(
            'Failed to complete the data preprocessing process: %s', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
