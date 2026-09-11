import os
import sys

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.config import RAW_DATA_PATH, path
from src.logging_utils import get_logger

logger = get_logger('data_ingestion', 'data_ingestion_errors.log')


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


def load_data(data_source: str) -> pd.DataFrame:
    """Load the raw dataset from a local path or URL."""
    try:
        df = pd.read_csv(data_source)
        logger.debug('Data loaded from %s', data_source)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing values, duplicates, and blank comments."""
    try:
        df = df.dropna()
        df = df.drop_duplicates()
        df = df[df['clean_comment'].str.strip() != '']

        logger.debug(
            'Data preprocessing completed: %d rows retained', len(df))
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets, creating the raw folder if needed."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(
            raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)

        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        params = load_params(path('params.yaml'))
        test_size = params['data_ingestion']['test_size']
        random_state = params['data_ingestion'].get('random_state', 42)

        df = load_data(RAW_DATA_PATH)
        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=random_state)

        save_data(train_data, test_data, data_path=path('data'))

    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
