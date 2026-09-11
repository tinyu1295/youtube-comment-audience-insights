"""Tests for the ingestion stage's cleaning and parameter loading."""

import pandas as pd
import pytest
import yaml

from src.data.data_ingestion import load_data, load_params, preprocess_data


def test_preprocess_data_drops_nulls_duplicates_and_blanks():
    df = pd.DataFrame({
        'clean_comment': ['good', 'good', None, '   ', 'bad'],
        'category': [1, 1, 0, 0, -1],
    })

    result = preprocess_data(df)

    assert sorted(result['clean_comment']) == ['bad', 'good']


def test_preprocess_data_does_not_mutate_input():
    df = pd.DataFrame({'clean_comment': ['a', 'a'], 'category': [1, 1]})
    preprocess_data(df)
    assert len(df) == 2


def test_preprocess_data_raises_on_missing_column():
    df = pd.DataFrame({'wrong_column': ['a'], 'category': [1]})
    with pytest.raises(KeyError):
        preprocess_data(df)


def test_load_params_reads_yaml(tmp_path):
    params_file = tmp_path / 'params.yaml'
    params_file.write_text(yaml.safe_dump({'data_ingestion': {'test_size': 0.2}}))

    assert load_params(str(params_file))['data_ingestion']['test_size'] == 0.2


def test_load_params_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_params(str(tmp_path / 'nope.yaml'))


def test_load_data_reads_csv(tmp_path):
    csv = tmp_path / 'data.csv'
    csv.write_text('clean_comment,category\nhello,1\n')

    df = load_data(str(csv))

    assert list(df.columns) == ['clean_comment', 'category']
    assert len(df) == 1


def test_project_params_yaml_matches_pipeline_expectations():
    """The committed params.yaml must supply every key the stages read."""
    from src.config import path

    with open(path('params.yaml')) as f:
        params = yaml.safe_load(f)

    assert 0 < params['data_ingestion']['test_size'] < 1
    for key in ('ngram_range', 'max_features', 'learning_rate',
                'max_depth', 'n_estimators'):
        assert key in params['model_building'], f'missing {key}'
