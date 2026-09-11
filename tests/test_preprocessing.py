"""Tests for the text normalisation shared by training and serving."""

import pytest

from src.preprocessing import preprocess_comment


def test_lowercases_and_strips():
    assert preprocess_comment('  HELLO World  ') == 'hello world'


def test_collapses_newlines():
    assert '\n' not in preprocess_comment('first line\nsecond line')


def test_strips_html_and_symbols_but_keeps_sentence_punctuation():
    result = preprocess_comment('Great video! Really? Yes, indeed. @#$%^&*')
    assert '@' not in result and '#' not in result
    assert '!' in result and '?' in result and ',' in result


def test_removes_common_stopwords():
    # "the" and "is" carry no sentiment and should be dropped
    result = preprocess_comment('the video is here')
    assert 'the' not in result.split()
    assert 'is' not in result.split()


@pytest.mark.parametrize('word', ['not', 'but', 'however', 'no', 'yet'])
def test_retains_sentiment_bearing_stopwords(word):
    """Negations invert meaning, so they must survive stopword removal."""
    assert word in preprocess_comment(f'this {word} good').split()


def test_lemmatises_plurals():
    assert 'dog' in preprocess_comment('dogs').split()


def test_handles_empty_string():
    assert preprocess_comment('') == ''


def test_is_idempotent():
    """Running twice must not change the result, or repeated calls would drift."""
    once = preprocess_comment('The running dogs were not amazing!')
    assert preprocess_comment(once) == once


def test_non_string_input_does_not_raise():
    """A malformed row must not abort a whole batch."""
    assert preprocess_comment(None) is None
