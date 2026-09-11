"""Text preprocessing shared by the training pipeline and the inference API.

This lives in one place deliberately: if training and serving normalise text
differently the model sees a different distribution at inference time than it
was trained on, which degrades predictions silently.
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Negations and contrastive conjunctions carry sentiment, so they are kept
# rather than dropped with the rest of the stopwords.
_SENTIMENT_BEARING_STOPWORDS = {'not', 'but', 'however', 'no', 'yet'}

_lemmatizer = None
_stop_words = None
_nltk_ready = False


def ensure_nltk_data() -> None:
    """Download the NLTK corpora this module needs, if they are missing."""
    global _nltk_ready
    if _nltk_ready:
        return

    for resource in ('wordnet', 'stopwords'):
        if not _has_corpus(resource):
            nltk.download(resource, quiet=True)
    _nltk_ready = True


def _has_corpus(resource: str) -> bool:
    """Check for a corpus, which NLTK may store unpacked or as a zip."""
    for candidate in (f'corpora/{resource}', f'corpora/{resource}.zip'):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return False


def _get_lemmatizer() -> WordNetLemmatizer:
    global _lemmatizer
    if _lemmatizer is None:
        ensure_nltk_data()
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


def _get_stop_words() -> set:
    global _stop_words
    if _stop_words is None:
        ensure_nltk_data()
        _stop_words = set(stopwords.words('english')) - _SENTIMENT_BEARING_STOPWORDS
    return _stop_words


def preprocess_comment(comment: str) -> str:
    """Normalise a single comment: lowercase, strip noise, drop stopwords, lemmatise."""
    try:
        comment = comment.lower().strip()

        # Collapse newlines into spaces
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except sentence punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = _get_stop_words()
        comment = ' '.join(
            word for word in comment.split() if word not in stop_words)

        lemmatizer = _get_lemmatizer()
        comment = ' '.join(lemmatizer.lemmatize(word)
                           for word in comment.split())

        return comment
    except Exception:
        # A single malformed comment must not abort a whole batch.
        return comment
