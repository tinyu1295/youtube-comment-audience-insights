FROM python:3.11-slim

# libgomp is LightGBM's OpenMP runtime; the wheel links against it at import
# time and fails with an opaque error if it is missing from a slim image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a source change does not invalidate the pip layer.
COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt gunicorn==23.0.0

# Bake the NLTK corpora into the image rather than downloading them on the
# first request, which would make a cold start fail without network access.
ENV NLTK_DATA=/usr/share/nltk_data
RUN python -c "import nltk; nltk.download('wordnet', download_dir='$NLTK_DATA'); \
    nltk.download('stopwords', download_dir='$NLTK_DATA')"

COPY src/ ./src/
COPY flask_api/ ./flask_api/
COPY models/ ./models/
COPY params.yaml ./

# Run as an unprivileged user.
RUN useradd --create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Two workers with a request timeout that accommodates word-cloud rendering,
# which is the slowest endpoint. Model load happens once per worker at boot.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", \
     "--timeout", "120", "--access-logfile", "-", "flask_api.main:app"]
