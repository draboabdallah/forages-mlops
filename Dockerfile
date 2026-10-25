FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/model.joblib

WORKDIR /app

# Dépendances d'abord : cette couche reste en cache tant que requirements-api.txt ne change pas.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY forages/ forages/
COPY models/ models/

# Exécution sans privilèges root.
RUN useradd --create-home --uid 1000 api && chown -R api /app
USER api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "forages.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
