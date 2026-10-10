# syntax=docker/dockerfile:1
# Heart Disease Risk API — production image
# Build:  docker build -t heartrisk-api:latest .
# Run:    docker run -p 8000:8000 heartrisk-api:latest

# ---- Stage 1: build wheels (keeps compilers/pip caches out of the final image) ----
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements-serve.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-serve.txt
COPY pyproject.toml .
COPY src ./src
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

# ---- Stage 2: runtime ----
FROM python:3.12-slim
LABEL org.opencontainers.image.title="heartrisk-api" \
      org.opencontainers.image.description="Heart disease risk prediction API (FastAPI + scikit-learn)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.joblib \
    METADATA_PATH=/app/models/metadata.json \
    LOG_LEVEL=INFO

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

# Bind-mount the wheels instead of COPYing them, so they never become an image layer.
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip install --no-cache-dir /wheels/*

# The model is baked into the image so a given tag always serves the same model.
COPY models/model.joblib models/metadata.json ./models/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "heartrisk.api:app", "--host", "0.0.0.0", "--port", "8000"]
