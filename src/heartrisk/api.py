"""FastAPI service: heart-disease risk prediction.

Endpoints
    GET  /health          liveness + model status (used by Docker / Kubernetes probes)
    GET  /model           metadata of the loaded model (version, metrics, params)
    POST /predict         one patient  -> prediction, probability, confidence, risk band
    POST /predict/batch   up to 100 patients
    GET  /metrics         Prometheus exposition format
    GET  /docs            interactive OpenAPI UI

Configuration (environment variables)
    MODEL_PATH       default models/model.joblib
    METADATA_PATH    default models/metadata.json
    LOG_LEVEL        default INFO
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from heartrisk import __version__, config
from heartrisk.monitoring import (
    MODEL_INFO,
    RequestLoggingMiddleware,
    configure_logging,
    log_event,
    record_prediction,
)
from heartrisk.schemas import BatchRequest, BatchResponse, HealthResponse, Patient, Prediction, PredictResponse

log = logging.getLogger("heartrisk.api")

# Risk bands on the predicted probability — coarse buckets that are easier to act on
# than a raw number. The decision threshold itself comes from the model metadata.
RISK_BANDS = ((0.3, "low"), (0.6, "moderate"), (1.01, "high"))


class ModelService:
    """Holds the loaded pipeline and its metadata."""

    def __init__(self) -> None:
        self.pipeline = None
        self.metadata: dict = {}
        self.started = time.time()

    def load(self, model_path: Path, metadata_path: Path) -> None:
        self.pipeline = joblib.load(model_path)
        self.metadata = json.loads(metadata_path.read_text())
        MODEL_INFO.labels(self.name, self.version).set(1)
        log_event(log, "model loaded", model_name=self.name, model_version=self.version, path=str(model_path))

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None

    @property
    def name(self) -> str:
        return self.metadata.get("model_name", "unknown")

    @property
    def version(self) -> str:
        return self.metadata.get("model_version", "unknown")

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("decision_threshold", 0.5))

    def predict(self, patients: list[Patient]) -> list[Prediction]:
        if not self.loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        frame = pd.DataFrame([p.model_dump() for p in patients])[config.FEATURES]
        # Optional fields arrive as None; the pipeline's imputers expect NaN.
        frame = frame.astype(float)
        probas = self.pipeline.predict_proba(frame)[:, 1]
        results = []
        for p in probas:
            p = float(p)
            pred = int(p >= self.threshold)
            label = "disease" if pred else "no disease"
            record_prediction(label, p)
            results.append(Prediction(
                prediction=pred, label=label, probability_disease=round(p, 4),
                confidence=round(max(p, 1 - p), 4),
                risk_band=next(band for limit, band in RISK_BANDS if p < limit),
            ))
        return results


service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    model_path = Path(os.environ.get("MODEL_PATH", config.MODELS_DIR / "model.joblib"))
    metadata_path = Path(os.environ.get("METADATA_PATH", config.MODELS_DIR / "metadata.json"))
    try:
        service.load(model_path, metadata_path)
    except FileNotFoundError:
        # Start anyway so /health can report the problem; /predict returns 503.
        log.exception("Model files not found: %s / %s", model_path, metadata_path)
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    version=__version__,
    description="Predicts presence of heart disease from 13 clinical features (UCI Cleveland).",
    lifespan=lifespan,
)
app.add_middleware(RequestLoggingMiddleware)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if service.loaded else "degraded",
        model_loaded=service.loaded,
        model_version=service.version if service.loaded else None,
        uptime_seconds=round(time.time() - service.started, 1),
    )


@app.get("/model")
def model_info():
    if not service.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return service.metadata


@app.post("/predict", response_model=PredictResponse)
def predict(patient: Patient, request: Request):
    result = service.predict([patient])[0]
    log_event(log, "prediction", request_id=request.state.request_id, **result.model_dump())
    return PredictResponse(**result.model_dump(), model_name=service.name, model_version=service.version,
                           request_id=request.state.request_id)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request):
    results = service.predict(batch.patients)
    log_event(log, "batch prediction", request_id=request.state.request_id, n=len(results),
              positives=sum(r.prediction for r in results))
    return BatchResponse(predictions=results, model_name=service.name, model_version=service.version,
                         request_id=request.state.request_id)


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
