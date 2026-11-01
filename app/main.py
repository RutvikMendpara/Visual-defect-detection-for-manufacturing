"""Inspection API.

Run locally: uv run uvicorn app.main:app --port 8000
Docs at http://localhost:8000/docs
"""

import base64
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from defect_detection.inference import Classifier, image_stats

MAX_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp"}
Image.MAX_IMAGE_PIXELS = 40_000_000  # refuse decompression bombs

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("inspection")


class Prediction(BaseModel):
    label: str
    defect_probability: float
    threshold: float
    model: str
    latency_ms: float


class PredictionWithHeatmap(Prediction):
    heatmap_png: str  # base64


class Health(BaseModel):
    status: str
    model: str
    threshold: float


@asynccontextmanager
async def lifespan(app):
    # load once, not per request: creating the ONNX session takes far longer than a prediction
    app.state.classifier = Classifier(os.environ.get("MODEL_DIR", "models/deploy"))
    log.info(json.dumps({"event": "model_loaded", "model": app.state.classifier.meta["name"]}))
    yield


app = FastAPI(title="Casting defect inspection", version="1.0", lifespan=lifespan)


def log_prediction(endpoint, filename, result, total_ms, img):
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "file": filename,
        "label": result["label"],
        "defect_probability": round(result["defect_probability"], 6),
        "model_ms": round(result["latency_ms"], 2),
        "total_ms": round(total_ms, 2),
        # input stats so changes in lighting or focus show up in the logs
        **image_stats(img),
    }
    log.info(json.dumps(record))
    # optional file with one JSON line per prediction, read by defect_detection.monitor
    if log_file := os.environ.get("LOG_FILE"):
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")


async def read_image(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"unsupported file type {file.content_type!r}, send a JPEG, PNG or BMP image")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file too large, limit is {MAX_BYTES // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(400, "empty file")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except UnidentifiedImageError:
        raise HTTPException(400, "not a valid image file") from None
    except Image.DecompressionBombError:
        raise HTTPException(400, "image dimensions too large") from None
    except OSError:
        raise HTTPException(400, "image file is corrupt or truncated") from None
    return img


def classifier(request: Request) -> Classifier:
    clf = getattr(request.app.state, "classifier", None)
    if clf is None:
        raise HTTPException(503, "model not loaded")
    return clf


@app.get("/health", response_model=Health)
def health(request: Request):
    clf = classifier(request)
    return {"status": "ok", "model": clf.meta["name"], "threshold": clf.threshold}


@app.post("/predict", response_model=Prediction)
async def predict(request: Request, file: UploadFile = File(...)):
    start = time.perf_counter()
    img = await read_image(file)
    result = classifier(request).predict(img)
    log_prediction("/predict", file.filename, result, (time.perf_counter() - start) * 1000, img)
    return result


@app.post("/predict/heatmap", response_model=PredictionWithHeatmap)
async def predict_heatmap(request: Request, file: UploadFile = File(...)):
    start = time.perf_counter()
    img = await read_image(file)
    result = classifier(request).predict(img, heatmap=True)
    buf = io.BytesIO()
    result.pop("heatmap").save(buf, format="PNG")
    result["heatmap_png"] = base64.b64encode(buf.getvalue()).decode()
    log_prediction("/predict/heatmap", file.filename, result, (time.perf_counter() - start) * 1000, img)
    return result
