import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as api


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir[0]))
    with TestClient(api.app) as c:
        yield c


def jpeg_bytes(size=(300, 300)):
    buf = io.BytesIO()
    Image.new("RGB", size, (130, 130, 130)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": "test-model", "threshold": 0.5}


def test_predict_response_format(client):
    r = client.post("/predict", files={"file": ("part.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"label", "defect_probability", "threshold", "model", "latency_ms"}
    assert body["label"] in {"ok", "defective"}
    assert 0 <= body["defect_probability"] <= 1


def test_heatmap_is_a_png(client):
    r = client.post("/predict/heatmap", files={"file": ("part.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    img = Image.open(io.BytesIO(base64.b64decode(r.json()["heatmap_png"])))
    assert img.format == "PNG" and img.size == (300, 300)


@pytest.mark.parametrize("name, data, content_type, status", [
    ("notes.txt", b"hello", "text/plain", 415),
    ("fake.jpg", b"definitely not a jpeg", "image/jpeg", 400),
    ("empty.jpg", b"", "image/jpeg", 400),
    ("cut.jpg", jpeg_bytes()[:200], "image/jpeg", 400),
])
def test_bad_input_gets_clear_error(client, name, data, content_type, status):
    r = client.post("/predict", files={"file": (name, data, content_type)})
    assert r.status_code == status
    assert isinstance(r.json()["detail"], str) and r.json()["detail"]


def test_too_large_file(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_BYTES", 1000)
    r = client.post("/predict", files={"file": ("big.jpg", jpeg_bytes((600, 600)), "image/jpeg")})
    assert r.status_code == 413


def test_missing_file(client):
    assert client.post("/predict").status_code == 422
