"""Run the exported ONNX model.

Only needs numpy, Pillow and onnxruntime, not torch, so the API image and the
demo stay small.
"""

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def preprocess(img, size=224):
    """Same as data.eval_transform, without torch. Returns a (1, 3, size, size) float32 array."""
    gray = img.convert("L").resize((size, size), Image.BILINEAR)
    x = np.asarray(gray, dtype=np.float32) / 255.0
    x = (np.repeat(x[None], 3, axis=0) - MEAN) / STD
    return x[None].astype(np.float32)


def image_stats(img):
    """Cheap numbers about the input itself, logged with every prediction to spot drift."""
    a = np.asarray(img.convert("L").resize((128, 128)), dtype=np.float32)
    laplacian = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1])
    return {
        "brightness": round(float(a.mean()), 2),
        "contrast": round(float(a.std()), 2),
        "sharpness": round(float(laplacian.var()), 2),  # drops when the image is blurry
    }


def heatmap_overlay(img, cam, defective, size=None):
    """Red overlay where the model looked. `cam` is the raw 7x7 class activation map."""
    cam = cam if defective else -cam  # for OK, show what pushed it towards OK
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    size = size or img.size
    base = img.convert("L").resize(size).convert("RGB")
    mask = Image.fromarray((cam * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    red = Image.new("RGB", size, (255, 0, 0))
    alpha = mask.point(lambda v: int(v * 0.55))
    return Image.composite(red, base, alpha)


class Classifier:
    def __init__(self, model_dir, threads=None):
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "model.json").read_text())
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_dir / self.meta["file"]), opts,
                                            providers=["CPUExecutionProvider"])
        self.threshold = self.meta["threshold"]
        self.image_size = self.meta["image_size"]

    def run(self, x):
        logit, cam = self.session.run(None, {"image": x})
        return logit[:, 0], cam

    def predict(self, img, heatmap=False):
        start = time.perf_counter()
        logit, cam = self.run(preprocess(img, self.image_size))
        prob = float(1 / (1 + np.exp(-logit[0])))
        defective = prob > self.threshold
        result = {
            "label": "defective" if defective else "ok",
            "defect_probability": prob,
            "threshold": self.threshold,
            "model": self.meta["name"],
        }
        if heatmap:
            result["heatmap"] = heatmap_overlay(img, cam[0], defective)
        result["latency_ms"] = (time.perf_counter() - start) * 1000
        return result
