import numpy as np
import torch
from PIL import Image

from defect_detection.inference import Classifier, preprocess
from defect_detection.monitor import psi


def test_onnx_matches_pytorch(model_dir):
    path, torch_model = model_dir
    clf = Classifier(path)
    x = np.random.default_rng(0).normal(size=(4, 3, 224, 224)).astype(np.float32)
    onnx_logit, onnx_cam = clf.run(x)
    with torch.no_grad():
        torch_logit, torch_cam = torch_model(torch.from_numpy(x))
    np.testing.assert_allclose(onnx_logit, torch_logit[:, 0].numpy(), atol=1e-3)
    np.testing.assert_allclose(onnx_cam, torch_cam.numpy(), atol=1e-2)


def test_predict_output(model_dir):
    clf = Classifier(model_dir[0])
    img = Image.new("RGB", (300, 300), (128, 128, 128))
    result = clf.predict(img, heatmap=True)
    assert set(result) == {"label", "defect_probability", "threshold", "model", "latency_ms", "heatmap"}
    assert 0 <= result["defect_probability"] <= 1
    assert result["label"] == ("defective" if result["defect_probability"] > result["threshold"] else "ok")
    assert result["heatmap"].size == img.size


def test_any_image_size_and_mode_works(model_dir):
    clf = Classifier(model_dir[0])
    for img in [Image.new("L", (512, 512)), Image.new("RGBA", (100, 300)), Image.new("RGB", (1024, 768))]:
        assert preprocess(img).shape == (1, 3, 224, 224)
        assert clf.predict(img)["label"] in {"ok", "defective"}


def test_psi_flags_shift_but_not_noise():
    rng = np.random.default_rng(0)
    ref = rng.normal(140, 10, 2000)
    assert psi(ref, rng.normal(140, 10, 500)) < 0.1
    assert psi(ref, rng.normal(110, 10, 500)) > 0.25
