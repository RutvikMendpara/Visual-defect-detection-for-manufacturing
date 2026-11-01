import json

import pytest
import timm
import torch

from defect_detection.export import LogitAndCam


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """A small untrained ResNet-18 exported the same way as the real one.

    CI has no dataset and no trained weights, so tests check the plumbing, not accuracy.
    """
    out = tmp_path_factory.mktemp("model")
    torch.manual_seed(0)
    model = LogitAndCam(timm.create_model("resnet18", pretrained=False, num_classes=1)).eval()
    torch.onnx.export(model, (torch.randn(2, 3, 224, 224),), str(out / "model.onnx"),
                      input_names=["image"], output_names=["logit", "cam"],
                      dynamic_shapes={"x": {0: torch.export.Dim("batch", max=256)}}, dynamo=True,
                      external_data=False, verbose=False)
    meta = {"name": "test-model", "file": "model.onnx", "threshold": 0.5, "threshold_logit": 0.0,
            "image_size": 224, "source_run": "test"}
    (out / "model.json").write_text(json.dumps(meta))
    return out, model
