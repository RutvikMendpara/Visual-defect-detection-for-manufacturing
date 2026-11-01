from collections import defaultdict

import numpy as np
import pytest
import torch
from PIL import Image

from defect_detection.data import RandomRotateShift, eval_transform, load_splits, train_transform
from defect_detection.inference import preprocess


@pytest.fixture
def part_image():
    # fake impeller: bright ring on a grey background, 512x512 like the originals
    y, x = np.mgrid[:512, :512]
    r = np.hypot(x - 256, y - 256)
    a = np.where((r > 80) & (r < 220), 200, 140).astype(np.uint8)
    return Image.fromarray(a).convert("RGB")


def test_numpy_preprocess_matches_torch_transform(part_image):
    # the API uses the numpy version, training used torchvision. They must agree
    # or the deployed model sees different inputs from the ones it was trained on.
    expected = eval_transform()(part_image.convert("L")).numpy()
    got = preprocess(part_image)
    assert got.shape == (1, 3, 224, 224)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got[0], expected, atol=1e-5)


def test_grayscale_becomes_three_identical_channels(part_image):
    x = preprocess(part_image)[0]
    # undo the per-channel normalisation and the three channels are the same picture
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    raw = x * std + mean
    np.testing.assert_allclose(raw[0], raw[1], atol=1e-5)
    np.testing.assert_allclose(raw[1], raw[2], atol=1e-5)


def test_train_transform_shape_and_randomness(part_image):
    t = train_transform()
    torch.manual_seed(0)
    a, b = t(part_image.convert("L")), t(part_image.convert("L"))
    assert a.shape == (3, 224, 224) and a.dtype == torch.float32
    assert not torch.equal(a, b), "augmentation should differ between calls"


def test_rotation_keeps_size_and_has_no_black_corners(part_image):
    img = part_image.convert("L").resize((224, 224))
    torch.manual_seed(1)
    out = RandomRotateShift()(img)
    assert out.size == img.size
    corners = np.asarray(out)[[0, 0, -1, -1], [0, -1, 0, -1]]
    assert corners.min() > 100, "corners should be filled with background, not black"


def test_split_file_has_no_part_in_two_splits():
    rows = load_splits("data/splits.csv")
    splits_per_group = defaultdict(set)
    for r in rows:
        splits_per_group[r["group"]].add(r["split"])
    assert all(len(s) == 1 for s in splits_per_group.values())
    assert len({r["path"] for r in rows}) == len(rows)


def test_split_file_is_stratified():
    rows = load_splits("data/splits.csv")
    rate = {s: np.mean([r["label"] for r in rows if r["split"] == s]) for s in ["train", "val", "test"]}
    assert max(rate.values()) - min(rate.values()) < 0.02
