"""Export to ONNX, check it against PyTorch, quantize to INT8 and benchmark.

  uv run python -m defect_detection.export onnx     --run models/resnet18-lr0.001
  uv run python -m defect_detection.export quantize --run models/resnet18-lr0.001
  uv run python -m defect_detection.export bench    --run models/resnet18-lr0.001
  uv run python -m defect_detection.export deploy   --run models/resnet18-lr0.001 --file model.int8.onnx
"""

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torch import nn

from defect_detection.data import DATA_DIR, SEED, load_splits, SPLIT_FILE
from defect_detection.evaluate import choose_threshold, cpu_name, load_model, summarize
from defect_detection.inference import Classifier, preprocess


class LogitAndCam(nn.Module):
    """Model output plus a class activation map, so the heatmap works without gradients.

    The head is global average pooling + one linear layer, so the Grad-CAM weights are
    just the linear weights. Applying them at every location gives the same map.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        features = self.model.forward_features(x)
        logit = self.model.forward_head(features)
        cam = torch.einsum("c,nchw->nhw", self.model.get_classifier().weight[0], features)
        return logit, cam


def split_paths(split):
    return [r for r in load_splits(SPLIT_FILE) if r["split"] == split]


def write_meta(run_dir, file, name, logit_threshold):
    """Metadata next to each ONNX file: <file>.json. The API and demo read it."""
    run_dir = Path(run_dir)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
    meta = {"name": name, "file": file, "threshold": float(1 / (1 + np.exp(-logit_threshold))),
            "threshold_logit": logit_threshold, "image_size": ckpt["image_size"], "source_run": run_dir.name}
    (run_dir / f"{file}.json").write_text(json.dumps(meta, indent=2))
    return meta


def val_threshold(onnx_path):
    """Choose the threshold for this exact file on val, same rule as evaluate.choose_threshold."""
    rows = split_paths("val")
    x = np.concatenate([preprocess(Image.open(DATA_DIR / r["path"])) for r in rows])
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    logits = session.run(None, {"image": x})[0][:, 0]
    labels = np.array([r["label"] for r in rows])
    threshold, low, high = choose_threshold(labels, logits)
    print(f"{Path(onnx_path).name}: valid logit range on val {low:.2f} .. {high:.2f}, threshold {threshold:.2f}")
    return threshold


def cmd_onnx(run_dir):
    run_dir = Path(run_dir)
    model, ckpt = load_model(run_dir, "cpu")
    wrapped = LogitAndCam(model).eval()
    size = ckpt["image_size"]
    out = run_dir / "model.onnx"
    # example batch of 2: torch.export treats a size-1 dim as fixed even when marked dynamic
    torch.onnx.export(wrapped, (torch.randn(2, 3, size, size),), str(out),
                      input_names=["image"], output_names=["logit", "cam"],
                      dynamic_shapes={"x": {0: torch.export.Dim("batch", max=256)}}, dynamo=True,
                      external_data=False)
    # same model as best.pt (checked below), so it keeps the threshold from evaluate.py
    write_meta(run_dir, "model.onnx", f"{ckpt['model']}-fp32",
               json.loads((run_dir / "threshold.json").read_text())["logit"])
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    verify(run_dir, wrapped, out)


@torch.no_grad()
def verify(run_dir, wrapped, onnx_path, n=64):
    """Same images through PyTorch and ONNX Runtime, outputs should match closely."""
    rows = split_paths("test")[:n]
    x = np.concatenate([preprocess(Image.open(DATA_DIR / r["path"])) for r in rows])
    torch_logit, torch_cam = wrapped(torch.from_numpy(x))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logit, onnx_cam = session.run(None, {"image": x})
    logit_diff = np.abs(torch_logit.numpy() - onnx_logit).max()
    cam_diff = np.abs(torch_cam.numpy() - onnx_cam).max()
    threshold = json.loads((Path(run_dir) / "threshold.json").read_text())["logit"]
    same = ((torch_logit.numpy() > threshold) == (onnx_logit > threshold)).mean()
    print(f"verify on {n} test images: max logit diff {logit_diff:.2e}, max cam diff {cam_diff:.2e}, "
          f"same decision {same:.0%}")
    assert logit_diff < 1e-3, "ONNX output doesn't match PyTorch"


def cmd_quantize(run_dir, n_calib=200):
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    run_dir = Path(run_dir)
    fp32 = run_dir / "model.onnx"
    prepped = run_dir / "model.prep.onnx"
    out = run_dir / "model.int8.onnx"
    quant_pre_process(str(fp32), str(prepped))

    # calibration images from train only, val and test stay untouched
    train = split_paths("train")
    picks = np.random.default_rng(SEED).choice(len(train), size=n_calib, replace=False)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.images = iter(train[i]["path"] for i in picks)

        def get_next(self):
            path = next(self.images, None)
            return None if path is None else {"image": preprocess(Image.open(DATA_DIR / path))}

    quantize_static(str(prepped), str(out), Reader(), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    prepped.unlink()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), calibrated on {n_calib} train images")
    # INT8 shifts the scores a bit, so it gets its own threshold, again chosen on val
    name = json.loads((run_dir / "model.onnx.json").read_text())["name"].replace("fp32", "int8")
    write_meta(run_dir, "model.int8.onnx", name, val_threshold(out))


def time_it(fn, paths, warmup=10, runs=500):
    """Latency per image in ms, from reading the file to the probability."""
    for i in range(warmup):
        fn(paths[i % len(paths)])
    times = []
    for i in range(runs):
        start = time.perf_counter()
        fn(paths[i % len(paths)])
        times.append((time.perf_counter() - start) * 1000)
    return np.array(times)


def cmd_bench(run_dir, threads_list=(1, 4)):
    run_dir = Path(run_dir)
    rows = split_paths("test")
    paths = [DATA_DIR / r["path"] for r in rows]
    labels = np.array([r["label"] for r in rows])
    model, ckpt = load_model(run_dir, "cpu")
    size = ckpt["image_size"]

    def torch_version(threads):
        torch.set_num_threads(threads)

        @torch.no_grad()
        def prob(path):
            x = torch.from_numpy(preprocess(Image.open(path), size))
            return torch.sigmoid(model(x)).item()
        return prob

    def onnx_version(file):
        def make(threads):
            clf = Classifier.__new__(Classifier)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            clf.session = ort.InferenceSession(str(run_dir / file), opts, providers=["CPUExecutionProvider"])

            def prob(path):
                logit, _ = clf.run(preprocess(Image.open(path), size))
                return float(1 / (1 + np.exp(-logit[0])))
            return prob
        return make

    versions = [
        ("PyTorch FP32", torch_version, run_dir / "best.pt", run_dir / "model.onnx.json"),
        ("ONNX FP32", onnx_version("model.onnx"), run_dir / "model.onnx", run_dir / "model.onnx.json"),
        ("ONNX INT8", onnx_version("model.int8.onnx"), run_dir / "model.int8.onnx", run_dir / "model.int8.onnx.json"),
    ]
    lines = [
        f"CPU: {cpu_name()} ({platform.machine()}), onnxruntime {ort.__version__}, torch {torch.__version__}",
        "batch 1, file read + preprocessing + model, 10 warm-up, 500 timed, test split for metrics",
        "each version uses its own threshold chosen on val",
        "",
        "| Version | Threshold | Recall | Precision | False alarms | Max prob diff vs PyTorch | "
        + " | ".join(f"{t} thread{'s' if t > 1 else ''}: median / p95 ms, img/s" for t in threads_list)
        + " | File size |",
        "| --- " * (7 + len(threads_list)) + "|",
    ]
    reference = None
    for name, make, file, meta_file in versions:
        threshold = json.loads(meta_file.read_text())["threshold"]
        prob = make(max(threads_list))
        probs = np.array([prob(p) for p in paths])
        reference = probs if reference is None else reference
        m = summarize(labels, probs, threshold)
        cells = [name, f"{threshold:.3f}", f"{m['recall']:.3f} ({m['tp']}/{m['tp'] + m['fn']})",
                 f"{m['precision']:.3f}", f"{m['fp']}/{m['fp'] + m['tn']}",
                 f"{np.abs(probs - reference).max():.1e}"]
        for t in threads_list:
            times = time_it(make(t), paths)
            cells.append(f"{np.median(times):.1f} / {np.percentile(times, 95):.1f}, {1000 / times.mean():.0f}")
        cells.append(f"{file.stat().st_size / 1e6:.1f} MB")
        lines.append("| " + " | ".join(cells) + " |")
        print(lines[-1], flush=True)

    report = "\n".join(lines)
    Path("reports/benchmark.md").write_text(report + "\n")
    print("\n" + report)


def cmd_deploy(run_dir, file, out=Path("models/deploy")):
    """Copy the chosen ONNX file and its metadata to one folder the API and demo load from."""
    run_dir = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / file, out / "model.onnx")
    meta = json.loads((run_dir / f"{file}.json").read_text())
    meta["file"] = "model.onnx"
    (out / "model.json").write_text(json.dumps(meta, indent=2))
    print(f"copied {run_dir / file} -> {out / 'model.onnx'} ({meta['name']}, threshold {meta['threshold']:.4f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["onnx", "quantize", "bench", "deploy"]:
        p = sub.add_parser(name)
        p.add_argument("--run", type=Path, required=True)
        if name == "deploy":
            p.add_argument("--file", default="model.int8.onnx")
    args = parser.parse_args()
    if args.command == "onnx":
        cmd_onnx(args.run)
    elif args.command == "quantize":
        cmd_quantize(args.run)
    elif args.command == "bench":
        cmd_bench(args.run)
    else:
        cmd_deploy(args.run, args.file)


if __name__ == "__main__":
    main()
