"""Predictions, metrics, threshold choice, test evaluation, latency and Grad-CAM.

Positive class = defective.

  uv run python -m defect_detection.evaluate threshold --run models/resnet18-lr0.001
  uv run python -m defect_detection.evaluate test      --run models/resnet18-lr0.001
  uv run python -m defect_detection.evaluate latency   --run models/resnet18-lr0.001
  uv run python -m defect_detection.evaluate gradcam   --run models/resnet18-lr0.001
"""

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from defect_detection.data import DATA_DIR, SEED, CastingDataset, eval_transform, make_loader, unnormalize

RECALL_TARGET = 0.99
MAX_FALSE_ALARM = 0.05


@torch.no_grad()
def predict(model, loader, device):
    """Return logits, labels and paths for every image in the loader."""
    model.eval()
    logits, labels, paths = [], [], []
    for x, y, p in loader:
        logits.append(model(x.to(device)).squeeze(1).float().cpu())
        labels.append(y)
        paths += list(p)
    return torch.cat(logits), torch.cat(labels), paths


def recall_at_false_alarm(labels, probs, max_far=0.05):
    """Best recall on defective parts while flagging at most `max_far` of the OK parts.

    Returns the recall and the threshold that gets it (predict defective if prob > threshold).
    """
    ok = np.sort(probs[labels == 0])[::-1]
    allowed = int(max_far * len(ok))
    threshold = ok[allowed] if allowed < len(ok) else 0.0
    return (probs[labels == 1] > threshold).mean(), threshold


def summarize(labels, probs, threshold=0.5):
    labels, probs = np.asarray(labels), np.asarray(probs)
    pred = probs > threshold
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    tn = int((~pred & (labels == 0)).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall_far5, _ = recall_at_false_alarm(labels, probs, 0.05)
    return {
        "recall": recall,
        "precision": precision,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "false_alarm_rate": fp / max(fp + tn, 1),
        "pr_auc": average_precision_score(labels, probs),
        "roc_auc": roc_auc_score(labels, probs),
        "recall_at_far5": recall_far5,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def choose_threshold(labels, logits, recall_target=RECALL_TARGET, max_far=MAX_FALSE_ALARM):
    """Pick a logit threshold on validation data.

    Every threshold between `low` and `high` meets both targets on val. Taking the
    middle leaves the most room for test scores to land a bit differently.
    Done on logits because probabilities squash everything near 0 and 1.
    """
    ok = np.sort(logits[labels == 0])[::-1]
    bad = np.sort(logits[labels == 1])
    low = ok[int(max_far * len(ok))]  # above this, at most max_far of OK parts get flagged
    high = bad[int((1 - recall_target) * len(bad))]  # below this, recall stays >= target
    if low >= high:
        print("warning: no threshold meets both targets on val, keeping the false alarm limit")
        return float(low), float(low), float(high)
    return float((low + high) / 2), float(low), float(high)


def load_model(run_dir, device="cpu"):
    ckpt = torch.load(Path(run_dir) / "best.pt", map_location=device)
    model = timm.create_model(ckpt["model"], pretrained=False, num_classes=1)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def device_name():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def cmd_threshold(run_dir):
    device = device_name()
    model, ckpt = load_model(run_dir, device)
    logits, labels, _ = predict(model, make_loader("val", 64, image_size=ckpt["image_size"]), device)
    logits, labels = logits.numpy(), labels.numpy()
    threshold, low, high = choose_threshold(labels, logits)
    prob = float(1 / (1 + np.exp(-threshold)))
    val = summarize(labels, 1 / (1 + np.exp(-logits)), prob)
    result = {
        "chosen_on": "val",
        "logit": threshold,
        "prob": prob,
        "valid_range_logit": [low, high],
        "recall_target": RECALL_TARGET,
        "max_false_alarm": MAX_FALSE_ALARM,
        "val": val,
    }
    (Path(run_dir) / "threshold.json").write_text(json.dumps(result, indent=2))
    print(f"valid logit range on val: {low:.2f} .. {high:.2f}")
    print(f"threshold: logit {threshold:.2f} = prob {prob:.4f}")
    print(f"val at that threshold: recall {val['recall']:.3f}, false alarms {val['fp']}/{val['fp'] + val['tn']}, "
          f"missed {val['fn']}/{val['fn'] + val['tp']}")


def wilson_upper(errors, n, z=1.96):
    """Upper end of the 95% Wilson interval for an error rate. 0 errors out of n is not 0%."""
    p = errors / n
    centre = p + z * z / (2 * n)
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre + spread) / (1 + z * z / n)


def per_thousand(m, defect_rate):
    defective = 1000 * defect_rate
    good = 1000 - defective
    missed = defective * (1 - m["recall"])
    rechecked = good * m["false_alarm_rate"]
    return defective, missed, rechecked


def cmd_test(run_dir, line_defect_rate, again):
    run_dir = Path(run_dir)
    out = run_dir / "test_results.json"
    if out.exists() and not again:
        print(f"{out} already exists. The test set is meant to be used once, pass --again if you really mean it.")
        print(out.read_text())
        return
    thr = json.loads((run_dir / "threshold.json").read_text())

    device = device_name()
    model, ckpt = load_model(run_dir, device)
    logits, labels, paths = predict(model, make_loader("test", 64, image_size=ckpt["image_size"]), device)
    probs = torch.sigmoid(logits).numpy()
    labels = labels.numpy()
    m = summarize(labels, probs, thr["prob"])

    with open(run_dir / "test_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "logit", "prob", "pred", "correct"])
        for p, y, lg, pr in zip(paths, labels, logits.numpy(), probs):
            pred = int(pr > thr["prob"])
            writer.writerow([p, int(y), f"{lg:.4f}", f"{pr:.6f}", pred, int(pred == y)])

    m = {k: float(v) for k, v in m.items()}
    test_rate = labels.mean()
    lines = [
        f"threshold {thr['prob']:.4f} (chosen on val, logit {thr['logit']:.2f})",
        f"recall {m['recall']:.4f}  precision {m['precision']:.4f}  f1 {m['f1']:.4f}  "
        f"false alarm rate {m['false_alarm_rate']:.4f}",
        f"roc_auc {m['roc_auc']:.4f}  pr_auc {m['pr_auc']:.4f}",
        "",
        "                 predicted OK   predicted defective",
        f"actually OK      {m['tn']:>12.0f}   {m['fp']:>19.0f}",
        f"actually defect  {m['fn']:>12.0f}   {m['tp']:>19.0f}",
        "",
    ]
    n_bad, n_ok = int(m["tp"] + m["fn"]), int(m["tn"] + m["fp"])
    m["miss_rate_upper95"] = wilson_upper(m["fn"], n_bad)
    m["false_alarm_upper95"] = wilson_upper(m["fp"], n_ok)
    lines += [
        f"miss rate {m['fn']:.0f}/{n_bad}, could really be up to {m['miss_rate_upper95']:.1%} (95% upper bound)",
        f"false alarms {m['fp']:.0f}/{n_ok}, could really be up to {m['false_alarm_upper95']:.1%} (95% upper bound)",
        "",
    ]
    for name, rate in [("test set mix", test_rate), ("assumed line", line_defect_rate)]:
        defective, missed, rechecked = per_thousand(m, rate)
        lines.append(f"per 1000 parts ({name}, {rate:.0%} defective): {defective:.0f} defective, "
                     f"{missed:.1f} slip through, {rechecked:.1f} good parts re-checked")
    print("\n".join(lines))
    out.write_text(json.dumps({"threshold": thr["prob"], "metrics": m, "n": len(labels)}, indent=2))


def cpu_name():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return platform.processor()


@torch.no_grad()
def cmd_latency(run_dir, warmup=10, runs=500):
    model, ckpt = load_model(run_dir, "cpu")
    transform = eval_transform(ckpt["image_size"])
    paths = [DATA_DIR / r["path"] for r in CastingDataset("test", transform).rows]

    def one(path):
        # full path a request would take: read the file, preprocess, run the model
        x = transform(Image.open(path).convert("L")).unsqueeze(0)
        return torch.sigmoid(model(x)).item()

    for i in range(warmup):
        one(paths[i % len(paths)])
    times = []
    for i in range(runs):
        start = time.perf_counter()
        one(paths[i % len(paths)])
        times.append((time.perf_counter() - start) * 1000)
    times = np.array(times)
    print(f"{ckpt['model']} on {cpu_name()}, torch {torch.__version__}, {torch.get_num_threads()} threads")
    print(f"batch 1, preprocessing included, {warmup} warm-up, {runs} timed")
    print(f"median {np.median(times):.1f} ms, p95 {np.percentile(times, 95):.1f} ms, mean {times.mean():.1f} ms")


def grad_cam(model, x, layer):
    """Heatmap of where `layer`'s features pushed the prediction the way it went."""
    store = {}

    def keep(module, inputs, output):
        store["act"] = output
        output.register_hook(lambda grad: store.update(grad=grad))

    handle = layer.register_forward_hook(keep)
    logit = model(x).squeeze()
    handle.remove()
    # for an OK prediction, explain the OK decision instead of the (low) defective score
    (logit if logit > 0 else -logit).backward()

    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear")[0, 0]
    return (cam / (cam.max() + 1e-8)).detach().numpy(), torch.sigmoid(logit).item()


def cmd_gradcam(run_dir, extra_paths=(), per_class=4, out=Path("reports/figures/gradcam.png")):
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    model, ckpt = load_model(run_dir, "cpu")
    layer = model.layer4 if hasattr(model, "layer4") else list(model.children())[-3]
    thr = json.loads((run_dir / "threshold.json").read_text())["prob"]
    with open(run_dir / "test_predictions.csv") as f:
        preds = list(csv.DictReader(f))

    wrong = [r for r in preds if r["correct"] == "0"]
    rng = np.random.default_rng(SEED)
    picks = list(wrong)
    for label in ["1", "0"]:
        right = [r for r in preds if r["correct"] == "1" and r["label"] == label]
        picks += list(rng.choice(right, size=min(per_class, len(right)), replace=False))
    picks += [r for r in preds if r["path"] in extra_paths and r not in picks]

    transform = eval_transform(ckpt["image_size"])
    fig, axes = plt.subplots(len(picks), 2, figsize=(5, 2.6 * len(picks)))
    axes = np.atleast_2d(axes)
    for row, r in zip(axes, picks):
        x = transform(Image.open(DATA_DIR / r["path"]).convert("L")).unsqueeze(0)
        cam, prob = grad_cam(model, x, layer)
        img = unnormalize(x[0])
        truth = "defective" if r["label"] == "1" else "ok"
        verdict = "defective" if prob > thr else "ok"
        tag = "" if truth == verdict else "  WRONG"
        row[0].imshow(img, cmap="gray", vmin=0, vmax=1)
        row[0].set_title(f"{Path(r['path']).name}\ntrue {truth}", fontsize=7)
        row[1].imshow(img, cmap="gray", vmin=0, vmax=1)
        row[1].imshow(cam, cmap="jet", alpha=0.4)
        row[1].set_title(f"pred {verdict} ({prob:.3f}){tag}", fontsize=7)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out} ({len(wrong)} wrong, {len(picks) - len(wrong)} others)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["threshold", "test", "latency", "gradcam"]:
        p = sub.add_parser(name)
        p.add_argument("--run", type=Path, required=True, help="folder with best.pt")
        if name == "test":
            p.add_argument("--line-defect-rate", type=float, default=0.05,
                           help="defect rate to assume for the per-1000 numbers")
            p.add_argument("--again", action="store_true")
        if name == "gradcam":
            p.add_argument("--extra", nargs="*", default=[], help="extra test image paths to include")
            p.add_argument("--out", type=Path, default=Path("reports/figures/gradcam.png"))
    args = parser.parse_args()

    if args.command == "threshold":
        cmd_threshold(args.run)
    elif args.command == "test":
        cmd_test(args.run, args.line_defect_rate, args.again)
    elif args.command == "latency":
        cmd_latency(args.run)
    else:
        cmd_gradcam(args.run, args.extra, out=args.out)


if __name__ == "__main__":
    main()
