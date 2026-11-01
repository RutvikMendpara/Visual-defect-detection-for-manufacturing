"""Watch the prediction logs for drift.

Every prediction from the API (LOG_FILE) and the demo (logs/predictions.jsonl) logs the
defect probability plus brightness, contrast and sharpness of the input. This compares a
recent window of those against a reference taken from the validation set.

  uv run python -m defect_detection.monitor reference              # once per deployed model
  uv run python -m defect_detection.monitor check --log logs/predictions.jsonl
  uv run python -m defect_detection.monitor simulate               # darkened / blurred test images
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from defect_detection.data import DATA_DIR, SPLIT_FILE, load_splits
from defect_detection.inference import Classifier, image_stats

FEATURES = ["brightness", "contrast", "sharpness", "defect_probability"]
PSI_WARN = 0.1
PSI_DRIFT = 0.25  # the usual rule of thumb: < 0.1 stable, 0.1-0.25 watch, > 0.25 drifted
MODEL_DIR = Path("models/deploy")


def psi(reference, current, bins=10):
    """Population stability index: how far the current distribution moved from the reference."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def uncertain_share(probs, low=0.05, high=0.95):
    probs = np.asarray(probs)
    return float(((probs > low) & (probs < high)).mean())


def records_for(classifier, images, source):
    out = []
    for img in images:
        result = classifier.predict(img)
        out.append({"source": source, "label": result["label"],
                    "defect_probability": result["defect_probability"], **image_stats(img)})
    return out


def split_images(split):
    return [Image.open(DATA_DIR / r["path"]) for r in load_splits(SPLIT_FILE) if r["split"] == split]


def cmd_reference(model_dir=MODEL_DIR):
    records = records_for(Classifier(model_dir), split_images("val"), "val")
    reference = {f: [r[f] for r in records] for f in FEATURES}
    (Path(model_dir) / "reference.json").write_text(json.dumps(reference))
    print(f"wrote {model_dir}/reference.json from {len(records)} val images")


def compare(reference, records):
    rows = {}
    for f in FEATURES:
        current = [r[f] for r in records]
        rows[f] = {"psi": psi(reference[f], current),
                   "ref_mean": float(np.mean(reference[f])), "cur_mean": float(np.mean(current))}
    rows["uncertain_share"] = {"ref": uncertain_share(reference["defect_probability"]),
                               "cur": uncertain_share([r["defect_probability"] for r in records])}
    rows["defect_rate"] = {"cur": float(np.mean([r["label"] == "defective" for r in records]))}
    return rows


def status(value):
    return "DRIFT" if value > PSI_DRIFT else "watch" if value > PSI_WARN else "ok"


def format_report(name, rows):
    lines = [f"[{name}]"]
    for f in FEATURES:
        r = rows[f]
        lines.append(f"  {f:<19} psi {r['psi']:6.3f}  {status(r['psi']):<5}  "
                     f"mean {r['ref_mean']:8.2f} -> {r['cur_mean']:8.2f}")
    u = rows["uncertain_share"]
    lines.append(f"  {'uncertain (0.05-0.95)':<19} {u['ref']:.1%} -> {u['cur']:.1%}")
    lines.append(f"  {'flagged defective':<19} {rows['defect_rate']['cur']:.1%}")
    return "\n".join(lines)


def cmd_check(log_file, window=200, model_dir=MODEL_DIR):
    reference = json.loads((Path(model_dir) / "reference.json").read_text())
    lines = Path(log_file).read_text().splitlines()[-window:]
    records = [json.loads(line) for line in lines]
    rows = compare(reference, records)
    print(format_report(f"last {len(records)} predictions in {log_file}", rows))
    drifted = [f for f in FEATURES if rows[f]["psi"] > PSI_DRIFT]
    print(f"\nDRIFT in {', '.join(drifted)}. Check the camera and lighting before trusting the results."
          if drifted else "\nno drift")
    return drifted


SCENARIOS = {
    "normal": lambda im: im,
    "lamp dimmed (brightness x0.6)": lambda im: ImageEnhance.Brightness(im).enhance(0.6),
    "camera out of focus (blur r2)": lambda im: im.filter(ImageFilter.GaussianBlur(2)),
    "slow lamp drift (brightness x0.85)": lambda im: ImageEnhance.Brightness(im).enhance(0.85),
}


def cmd_simulate(model_dir=MODEL_DIR, out=Path("reports/monitoring.txt")):
    import matplotlib.pyplot as plt

    classifier = Classifier(model_dir)
    reference = json.loads((Path(model_dir) / "reference.json").read_text())
    rows = [r for r in load_splits(SPLIT_FILE) if r["split"] == "test"]
    images = [Image.open(DATA_DIR / r["path"]) for r in rows]
    labels = np.array([r["label"] for r in rows])

    reports, results = [], {}
    for name, change in SCENARIOS.items():
        records = records_for(classifier, [change(im) for im in images], name)
        results[name] = records
        pred = np.array([r["label"] == "defective" for r in records])
        missed, false_alarms = int((~pred & (labels == 1)).sum()), int((pred & (labels == 0)).sum())
        reports.append(format_report(name, compare(reference, records))
                       + f"\n  {'actual effect':<19} missed {missed}/{(labels == 1).sum()}, "
                         f"false alarms {false_alarms}/{(labels == 0).sum()}")
    text = "Reference: val set. Current: test set with a simulated change.\n\n" + "\n\n".join(reports)
    print(text)
    out.write_text(text + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for ax, f in zip(axes, ["brightness", "sharpness", "defect_probability"]):
        shown = list(SCENARIOS)[1:3]
        values = reference[f] + [r[f] for name in shown for r in results[name]]
        bins = np.linspace(min(values), max(values), 30) if f != "defect_probability" else np.linspace(0, 1, 21)
        ax.hist(reference[f], bins=bins, alpha=0.5, density=True, label="reference (val)")
        for name in shown:
            ax.hist([r[f] for r in results[name]], bins=bins, alpha=0.5, density=True, label=name)
        ax.set_title(f)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("reports/figures/drift.png", dpi=100)
    plt.close(fig)
    print("wrote reports/monitoring.txt and reports/figures/drift.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("reference")
    check = sub.add_parser("check")
    check.add_argument("--log", type=Path, default=Path("logs/predictions.jsonl"))
    check.add_argument("--window", type=int, default=200)
    sub.add_parser("simulate")
    args = parser.parse_args()
    if args.command == "reference":
        cmd_reference()
    elif args.command == "check":
        cmd_check(args.log, args.window)
    else:
        cmd_simulate()


if __name__ == "__main__":
    main()
