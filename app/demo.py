"""Gradio demo: inspect one image, or run a folder through like a conveyor.

Run locally: uv run python app/demo.py
"""

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from PIL import Image

from defect_detection.inference import Classifier, image_stats

HERE = Path(__file__).parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models/deploy"))
EXAMPLES = Path(os.environ.get("EXAMPLES_DIR", HERE / "examples"))
LOG_FILE = Path(os.environ.get("LOG_FILE", "logs/predictions.jsonl"))
FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", "feedback"))

classifier = Classifier(MODEL_DIR)
example_files = sorted(EXAMPLES.glob("*.jpeg"))


def log(record, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def run(img, source):
    result = classifier.predict(img, heatmap=True)
    log({"ts": datetime.now(timezone.utc).isoformat(), "source": source, "label": result["label"],
         "defect_probability": round(result["defect_probability"], 6),
         "model_ms": round(result["latency_ms"], 2), **image_stats(img)}, LOG_FILE)
    return result


def verdict_text(result):
    p = result["defect_probability"]
    sure = "" if abs(p - result["threshold"]) > 0.3 else " (close to the threshold, worth a second look)"
    return (f"## {'DEFECTIVE' if result['label'] == 'defective' else 'OK'}\n"
            f"Defect probability **{p:.3f}**, threshold {result['threshold']:.3f}{sure}  \n"
            f"{result['latency_ms']:.1f} ms on CPU, model {result['model']}")


def inspect(img):
    if img is None:
        return None, "Upload an image or pick an example.", None
    result = run(img, "upload")
    return result["heatmap"], verdict_text(result), result


def flag_wrong(img, result):
    """Save the image with the label the user says is right, for retraining later."""
    if img is None or not result:
        return "Inspect an image first."
    corrected = "ok" if result["label"] == "defective" else "defective"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    img.save(FEEDBACK_DIR / f"{stamp}.png")
    log({"ts": stamp, "image": f"{stamp}.png", "predicted": result["label"], "corrected_label": corrected,
         "defect_probability": result["defect_probability"], "model": result["model"]},
        FEEDBACK_DIR / "feedback.jsonl")
    return f"Saved as {corrected}. Thanks, it goes into the review queue for the next training round."


def conveyor(delay):
    """Stream the sample folder one part at a time with running counts."""
    counts = {"ok": 0, "defective": 0}
    start = time.perf_counter()
    model_ms = 0.0
    belt = random.Random(42).sample(example_files, len(example_files))  # mixed like a real line
    for i, path in enumerate(belt, 1):
        img = Image.open(path)
        result = run(img, "conveyor")
        counts[result["label"]] += 1
        model_ms += result["latency_ms"]
        elapsed = time.perf_counter() - start
        stats = (f"**Part {i}/{len(example_files)}**: {path.name} → **{result['label'].upper()}** "
                 f"({result['defect_probability']:.3f})\n\n"
                 f"OK: **{counts['ok']}** · Defective: **{counts['defective']}**\n\n"
                 f"Line speed here: {60 * i / elapsed:.0f} parts/min (set by the delay). "
                 f"Model alone could do about {60_000 * i / model_ms:,.0f} parts/min.")
        yield result["heatmap"], stats
        time.sleep(delay)


with gr.Blocks(title="Visual Defect Detection for Manufacturing") as demo:
    gr.Markdown(
        "# Visual Defect Detection for Manufacturing\n"
        "Classifies cast pump impellers as OK or defective from a top-view photo. "
        "The red overlay shows where the model looked. ResNet-18, INT8 ONNX, runs on CPU.\n\n"
        "This assists an inspector, it doesn't replace one. Trained on one camera and lighting setup.\n\n"
        "Sample images: [Real-life Industrial Dataset of Casting Product]"
        "(https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product), "
        "CC BY-NC-ND 4.0. None of them were used for training."
    )
    last_result = gr.State()
    with gr.Tab("Inspect"):
        with gr.Row():
            image_in = gr.Image(type="pil", label="Part photo", height=360)
            heatmap_out = gr.Image(label="Where the model looked", height=360)
        verdict = gr.Markdown()
        with gr.Row():
            inspect_btn = gr.Button("Inspect", variant="primary")
            flag_btn = gr.Button("Flag as wrong")
        flag_msg = gr.Markdown()
        gr.Examples([[str(p)] for p in example_files[:8]], inputs=image_in)
        inspect_btn.click(inspect, image_in, [heatmap_out, verdict, last_result])
        image_in.upload(inspect, image_in, [heatmap_out, verdict, last_result])
        flag_btn.click(flag_wrong, [image_in, last_result], flag_msg)
    with gr.Tab("Conveyor"):
        gr.Markdown(f"Runs the {len(example_files)} sample parts through one by one, like a camera on a line.")
        delay = gr.Slider(0.0, 2.0, value=0.5, step=0.1, label="Seconds between parts")
        start_btn = gr.Button("Start", variant="primary")
        with gr.Row():
            belt_img = gr.Image(label="Current part", height=360)
            belt_stats = gr.Markdown()
        start_btn.click(conveyor, delay, [belt_img, belt_stats])


if __name__ == "__main__":
    demo.launch()
