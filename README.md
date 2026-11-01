# Casting Defect Detection

[![tests](https://github.com/RutvikMendpara/Visual-defect-detection-for-manufacturing/actions/workflows/tests.yml/badge.svg)](https://github.com/RutvikMendpara/Visual-defect-detection-for-manufacturing/actions/workflows/tests.yml)

Classifies cast submersible pump impellers as OK or defective from a top-view photo, fast enough to run on a normal CPU. Fine-tuned ResNet-18, trained on the [Kaggle casting dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product).

Done so far: leakage check, clean splits, training, evaluation, INT8 ONNX export, FastAPI service in Docker, Gradio demo, drift monitoring, tests and CI. Still to come: anomaly detection (PatchCore).

## The problem

**Who it's for.** QA inspectors at a casting foundry who check impellers by eye as they come off the line. It can also help quality managers who want consistent inspection across shifts.

**Why it matters.** When a defective part slips through, it reaches the customer. That leads to returns, warranty claims, rework and, in the worst case, legal trouble and a damaged reputation. Manual inspection has its own costs. Inspectors get tired over a long shift, and two people can judge the same part differently. A part that passes in the morning might fail in the evening.

**What it does.** Takes a photo of each part and labels it OK or defective, with a target of under 50 ms per image on a normal CPU. It helps inspectors, it doesn't replace them.

## Results

Positive class = defective. Threshold chosen on the validation split and applied unchanged to the test split (258 images, 156 defective, 102 OK).

| Metric | Target | Result |
| --- | --- | --- |
| Recall (defective) | ≥ 99% at the chosen threshold | 100% (156/156) |
| False alarm rate | ≤ 5% at the same threshold | 0% (0/102) |
| Confusion matrix | report always | missed 0 of 156, flagged 0 of 102 |
| F1 | report only | 1.000 |
| ROC-AUC | ≥ 0.99 | 1.000 |
| CPU latency, deployed INT8 model | median < 50 ms | 4.3 ms median, 4.8 ms p95 (10.3 ms on 1 thread) |
| Through the API (Docker) | | 7.9 ms median, 8.6 ms p95 |

Latency: Apple M4 CPU, batch 1, file read + preprocessing included, 10 warm-up images excluded, 500 timed. The deployed model is INT8 ONNX (11 MB). It has its own threshold chosen on val, because quantization shifts the scores. PyTorch FP32 vs ONNX FP32 vs INT8 in [reports/deployment.md](reports/deployment.md).

These numbers look too good, so a few caveats:

- The test set is small. 0 misses out of 156 means the real miss rate could still be up to 2.4% (95% upper bound). 0 false alarms out of 102 could be up to 3.6%.
- All photos come from one setup with fixed lighting. A lighting stress test shows the model starts missing defects when brightness or contrast moves more than about 20-40% from normal. Details in [reports/evaluation.md](reports/evaluation.md).

## The leakage problem

The official 300×300 train/test split can't be trusted. The images were augmented before splitting, so the same part shows up rotated or flipped on both sides. A rotation-aware perceptual hash check found that **71.7% of the official test images have a near-copy in train** (at least 37% even with the strictest cut-off). High scores on that split mostly measure memorisation.

So I built new splits from the 1300 un-augmented 512×512 originals: 60/20/20, stratified, near-duplicate photos kept in the same split, saved in `data/splits.csv`. All results above are on that split. Full write-up in [DATA.md](DATA.md).

## How to run

Needs [uv](https://docs.astral.sh/uv/). Download the dataset from Kaggle into `data/raw/` so you have `data/raw/casting_data/...` and `data/raw/casting_512x512/...`.

```bash
uv sync

# data
uv run python -m defect_detection.leakage        # leakage report -> reports/
uv run python -m defect_detection.data split     # rebuild data/splits.csv (already in the repo)
uv run python -m defect_detection.data check     # load a batch, save augmented examples

# train (logs to MLflow, saves models/<run>/best.pt)
uv run python -m defect_detection.train --model resnet18 --lr 1e-3 --epochs 20

# evaluate
uv run python -m defect_detection.evaluate threshold --run models/resnet18-lr0.001
uv run python -m defect_detection.evaluate test      --run models/resnet18-lr0.001
uv run python -m defect_detection.evaluate latency   --run models/resnet18-lr0.001
uv run python -m defect_detection.evaluate gradcam   --run models/resnet18-lr0.001

# see runs
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# export, quantize, benchmark, pick the deployed model
uv run python -m defect_detection.export onnx     --run models/resnet18-lr0.001
uv run python -m defect_detection.export quantize --run models/resnet18-lr0.001
uv run python -m defect_detection.export bench    --run models/resnet18-lr0.001
uv run python -m defect_detection.export deploy   --run models/resnet18-lr0.001 --file model.int8.onnx
uv run python -m defect_detection.monitor reference

# tests
uv run pytest
```

## Running the service

Needs `models/deploy/` from the export step above.

```bash
docker build -t casting-inspection .
docker run --rm -p 8000:8000 casting-inspection
curl -F "file=@part.jpeg;type=image/jpeg" localhost:8000/predict
```

API docs at http://localhost:8000/docs. `POST /predict` returns the label, defect probability, threshold and latency, `POST /predict/heatmap` adds the heatmap, `GET /health` checks the model is loaded. Bad files get a clear 4xx error.

Demo: `uv run python app/demo.py`, then http://localhost:7860. Upload a photo or use the samples, run the conveyor simulation, flag wrong predictions. Check the prediction log for drift with `uv run python -m defect_detection.monitor check`.

Everything uses seed 42, so training gives the same numbers again (tested on Apple MPS).

## Repo layout

```
src/defect_detection/
  leakage.py     near-duplicate check between official train and test
  data.py        splits, Dataset, transforms
  train.py       fine-tuning loop
  evaluate.py    threshold, test metrics, latency, Grad-CAM
  tracking.py    MLflow setup
  export.py      ONNX export, INT8 quantization, benchmark
  inference.py   ONNX inference without torch, used by the API and demo
  monitor.py     drift checks on the prediction logs
app/
  main.py        FastAPI service
  demo.py        Gradio demo
  examples/      sample test images for the demo
data/splits.csv  the train/val/test split used everywhere
reports/         leakage, experiments, evaluation, deployment, figures
scripts/         Hugging Face Space build
tests/           pytest, runs in CI on every push
notebooks/       EDA only
```

## Docs

- [DATA.md](DATA.md): dataset, leakage check, splits, augmentation choices
- [reports/experiments.md](reports/experiments.md): model choice, training runs and what each one showed
- [reports/evaluation.md](reports/evaluation.md): threshold, test results, error analysis, Grad-CAM
- [reports/deployment.md](reports/deployment.md): ONNX and INT8 benchmark, API, Docker, monitoring, demo, tests

## Limitations

- Trained on one camera, one lighting setup. A new setup needs fresh images and recalibration before use.
- The validation set is saturated (every run separates it perfectly), so it can't tell similar models apart anymore.
- The dataset is about 60% defective. A real line has far fewer defects, which would lower precision.
- Monitoring only checks input stats and confidence. Without labels from the line it can't measure accuracy directly, that needs the flagged images to be reviewed.
- Licence of the dataset is CC BY-NC-ND 4.0, so this is for learning and portfolio use only.

MVTec AD targets will be set separately per category later.
