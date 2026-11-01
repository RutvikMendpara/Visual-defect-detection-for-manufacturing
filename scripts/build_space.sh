#!/usr/bin/env bash
# Put together a folder that can be uploaded as a Hugging Face Space.
#   scripts/build_space.sh
#   uvx hf auth login
#   uvx hf upload <your-username>/casting-defect-inspection build/space . --repo-type space
set -euo pipefail

out=build/space
rm -rf "$out"
mkdir -p "$out/defect_detection" "$out/models"

cp app/demo.py "$out/app.py"
cp -r app/examples "$out/examples"
cp -r models/deploy "$out/models/deploy"
cp src/defect_detection/__init__.py src/defect_detection/inference.py "$out/defect_detection/"

gradio_version=$(uv pip list 2>/dev/null | awk '/^gradio /{print $2}')
grep -E "^(onnxruntime|numpy|pillow)==" app/requirements.txt > "$out/requirements.txt"

cat > "$out/README.md" <<README
---
title: Casting Defect Inspection
sdk: gradio
sdk_version: $gradio_version
app_file: app.py
python_version: "3.13"
license: cc-by-nc-nd-4.0
short_description: Spots defects in cast pump impellers from a photo
---

Demo for https://github.com/RutvikMendpara/Visual-defect-detection-for-manufacturing

ResNet-18 fine-tuned on the Kaggle casting dataset, exported to INT8 ONNX, runs on the free CPU.
Sample images are from the test split and were not used for training.
README

du -sh "$out"
echo "built $out"
