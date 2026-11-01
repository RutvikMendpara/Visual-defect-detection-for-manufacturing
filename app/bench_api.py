"""End-to-end latency through the API, to compare with the raw model numbers.

Start the API first (docker run or uvicorn), then:
  uv run python app/bench_api.py --url http://localhost:8000
"""

import argparse
import time
from pathlib import Path

import httpx
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--images", type=Path, default=Path("data/raw/casting_512x512/casting_512x512"))
    parser.add_argument("--runs", type=int, default=300)
    args = parser.parse_args()

    files = sorted(args.images.glob("*/*.jpeg"))[:50]
    payloads = [(f.name, f.read_bytes()) for f in files]

    with httpx.Client(base_url=args.url, timeout=10) as client:
        def call(i):
            name, data = payloads[i % len(payloads)]
            start = time.perf_counter()
            r = client.post("/predict", files={"file": (name, data, "image/jpeg")})
            r.raise_for_status()
            return (time.perf_counter() - start) * 1000, r.json()["latency_ms"]

        for i in range(10):
            call(i)
        results = np.array([call(i) for i in range(args.runs)])

    total, model = results[:, 0], results[:, 1]
    print(f"{args.runs} sequential requests to {args.url}, 10 warm-up")
    print(f"end to end : median {np.median(total):.1f} ms, p95 {np.percentile(total, 95):.1f} ms")
    print(f"model part : median {np.median(model):.1f} ms, p95 {np.percentile(model, 95):.1f} ms (preprocess + ONNX)")
    print(f"overhead   : median {np.median(total - model):.1f} ms (HTTP, upload, multipart parsing, JPEG decode, JSON, logging)")


if __name__ == "__main__":
    main()
