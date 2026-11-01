CPU: Apple M4 (arm64), onnxruntime 1.31.1, torch 2.15.0
batch 1, file read + preprocessing + model, 10 warm-up, 500 timed, test split for metrics
each version uses its own threshold chosen on val

| Version | Threshold | Recall | Precision | False alarms | Max prob diff vs PyTorch | 1 thread: median / p95 ms, img/s | 4 threads: median / p95 ms, img/s | File size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch FP32 | 0.154 | 1.000 (156/156) | 1.000 | 0/102 | 0.0e+00 | 15.9 / 16.6, 63 | 11.7 / 12.9, 85 | 44.8 MB |
| ONNX FP32 | 0.154 | 1.000 (156/156) | 1.000 | 0/102 | 2.0e-06 | 18.3 / 18.7, 55 | 9.5 / 10.1, 105 | 44.8 MB |
| ONNX INT8 | 0.431 | 1.000 (156/156) | 1.000 | 0/102 | 2.4e-01 | 10.3 / 10.5, 97 | 4.3 / 4.8, 228 | 11.3 MB |
