# Problem

## Who this is for

QA inspectors at a casting foundry who check submersible pump impellers by eye as they come off the line. It can also help quality managers who want consistent inspection across shifts.

## Why it matters

When a defective part slips through, it reaches the customer. That leads to returns, warranty claims, rework and, in the worst case, legal trouble and a damaged reputation.

Manual inspection has its own costs. Inspectors get tired over a long shift, and two people can judge the same part differently. A part that passes in the morning might fail in the evening.

## What the system does

It takes a photo of each part and labels it OK or defective, with a target of under 50 ms per image on a normal CPU.

## Limitations

It helps inspectors, it does not replace them. It was trained on images taken with fixed lighting and a fixed camera angle, so a new setup would need fresh images and recalibration before use.

## Metrics

Positive class = defective. The decision threshold is chosen on the validation split, so the test set stays untouched until the final evaluation.

| Metric             | Definition                            | Target                        | Result |
| ------------------ | ------------------------------------- | ----------------------------- | ------ |
| Recall (defective) | TP / all truly defective parts        | ≥ 99% at the chosen threshold |        |
| False alarm rate   | FP / all truly OK parts               | ≤ 5% at the same threshold    |        |
| Confusion matrix   | Raw counts, e.g. "missed 3 of 453"    | Report always                 |        |
| F1                 | Balance of precision and recall       | Report only, no target        |        |
| ROC-AUC            | Ranking quality across all thresholds | ≥ 0.99 (casting data)         |        |
| CPU latency        | Median and p95 per image              | Median < 50 ms                |        |

**Latency setup:** batch size 1, preprocessing included (load, resize, normalize), 10 warm-up images excluded, 500 timed images, CPU model written next to the result.

**MVTec AD:** targets will be set separately per category later.
