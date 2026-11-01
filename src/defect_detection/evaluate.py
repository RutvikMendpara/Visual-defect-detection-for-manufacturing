"""Predictions and metrics. Positive class = defective."""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


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
    tp = (pred & (labels == 1)).sum()
    fp = (pred & (labels == 0)).sum()
    fn = (~pred & (labels == 1)).sum()
    tn = (~pred & (labels == 0)).sum()
    recall_far5, _ = recall_at_false_alarm(labels, probs, 0.05)
    return {
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "false_alarm_rate": fp / max(fp + tn, 1),
        "pr_auc": average_precision_score(labels, probs),
        "roc_auc": roc_auc_score(labels, probs),
        "recall_at_far5": recall_far5,
    }
