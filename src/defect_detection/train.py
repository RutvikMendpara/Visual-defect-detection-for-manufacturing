"""Fine-tune a pretrained timm model on the casting split.

Example: uv run python -m defect_detection.train --model resnet18 --epochs 15 --lr 3e-4
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import timm
import torch
import torch.nn.functional as F

from defect_detection import tracking
from defect_detection.data import IMAGE_SIZE, SEED, make_loader
from defect_detection.evaluate import predict, summarize


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds cuda and mps


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total, count = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.float().to(device)
        logits = model(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item() * len(y)
        count += len(y)
    return total / count


def plot_curves(history, out):
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax1.set_title("loss")
    ax1.set_xlabel("epoch")
    ax1.legend()
    for key in ["val_pr_auc", "val_recall", "val_recall_at_far5"]:
        ax2.plot(epochs, [h[key] for h in history], label=key)
    ax2.set_title("validation")
    ax2.set_xlabel("epoch")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="resnet18", help="any timm model name")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    return parser.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or f"{args.model}-lr{args.lr:g}" + ("" if args.augment else "-noaug")
    out = args.out_dir / run_name
    out.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = pick_device()
    train_loader = make_loader("train", args.batch_size, image_size=args.image_size, augment=args.augment)
    val_loader = make_loader("val", 64, image_size=args.image_size)

    # one output logit for "defective", the backbone keeps its ImageNet weights
    model = timm.create_model(args.model, pretrained=True, num_classes=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(run_name=run_name, device=device)
    (out / "config.json").write_text(json.dumps(config, indent=2))

    tracking.setup()
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config)
        tracking.log_split()

        best, history = None, []
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            logits, labels, _ = predict(model, val_loader, device)
            val_loss = F.binary_cross_entropy_with_logits(logits, labels.float()).item()
            val = summarize(labels.numpy(), torch.sigmoid(logits).numpy())
            scheduler.step()

            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   **{f"val_{k}": float(v) for k, v in val.items()}}
            history.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)

            # PR-AUC first, val loss breaks ties (PR-AUC often sits at the same value for a few epochs)
            score = (val["pr_auc"], -val_loss)
            improved = best is None or score > best[0]
            if improved:
                best = (score, row)
                torch.save({"model": args.model, "image_size": args.image_size, "epoch": epoch,
                            "state_dict": model.state_dict(), "val": row}, out / "best.pt")
            print(f"epoch {epoch:>2}  train {train_loss:.4f}  val {val_loss:.4f}  "
                  f"pr_auc {val['pr_auc']:.4f}  recall {val['recall']:.3f}  "
                  f"recall@far5 {val['recall_at_far5']:.3f}  {time.time() - start:.0f}s{'  *' if improved else ''}")

        best_row = best[1]
        mlflow.log_metrics({f"best_{k}": v for k, v in best_row.items()})
        (out / "history.json").write_text(json.dumps(history, indent=2))
        plot_curves(history, out / "curves.png")
        for name in ["config.json", "history.json", "curves.png", "best.pt"]:
            mlflow.log_artifact(str(out / name))
        print(f"best epoch {best_row['epoch']}: pr_auc {best_row['val_pr_auc']:.4f}, saved {out / 'best.pt'}")


if __name__ == "__main__":
    main()
