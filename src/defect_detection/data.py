"""Splits, Dataset and transforms for the 512x512 originals.

The official 300x300 split leaks (see DATA.md), so splits are made from the
originals instead, keeping near-duplicate photos together in one split

Build the split file:  uv run python -m defect_detection.data split
Check a batch:         uv run python -m defect_detection.data check
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from defect_detection.leakage import ORIGINALS, distance_matrix, hash_rotations, label_of, list_images

SEED = 42
DATA_DIR = Path("data/raw")
SPLIT_FILE = Path("data/splits.csv")
IMAGE_SIZE = 224

# the pretrained weights were trained on ImageNet images normalised with these
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def group_near_duplicates(paths, threshold=8):
    """Give images that look like the same part the same group id."""
    rotated = [hash_rotations(p) for p in paths]
    dist = distance_matrix(rotated, np.stack([r[0] for r in rotated]))

    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(dist <= threshold)):
        if i != j:
            parent[find(i)] = find(j)
    return [find(i) for i in range(len(paths))]


def make_splits(data_dir, val_size=0.2, test_size=0.2, seed=SEED):
    paths = list_images(data_dir / ORIGINALS)
    labels = [label_of(p) for p in paths]
    groups = group_near_duplicates(paths)

    # split whole groups, stratified by label, so a part never shows up in two splits
    group_label = {}
    for g, y in zip(groups, labels):
        group_label[g] = max(group_label.get(g, 0), y)
    ids = sorted(group_label)
    rest, test = train_test_split(ids, test_size=test_size, random_state=seed,
                                  stratify=[group_label[g] for g in ids])
    train, val = train_test_split(rest, test_size=val_size / (1 - test_size), random_state=seed,
                                  stratify=[group_label[g] for g in rest])
    split_of = {g: "train" for g in train} | {g: "val" for g in val} | {g: "test" for g in test}

    return [
        {"path": str(p.relative_to(data_dir)), "label": y, "split": split_of[g], "group": g}
        for p, y, g in zip(paths, labels, groups)
    ]


def save_splits(rows, out):
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "split", "group"])
        writer.writeheader()
        writer.writerows(rows)


def load_splits(path):
    with open(path) as f:
        return [{**row, "label": int(row["label"])} for row in csv.DictReader(f)]


class RandomRotateShift:
    """Rotate by any angle and shift a few pixels.

    Rotating leaves empty corners. The default black fill (or any flat colour) leaves
    a visible square edge that only training images would have, so the image is
    padded by repeating its edge pixels first and cropped back afterwards.
    """

    def __init__(self, max_shift=0.02):
        self.max_shift = max_shift

    def __call__(self, img):
        w, h = img.size
        pad = w // 4  # enough to cover the corners at 45 degrees plus the shift
        padded = Image.fromarray(np.pad(np.asarray(img), pad, mode="edge"))
        angle = torch.empty(1).uniform_(0, 360).item()
        limit = self.max_shift * w
        dx, dy = torch.empty(2).uniform_(-limit, limit).round().int().tolist()
        rotated = padded.rotate(angle, Image.BILINEAR, translate=(dx, dy))
        return rotated.crop((pad, pad, pad + w, pad + h))


def to_tensor():
    return [
        T.Grayscale(num_output_channels=3),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]


def train_transform(size=IMAGE_SIZE):
    # reasons for each one are in DATA.md
    return T.Compose([
        T.Resize(size),
        RandomRotateShift(),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        *to_tensor(),
    ])


def eval_transform(size=IMAGE_SIZE):
    return T.Compose([T.Resize(size), *to_tensor()])


class CastingDataset(Dataset):
    def __init__(self, split, transform, split_file=SPLIT_FILE, data_dir=DATA_DIR):
        self.rows = [r for r in load_splits(split_file) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"no rows for split {split!r} in {split_file}")
        self.transform = transform
        self.data_dir = Path(data_dir)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        img = Image.open(self.data_dir / row["path"]).convert("L")
        return self.transform(img), row["label"], row["path"]


def make_loader(split, batch_size=32, num_workers=0, **kwargs):
    train = split == "train"
    dataset = CastingDataset(split, train_transform() if train else eval_transform(), **kwargs)
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      generator=generator, persistent_workers=num_workers > 0)


def unnormalize(x):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)[0]


def save_augmented_examples(out, n_images=4, n_augments=5):
    import matplotlib.pyplot as plt

    plain = CastingDataset("train", eval_transform())
    augmented = CastingDataset("train", train_transform())
    picks = torch.randperm(len(plain), generator=torch.Generator().manual_seed(SEED))[:n_images]

    fig, axes = plt.subplots(n_images, n_augments + 1, figsize=(2 * (n_augments + 1), 2.2 * n_images))
    for row, i in enumerate(picks.tolist()):
        img, label, path = plain[i]
        axes[row, 0].imshow(unnormalize(img), cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{'defective' if label else 'ok'} (plain)", fontsize=8)
        for col in range(1, n_augments + 1):
            axes[row, col].imshow(unnormalize(augmented[i][0]), cmap="gray", vmin=0, vmax=1)
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def check(batch_size=32):
    torch.manual_seed(SEED)
    for split in ["train", "val", "test"]:
        images, labels, paths = next(iter(make_loader(split, batch_size)))
        counts = Counter(labels.tolist())
        print(f"{split:<5} batch {tuple(images.shape)} {images.dtype}, "
              f"range {images.min():.2f}..{images.max():.2f}, "
              f"{counts[1]} defective / {counts[0]} ok, first: {paths[0]}")
    out = Path("reports/figures/augmented_examples.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_augmented_examples(out)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    split_cmd = sub.add_parser("split", help="build the split file")
    split_cmd.add_argument("--data-dir", type=Path, default=DATA_DIR)
    split_cmd.add_argument("--out", type=Path, default=SPLIT_FILE)
    sub.add_parser("check", help="load a batch and save augmented examples")
    args = parser.parse_args()

    if args.command == "check":
        check()
        return

    rows = make_splits(args.data_dir)
    save_splits(rows, args.out)

    counts = Counter((r["split"], r["label"]) for r in rows)
    for split in ["train", "val", "test"]:
        bad, ok = counts[(split, 1)], counts[(split, 0)]
        print(f"{split:<5} {bad + ok:>4} images, {bad} defective ({bad / (bad + ok):.1%})")
    merged = len(rows) - len({r["group"] for r in rows})
    print(f"{merged} images merged into near-duplicate groups")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
