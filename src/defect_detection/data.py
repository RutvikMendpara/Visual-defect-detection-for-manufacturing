"""Train/val/test split built from the 512x512 originals.

The official 300x300 split leaks (see DATA.md), so splits are made from the
originals instead, keeping near-duplicate photos together in one split

Run with: uv run python -m defect_detection.data
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from defect_detection.leakage import ORIGINALS, distance_matrix, hash_rotations, label_of, list_images

SEED = 42


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/splits.csv"))
    args = parser.parse_args()

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
