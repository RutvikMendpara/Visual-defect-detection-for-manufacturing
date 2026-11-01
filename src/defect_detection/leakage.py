"""Check how many images in the official test split are near-copies of training images.

Run with: uv run python -m defect_detection.leakage
"""

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps

OFFICIAL = Path("casting_data/casting_data")
ORIGINALS = Path("casting_512x512/casting_512x512")

HASH_SIZE = 16  # 16x16 = 256 bits. 64 bits was too coarse, every impeller looked alike
WORK_SIZE = 128
# rotating leaves empty corners, so only the middle of the image is hashed
CENTRE = (19, 19, 109, 109)


def dct_matrix(n):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2 / n)
    m[0] /= np.sqrt(2)
    return m


DCT = dct_matrix(HASH_SIZE * 4)


def phash(img):
    # same idea as imagehash.phash: keep the low frequencies of a DCT, threshold at the median
    small = img.resize((HASH_SIZE * 4, HASH_SIZE * 4), Image.LANCZOS)
    a = np.asarray(small, dtype=np.float64)
    low = (DCT @ a @ DCT.T)[:HASH_SIZE, :HASH_SIZE]
    return np.packbits(low > np.median(low))


def load_gray(path):
    return Image.open(path).convert("L").resize((WORK_SIZE, WORK_SIZE), Image.LANCZOS)


def hash_image(path):
    return phash(load_gray(path).crop(CENTRE))


def hash_rotations(path, step=5):
    """Hashes of the image and its mirror, rotated every `step` degrees."""
    img = load_gray(path)
    background = int(np.median(np.asarray(img)[:8]))
    hashes = []
    for version in (img, ImageOps.mirror(img)):
        for angle in range(0, 360, step):
            rotated = version.rotate(angle, Image.BILINEAR, fillcolor=background)
            hashes.append(phash(rotated.crop(CENTRE)))
    return np.stack(hashes)


def distance_matrix(rotated, plain):
    """Hamming distance from every query to every reference, using the best rotation of the query."""
    rows = []
    for q in rotated:
        bits = np.bitwise_count(q[:, None, :] ^ plain[None, :, :]).sum(axis=2)
        rows.append(bits.min(axis=0))
    return np.stack(rows).astype(int)


def list_images(folder):
    return sorted(folder.glob("*_front/*.jpeg"))


def label_of(path):
    return 1 if path.parent.name == "def_front" else 0


def plot_distances(test_dist, orig_dist, threshold, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, 66, 2)
    ax.hist(test_dist, bins=bins, alpha=0.6, density=True, label="test image -> closest train image")
    ax.hist(orig_dist, bins=bins, alpha=0.6, density=True, label="original -> closest other original")
    ax.axvline(threshold + 1, color="black", linestyle="--", label=f"cut-off ({threshold})")
    ax.set_xlabel("Hamming distance (256-bit pHash, best rotation/flip)")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def save_pairs(pairs, out, size=220):
    sheet = Image.new("RGB", (2 * size + 10, len(pairs) * (size + 20)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (test_path, train_path, dist) in enumerate(pairs):
        y = row * (size + 20)
        draw.text((4, y + 3), f"test {test_path.name}  |  train {train_path.name}  d={dist}", fill="black")
        sheet.paste(Image.open(test_path).convert("RGB").resize((size, size)), (0, y + 18))
        sheet.paste(Image.open(train_path).convert("RGB").resize((size, size)), (size + 10, y + 18))
    sheet.save(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--step", type=int, default=5, help="rotation step in degrees")
    args = parser.parse_args()

    train = list_images(args.data_dir / OFFICIAL / "train")
    test = list_images(args.data_dir / OFFICIAL / "test")
    originals = list_images(args.data_dir / ORIGINALS)
    print(f"{len(train)} train, {len(test)} test, {len(originals)} originals")

    print("hashing train...")
    train_hashes = np.stack([hash_image(p) for p in train])
    print("hashing test (all rotations)...")
    test_dist = distance_matrix([hash_rotations(p, args.step) for p in test], train_hashes)
    nearest = test_dist.argmin(axis=1)
    test_min = test_dist.min(axis=1)

    # baseline: the 512x512 originals should all be different parts, so their
    # nearest-neighbour distances show how close two *different* impellers can get
    print("hashing originals (all rotations)...")
    orig_rot = [hash_rotations(p, args.step) for p in originals]
    orig_dist = distance_matrix(orig_rot, np.stack([r[0] for r in orig_rot]))
    np.fill_diagonal(orig_dist, 10**6)
    orig_min = orig_dist.min(axis=1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(exist_ok=True)

    with open(args.out_dir / "leakage.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_image", "label", "closest_train_image", "distance", "leaked"])
        for path, idx, dist in zip(test, nearest, test_min):
            writer.writerow([path.relative_to(args.data_dir), label_of(path),
                             train[idx].relative_to(args.data_dir), dist, int(dist <= args.threshold)])

    lines = [
        f"rotation step {args.step} deg, mirror included, 256-bit pHash",
        f"cut-off: distance <= {args.threshold}",
        "",
        "cut-off  leaked test images  originals flagged (false alarms)",
    ]
    for t in [4, 6, 8, 10, 12, 16]:
        leaked = (test_min <= t).mean() * 100
        false = (orig_min <= t).mean() * 100
        lines.append(f"{t:>7}  {leaked:>17.1f}%  {false:>30.1f}%")
    leaked = test_min <= args.threshold
    for label, name in [(1, "defective"), (0, "ok")]:
        mask = np.array([label_of(p) == label for p in test])
        lines.append(f"{name}: {leaked[mask].sum()} of {mask.sum()} leaked")
    lines.append(f"total: {leaked.sum()} of {len(test)} ({leaked.mean() * 100:.1f}%)")
    summary = "\n".join(lines)
    print(summary)
    (args.out_dir / "leakage.txt").write_text(summary + "\n")

    plot_distances(test_min, orig_min, args.threshold, args.out_dir / "figures/leakage_distances.png")

    flagged = [i for i in range(len(test)) if leaked[i]]
    picks = random.Random(42).sample(flagged, min(10, len(flagged)))
    save_pairs([(test[i], train[nearest[i]], test_min[i]) for i in picks],
               args.out_dir / "figures/leakage_pairs.png")
    print(f"wrote {args.out_dir}/leakage.csv, leakage.txt and figures")


if __name__ == "__main__":
    main()
