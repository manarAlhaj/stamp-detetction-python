
import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = REPO_ROOT / "yolofiles" / "images"
LABELS_DIR = REPO_ROOT / "yolofiles" / "labels"
OUT_PATH = REPO_ROOT / "analysis" / "cv_folds.json"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def depositor_of(stem):
    return stem.split("_")[0] if "_" in stem else stem


def find_pairs(images_dir, labels_dir):
    pairs = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = labels_dir / f"{image.stem}.txt"
        if not label.exists():
            print(f"skip {image.name}: no matching label")
            continue
        pairs.append((image, label))
    return pairs


def assign_folds(groups, k, seed): #lptf
   
    import random
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)

    bins = [[] for _ in range(k)]
    bin_counts = [0] * k
    for dep, pairs in items:
        i = min(range(k), key=lambda b: bin_counts[b])
        bins[i].append(dep)
        bin_counts[i] += len(pairs)
    return bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default=str(IMAGES_DIR))
    ap.add_argument("--labels-dir", default=str(LABELS_DIR))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)

    pairs = find_pairs(images_dir, labels_dir)
    print(f"valid image/label pairs: {len(pairs)}")

    groups = defaultdict(list)
    for image, label in pairs:
        groups[depositor_of(image.stem)].append((image, label))
    print(f"depositors: {len(groups)}")

    bins = assign_folds(groups, args.k, args.seed)

    total_images = sum(len(v) for v in groups.values())
    folds = []
    for i, val_deps in enumerate(bins):
        val_image_count = sum(len(groups[d]) for d in val_deps)
        folds.append({
            "fold": i,
            "val_depositors": sorted(val_deps),
            "val_image_count": val_image_count,
        })
        print(f"fold {i}: {len(val_deps):3d} depositors, "
              f"{val_image_count:4d} images "
              f"({val_image_count / total_images:.1%})")

    out = {
        "k": args.k,
        "seed": args.seed,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "total_images": total_images,
        "total_depositors": len(groups),
        "folds": folds,
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {args.k} folds to {args.out}")


if __name__ == "__main__":
    main()