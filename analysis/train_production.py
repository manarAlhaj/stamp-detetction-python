"""
Train the actual production model: one single train/val split (from
splitbybank.py's split_result.json), not per-depositor CV folds. Use
cv_train.py/cv_split.py for the generalization diagnostic; use this to
produce the one model that actually ships.

Usage:
    python analysis/splitbybank.py                 # (re)build split_result.json
    python analysis/train_production.py --n-synth 20000 --gen-workers 12 --device 0
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

import generate
from cv_train import REPO, TRAIN_CFG, depositor_of, find_pairs

YOLOFILES_IMAGES = REPO / "yolofiles" / "images"
YOLOFILES_LABELS = REPO / "yolofiles" / "labels"
SPLIT_RESULT_PATH = REPO / "analysis" / "split_result.json"
RUN_ROOT = REPO / "production_run"
RESULTS_JSON = REPO / "analysis" / "production_results.json"


def build_dataset(split_result_path, images_dir, labels_dir, run_dir):
    with open(split_result_path) as f:
        val_depositors = set(json.load(f)["val_depositors"])

    train_images = run_dir / "images" / "train"
    val_images = run_dir / "images" / "val"
    train_labels = run_dir / "labels" / "train"
    val_labels = run_dir / "labels" / "val"
    for d in (train_images, val_images, train_labels, val_labels):
        d.mkdir(parents=True, exist_ok=True)

    import shutil
    n_train = n_val = 0
    for image, label in find_pairs(images_dir, labels_dir):
        is_val = depositor_of(image.stem) in val_depositors
        dst_images = val_images if is_val else train_images
        dst_labels = val_labels if is_val else train_labels
        shutil.copy2(image, dst_images / image.name)
        shutil.copy2(label, dst_labels / label.name)
        n_val += is_val
        n_train += not is_val

    return n_train, n_val


def write_data_yaml(run_dir, synth_dir, has_synth):
    lines = [
        "train:",
        f"  - {run_dir / 'images' / 'train'}",
    ]
    if has_synth:
        lines.append(f"  - {synth_dir / 'images'}")
    lines += [
        f"val: {run_dir / 'images' / 'val'}",
        "names:",
        "  0: stamp",
    ]
    data_yaml = run_dir / "data.yaml"
    data_yaml.write_text("\n".join(lines) + "\n")
    return data_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-result", default=str(SPLIT_RESULT_PATH))
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--device", default="0")
    ap.add_argument("--n-synth", type=int, default=20000)
    ap.add_argument("--gen-workers", type=int, default=1)
    ap.add_argument("--pool-size", type=int, default=None)
    ap.add_argument("--name", default="production_run")
    args = ap.parse_args()

    if args.pool_size:
        generate.CONFIG["pool_size"] = args.pool_size

    if RUN_ROOT.exists():
        import shutil
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True)

    n_train, n_val = build_dataset(args.split_result, YOLOFILES_IMAGES,
                                    YOLOFILES_LABELS, RUN_ROOT)
    print(f"real pages: {n_train} train, {n_val} val")

    synth_dir = RUN_ROOT / "synth"
    has_synth = args.n_synth > 0
    if has_synth:
        generate.generate(
            args.n_synth, str(synth_dir), generate.CONFIG, seed=0,
            save_stamps=False, split_result_path=args.split_result,
            workers=args.gen_workers,
        )

    data_yaml = write_data_yaml(RUN_ROOT, synth_dir, has_synth)

    model = YOLO(args.weights)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(REPO),
        name=args.name,
        exist_ok=True,
        plots=True,
        **TRAIN_CFG,
    )

    best_weights = REPO / args.name / "weights" / "best.pt"
    val_model = YOLO(str(best_weights))
    metrics = val_model.val(data=str(data_yaml), split="val", plots=True,
                             project=str(REPO), name=args.name, exist_ok=True)

    result = {
        "n_train_images": n_train,
        "n_val_images": n_val,
        "n_synth_images": args.n_synth,
        "weights": str(best_weights),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nmAP50={result['map50']:.3f}  mAP50-95={result['map50_95']:.3f}  "
          f"P={result['precision']:.3f}  R={result['recall']:.3f}")
    print(f"best weights: {best_weights}")
    print(f"saved metrics to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
