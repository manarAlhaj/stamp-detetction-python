

import argparse
import json
import shutil
import statistics
from pathlib import Path

from ultralytics import YOLO

import generate

REPO = Path(__file__).resolve().parent.parent
YOLOFILES_IMAGES = REPO / "yolofiles" / "images"
YOLOFILES_LABELS = REPO / "yolofiles" / "labels"
FOLDS_JSON = REPO / "analysis" / "cv_folds.json"
CV_ROOT = REPO / "cv_runs"
RESULTS_JSON = REPO / "analysis" / "cv_results.json"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


TRAIN_CFG = dict(
    degrees=10,
    translate=0.1,
    scale=0.2,
    fliplr=0.0,
    flipud=0.0,
    hsv_h=0,
    hsv_s=0,
    hsv_v=0.3,
    mosaic=0.0,
)


def depositor_of(stem):
    return stem.split("_")[0] if "_" in stem else stem


def find_pairs(images_dir, labels_dir):
    pairs = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = labels_dir / f"{image.stem}.txt"
        if label.exists():
            pairs.append((image, label))
    return pairs


def build_fold_dataset(fold, images_dir, labels_dir, fold_dir):
    val_depositors = set(fold["val_depositors"])

    train_images = fold_dir / "images" / "train"
    val_images = fold_dir / "images" / "val"
    train_labels = fold_dir / "labels" / "train"
    val_labels = fold_dir / "labels" / "val"
    for d in (train_images, val_images, train_labels, val_labels):
        d.mkdir(parents=True, exist_ok=True)

    n_train = n_val = 0
    for image, label in find_pairs(images_dir, labels_dir):
        is_val = depositor_of(image.stem) in val_depositors
        dst_images = val_images if is_val else train_images
        dst_labels = val_labels if is_val else train_labels
        shutil.copy2(image, dst_images / image.name)
        shutil.copy2(label, dst_labels / label.name)
        n_val += is_val
        n_train += not is_val

    split_result_path = fold_dir / "split_result.json"
    with open(split_result_path, "w") as f:
        json.dump({"val_depositors": sorted(val_depositors)}, f, indent=2)

    return n_train, n_val, split_result_path


def write_data_yaml(fold_dir, synth_dir, n_train_synth):
    lines = [
        "train:",
        f"  - {fold_dir / 'images' / 'train'}",
    ]
    if n_train_synth:
        lines.append(f"  - {synth_dir / 'images'}")
    lines += [
        f"val: {fold_dir / 'images' / 'val'}",
        "names:",
        "  0: stamp",
    ]
    data_yaml = fold_dir / "data.yaml"
    data_yaml.write_text("\n".join(lines) + "\n")
    return data_yaml


def run_fold(fold, args):
    i = fold["fold"]
    fold_dir = CV_ROOT / f"fold_{i}"
    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    fold_dir.mkdir(parents=True)

    n_train, n_val, split_result_path = build_fold_dataset(
        fold, YOLOFILES_IMAGES, YOLOFILES_LABELS, fold_dir)
    print(f"\n=== fold {i}: {n_train} real train pages, {n_val} real val pages "
          f"({len(fold['val_depositors'])} val depositors) ===")

    synth_dir = fold_dir / "synth"
    n_synth_boxes = 0
    if args.n_synth > 0:
        generate.generate(
            args.n_synth, str(synth_dir), generate.CONFIG, seed=i,
            save_stamps=False, split_result_path=str(split_result_path),
            workers=args.gen_workers,
        )
        n_synth_boxes = args.n_synth

    data_yaml = write_data_yaml(fold_dir, synth_dir, n_synth_boxes)

    model = YOLO(args.weights)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(CV_ROOT),
        name=f"fold_{i}_run",
        exist_ok=True,
        plots=False,
        **TRAIN_CFG,
    )

    best_weights = CV_ROOT / f"fold_{i}_run" / "weights" / "best.pt"
    val_model = YOLO(str(best_weights))
    metrics = val_model.val(data=str(data_yaml), split="val", plots=False)

    return {
        "fold": i,
        "val_depositors": fold["val_depositors"],
        "n_val_images": n_val,
        "n_train_images": n_train,
        "n_synth_images": n_synth_boxes,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def summarize(fold_results):
    keys = ["map50", "map50_95", "precision", "recall"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in fold_results]
        summary[k] = {
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default=str(FOLDS_JSON))
    ap.add_argument("--weights", default="yolo11n.pt",
                    help="starting weights - COCO-pretrained by default; do NOT "
                         "pass stampdetection/best.pt, it was already trained on "
                         "every depositor and would leak into every fold's val set")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=25,
                    help="stop a fold early if val metrics don't improve for "
                         "this many epochs (matches the original stampdetection "
                         "recipe); Ultralytics' own default is 100, i.e. "
                         "effectively never for a 100-epoch run")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--device", default="0")
    ap.add_argument("--n-synth", type=int, default=600,
                    help="synthetic pages to generate per fold, 0 to disable")
    ap.add_argument("--gen-workers", type=int, default=1,
                    help="worker processes for synthetic page generation "
                         "(analysis/generate.py); 1 = sequential")
    ap.add_argument("--pool-size", type=int, default=None,
                    help="override generate.CONFIG['pool_size'] (procedural "
                         "fake-stamp designs); grows analysis/proc_pool/ on "
                         "first fold's synthesis call if it's currently smaller")
    ap.add_argument("--only-fold", type=int, default=None,
                    help="run a single fold index (for smoke-testing)")
    ap.add_argument("--out", default=str(RESULTS_JSON))
    args = ap.parse_args()

    if args.pool_size:
        generate.CONFIG["pool_size"] = args.pool_size

    with open(args.folds) as f:
        folds_data = json.load(f)
    folds = folds_data["folds"]
    if args.only_fold is not None:
        folds = [f for f in folds if f["fold"] == args.only_fold]

    CV_ROOT.mkdir(exist_ok=True)

    fold_results = [run_fold(fold, args) for fold in folds]

    out = {
        "k": folds_data["k"],
        "weights": args.weights,
        "epochs": args.epochs,
        "n_synth_per_fold": args.n_synth,
        "folds": fold_results,
        "summary": summarize(fold_results) if len(fold_results) > 1 else None,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== CV results ===")
    for r in fold_results:
        print(f"fold {r['fold']}: mAP50={r['map50']:.3f}  mAP50-95={r['map50_95']:.3f}  "
              f"P={r['precision']:.3f}  R={r['recall']:.3f}  (n_val={r['n_val_images']})")
    if out["summary"]:
        for k, s in out["summary"].items():
            print(f"{k:10s} mean={s['mean']:.3f}  std={s['std']:.3f}  "
                  f"[{s['min']:.3f}, {s['max']:.3f}]")
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()