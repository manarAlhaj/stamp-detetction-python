"""
Re-run validation on already-trained fold weights, without retraining.

cv_train.py's own run had plots=False, so no val_batch*_labels.jpg /
val_batch*_pred.jpg / confusion matrix / PR curve got saved. Since each
fold's best.pt is already on disk, this just re-runs YOLO's val() against
it with plots=True - inference over the val set only, nothing trains.

Usage:
    python analysis/cv_eval.py
    python analysis/cv_eval.py --only-fold 2
"""

import argparse
import json

from ultralytics import YOLO

from cv_train import CV_ROOT, FOLDS_JSON, summarize


def eval_fold(i):
    weights = CV_ROOT / f"fold_{i}_run" / "weights" / "best.pt"
    data_yaml = CV_ROOT / f"fold_{i}" / "data.yaml"
    if not weights.exists():
        print(f"fold {i}: no weights at {weights}, skipping")
        return None
    if not data_yaml.exists():
        print(f"fold {i}: no data.yaml at {data_yaml}, skipping")
        return None

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split="val", plots=True,
                         project=str(CV_ROOT), name=f"fold_{i}_run", exist_ok=True)

    return {
        "fold": i,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default=str(FOLDS_JSON))
    ap.add_argument("--only-fold", type=int, default=None,
                    help="re-evaluate a single fold index instead of all of them")
    ap.add_argument("--out", default=str(CV_ROOT.parent / "analysis" / "cv_results_reeval.json"))
    args = ap.parse_args()

    with open(args.folds) as f:
        folds_data = json.load(f)
    fold_indices = [f["fold"] for f in folds_data["folds"]]
    if args.only_fold is not None:
        fold_indices = [i for i in fold_indices if i == args.only_fold]

    results = [r for i in fold_indices if (r := eval_fold(i)) is not None]

    print("\n=== CV results (re-evaluated) ===")
    for r in results:
        print(f"fold {r['fold']}: mAP50={r['map50']:.3f}  mAP50-95={r['map50_95']:.3f}  "
              f"P={r['precision']:.3f}  R={r['recall']:.3f}")

    summary = summarize(results) if len(results) > 1 else None
    if summary:
        for k, s in summary.items():
            print(f"{k:10s} mean={s['mean']:.3f}  std={s['std']:.3f}  "
                  f"[{s['min']:.3f}, {s['max']:.3f}]")

    with open(args.out, "w") as f:
        json.dump({"folds": results, "summary": summary}, f, indent=2)
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
