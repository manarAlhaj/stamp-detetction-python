"""
Run a trained model against a folder of real-world images and save the
annotated predictions for visual review - e.g. sanity-checking a CV fold
model or the production model against genuinely new, unseen-bank scans
before shipping.

Usage:
    python analysis/cv_pred.py --weights ../cv_runs/fold_0_run/weights/best.pt
    python analysis/cv_pred.py --weights ../production_run/weights/best.pt --conf 0.15
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "data27jul" / "testog"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to a best.pt")
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--conf", type=float, default=0.15,
                    help="lower than the usual 0.25 default, since we care "
                         "about seeing near-miss/low-confidence detections "
                         "on genuinely unseen banks, not just clean ones")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--project", default=str(REPO_ROOT / "mainbank_preds"))
    ap.add_argument("--name", default="run")
    args = ap.parse_args()

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    print(f"\n=== predictions ({Path(args.weights).parent.parent.name}) ===")
    for r in results:
        n = len(r.boxes)
        confs = [f"{c:.2f}" for c in r.boxes.conf.tolist()] if n else []
        print(f"{Path(r.path).name}: {n} stamp(s)  {confs}")

    print(f"\nannotated images saved to {args.project}/{args.name}/")


if __name__ == "__main__":
    main()
