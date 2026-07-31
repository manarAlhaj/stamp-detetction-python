

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import generate as gen

REPO_ROOT = Path(__file__).resolve().parent.parent

PAGES_DIR = gen.PAGES_DIR
CROPS_DIR = gen.CROPS_DIR

LABELS_DIR = str(REPO_ROOT / "yolofiles" / "labels")

OUT_DIR = str(REPO_ROOT / "analysis" / "recovered_backgrounds")

MARGIN = 12
INK_THRESH = 0.15    
DILATE_PX = 2       
INPAINT_RADIUS = 3
METHOD = cv2.INPAINT_TELEA   


def read_yolo_boxes(label_path, W, H):
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = parts[:5]
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            x0, y0 = int(cx - w / 2), int(cy - h / 2)
            x1, y1 = int(cx + w / 2), int(cy + h / 2)
            boxes.append((max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)))
    return boxes


def build_ink_mask(page, boxes, margin=MARGIN):

    H, W = page.shape
    mask = np.zeros((H, W), np.uint8)
    for (x0, y0, x1, y1) in boxes:
        mx0, my0 = max(x0 - margin, 0), max(y0 - margin, 0)
        mx1, my1 = min(x1 + margin, W), min(y1 + margin, H)
        local = page[my0:my1, mx0:mx1]
        alpha = gen.extract_alpha(local)
        local_mask = (alpha > INK_THRESH).astype(np.uint8) * 255
        if DILATE_PX:
            local_mask = cv2.dilate(local_mask, np.ones((DILATE_PX, DILATE_PX), np.uint8))
        mask[my0:my1, mx0:mx1] = np.maximum(mask[my0:my1, mx0:mx1], local_mask)
    return mask


def recover_page(page_path, label_path, out_path):
    page = gen.load_gray(page_path)
    H, W = page.shape
    boxes = read_yolo_boxes(label_path, W, H)
    if not boxes:
        return False
    mask = build_ink_mask(page, boxes)
    if mask.sum() == 0:
        return False
    clean = cv2.inpaint(page, mask, INPAINT_RADIUS, METHOD)
    Image.fromarray(clean).save(out_path)
    return True


def main(pages_dir, labels_dir, crops_dir, out_dir, excluded_depositors=frozenset()):
    os.makedirs(out_dir, exist_ok=True)
    stamped_stems = {gen.crop_stem(f) for f in os.listdir(crops_dir) if f.endswith(".png")}

    done, skipped, excluded = 0, 0, 0
    for f in sorted(os.listdir(pages_dir)):
        if not f.endswith(".png"):
            continue
        stem = gen.page_stem(f)
        if stem not in stamped_stems:
            continue  # already unstamped, nothing to recover here
        if gen.depositor_of(stem) in excluded_depositors:
            excluded += 1
            continue  # held-out depositor: never recover into a usable background

        label_path = os.path.join(labels_dir, f[:-4] + ".txt")
        ok = recover_page(
            os.path.join(pages_dir, f),
            label_path,
            os.path.join(out_dir, f),
        )
        done += int(ok)
        skipped += int(not ok)

    print(f"recovered {done} backgrounds, skipped {skipped} (no boxes found), "
          f"excluded {excluded} (held-out depositors) -> {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-result", default=gen.SPLIT_RESULT_PATH,
                    help="path to split_result.json; that file's val_depositors "
                         "are skipped so recovered backgrounds never leak held-out pages")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--pages-dir", default=PAGES_DIR)
    ap.add_argument("--labels-dir", default=LABELS_DIR)
    ap.add_argument("--crops-dir", default=CROPS_DIR)
    a = ap.parse_args()
    val_depositors = gen.load_val_depositors(a.split_result)
    print(f"excluding {len(val_depositors)} val depositors from background recovery")
    main(a.pages_dir, a.labels_dir, a.crops_dir, a.out, excluded_depositors=val_depositors)