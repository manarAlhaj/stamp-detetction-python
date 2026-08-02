# Main-bank real-world evaluation findings

**Test set**: 6 scans in `data27jul/testog/` — clearing/endorsement stamps from
three Bahraini banks (Al Salam Bank, BBK, National Bank of Bahrain). Confirmed
via MD5 against `yolofiles/images`, `data27jul/train`, `images_with_stamps`,
and `images_no_stamps` that none of these 6 images exist anywhere in the
training/labeled corpus — none of these three banks are represented as a
depositor in the dataset at all. This is a genuine out-of-sample test.

**Models compared**:
- `cv_runs/fold_{0-4}_run/weights/best.pt` — the 5 CV diagnostic fold models
  (each trained on ~150-160 real pages + only 2,000 synthetic pages, a
  deliberately lightweight config chosen for CV turnaround speed, not
  production quality)
- `stampdetection/best.pt` — the original pre-existing production model
  (YOLOv8n, ~600 training images, predates this session's work)

Run via `analysis/cv_pred.py --weights <path> --conf 0.15`.

## Results

| image | contents | fold0 | fold1 | fold2 | fold3 | fold4 | old yolov8 (600 img) |
|---|---|---|---|---|---|---|---|
| `image (2).png` | Al Salam Bank | 0.88 | 0.93 | 0.90 | 0.88 | 0.91 | 0.95 |
| `image (3).png` | Al Salam Bank | ✓ | ✓ | ✓ | ✓ | ✓ | 0.95 |
| `image (4).png` | BBK "Presentment" + Tasheelat Automotive (adjacent stamps) | **both** (0.82, 0.87) | Tasheelat only (0.79) | Tasheelat only (0.57) | **both** (0.48, 0.82) | Tasheelat only (0.91) | **both** (0.93, 0.95) |
| `image (5).png` | BBK "Presentment / Main Branch" | 0.87 (+ FP 0.55 on handwriting) | 0.88 | 0.84 | 0.77 | 0.52 | 0.94 |
| `image (6).png` | National Bank of Bahrain, Souq Waqef Branch | 0.90 | 0.95 | 0.93 | 0.88 | 0.92 | 0.95 |
| `image (7).png` | National Bank of Bahrain, Isa Town Branch | **0.74** | miss | miss | miss | miss | 0.79 |

## Observations

### 1. The same specific stamp instances fail consistently across independently-trained folds

Image (4)'s BBK/"Presentment" stamp is missed by 3 of 5 folds, while the
"Tasheelat Automotive" stamp on the *same page*, right next to it, is caught
by all 5. Image (7)'s NBB stamp is missed by 4 of 5 folds, while image (6)'s
NBB stamp — same bank, same design template — is caught by all 5.

This isn't fold-to-fold random noise: the same instances fail repeatedly
across models trained on different held-out depositor groups, which points to
a property of *those specific stamp images*, not of any one fold's training
data.

### 2. Rotation was a plausible first hypothesis, but doesn't hold up

`generate.py`'s synthesis never produces a near-axis-aligned stamp —
`rotation_deg=(5, 10)` means every synthetic training stamp is rotated at
least 5° in one direction or the other. Image (6) (visibly rotated ~15°,
caught by all 5 folds) vs. image (7) (perfectly horizontal, missed by 4 of 5)
initially fit this story well.

It breaks on closer inspection: image (5) is the *same* BBK "Presentment"
design as image (4)'s missed stamp, is also perfectly axis-aligned, and is
caught by all 5 folds. Rotation alone doesn't explain the split.

### 3. The actual differentiator looks like size + ink boldness + edge proximity

Zooming into the two BBK "Presentment" stamps side by side (image (4) vs.
image (5), same design/device):
- Image (5)'s stamp: larger, bolder ink, comfortably positioned within the
  page margin.
- Image (4)'s stamp: smaller, fainter ink, sitting right at the page's
  top-right corner.

The fragile cases across this whole test set share this profile — small,
faint, near an edge — rather than any single factor (rotation, or the
presence of a decorative icon) in isolation.

### 4. The old 600-image YOLOv8 model caught every hard case the lightweight CV folds mostly miss

This is the important one for interpreting "reasons behind the variance."
The CV fold models were deliberately trained at a lightweight diagnostic
scale (`n_synth=2000`/fold) to keep the 5-fold comparison fast — not at the
scale already shown to produce strong results (the earlier single-fold
20,000-synthetic benchmark, and `train_production.py`, which also uses
20,000). The old model, despite being an older architecture (YOLOv8n) trained
on far fewer total images, apparently had enough exposure (real image count,
composition, or augmentation range - not fully characterized here) to
generalize to these small/faint/edge-adjacent cases in a way the lightweight
CV folds don't.

**Conclusion**: this is evidence about the CV *diagnostic* scale being
intentionally too light for production comparison, not evidence that the new
pipeline underperforms the old one. The correct comparison point is
`production_run/weights/best.pt` (once `train_production.py` finishes)
against `stampdetection/best.pt`, not the CV fold models against it.

## Next step

```
python analysis/cv_pred.py --weights production_run/weights/best.pt
```

Compare against the `old yolov8 (600 img)` column above — that's the
meaningful production-readiness check for these three banks.
