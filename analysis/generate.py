
import argparse
import collections
import contextlib
import json
import multiprocessing as mp
import os
import random
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import generate_fake_stamps as fake

# must happen before any cv2 call, in the parent as well as workers: OpenCV
# otherwise spins up its own native thread pool, and forking a
# multi-threaded process (multiprocessing.Pool's default on Linux) can
# deadlock a child that inherits a lock held by one of those threads at the
# moment of fork.
cv2.setNumThreads(1)

# derived from this file's location (analysis/), not hardcoded, so the repo
# works from wherever it's cloned rather than one specific machine's home dir
REPO_ROOT = Path(__file__).resolve().parent.parent

PAGES_DIR = str(REPO_ROOT / "data27jul" / "train")
CROPS_DIR = str(REPO_ROOT / "stamp_crops_from_labels")
PROC_POOL_DIR = str(REPO_ROOT / "analysis" / "proc_pool")
SPLIT_RESULT_PATH = str(REPO_ROOT / "analysis" / "split_result.json")
RECOVERED_DIR = str(REPO_ROOT / "analysis" / "recovered_backgrounds")

CONFIG = dict(

    procedural_frac=0.75,


    pool_size=3000,

    # --- density -----------------------------------------------------------
    stamps_per_page=(5, 8),
    area_frac=(0.030, 0.075),


    rotation_deg=(5, 10),

    # --- overlap -------------------------------------------------------
    overlap_prob=0.2,
    overlap_offset=(0.15, 0.55),

    max_incidental_iou=0.3,

    # --- ink
    fade_prob=0.55,
    fade_range=(0.30, 1.00),
    ramp_prob=0.40,
    blotch_prob=0.50,
    erode_prob=0.25,
    dilate_prob=0.15,
    ink_darkness=(0.05, 0.30),

    # --- page scan degradation ---------------------------------------
    blur_sigma=(0.0, 1.2),
    noise_sigma=(2.0, 8.0),
    jpeg_quality=(45, 95),

    min_visible_frac=0.55,
)


# ---------------------------------------------------------------------------
# filename helpers / ink separation
# ---------------------------------------------------------------------------

def page_stem(f):
    m = re.match(r"^(.*)_back\.png$", f)
    return m.group(1) if m else f[:-4]


def crop_stem(f):
    m = re.match(r"^(.*)_back_stamp_\d+\.png$", f)
    if m:
        return m.group(1)
    return re.match(r"^(.*)_stamp_\d+\.png$", f).group(1)


def depositor_of(stem):
    return stem.split("_")[0]


def load_gray(path):
    im = Image.open(path)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    return np.asarray(im.convert("L"))


def extract_alpha(gray_u8):

    g = gray_u8.astype(np.float32)
    k = max(9, (int(min(g.shape) / 6)) | 1)
    paper = cv2.GaussianBlur(
        cv2.morphologyEx(g, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))),
        (0, 0), k / 3.0)
    a = np.clip(1.0 - g / np.maximum(paper, 1.0), 0.0, 1.0)
    return np.clip((a - 0.06) / 0.75, 0, 1)     # trims low-level residue


# ---------------------------------------------------------------------------
#  banks
# ---------------------------------------------------------------------------

def load_val_depositors(path):
    with open(path) as f:
        data = json.load(f)
    return set(data["val_depositors"])

def load_real_templates_by_depositor(crops_dir, excluded_depositors=frozenset()):
    by_dep = collections.defaultdict(list)
    for f in sorted(os.listdir(crops_dir)):
        if not f.endswith(".png"):
            continue
        dep = depositor_of(crop_stem(f))
        if dep in excluded_depositors:
            continue
        g = load_gray(os.path.join(crops_dir, f))
        if g.shape[0] * g.shape[1] < 2000:
            continue
        a = extract_alpha(g)
        if (a > 0.35).mean() < 0.03:
            continue
        by_dep[dep].append((a, f))
    return dict(by_dep)


def _make_one_template(task):
    idx, seed, pool_dir = task
    rng = random.Random(seed)
    img = None
    while img is None:          # make_template occasionally returns None
        img = fake.make_template(rng)   # (near-empty design); just retry
    Image.fromarray(img).save(f"{pool_dir}/proc_{idx:05d}.png")
    return idx


def ensure_procedural_pool(pool_dir, size, seed=0, workers=1):

    os.makedirs(pool_dir, exist_ok=True)
    existing_files = sorted(f for f in os.listdir(pool_dir) if f.endswith(".png"))
    existing = len(existing_files)
    if existing >= size:
        print(f"procedural pool: reusing {existing} existing designs in {pool_dir}/")
        return

    missing = size - existing
    print(f"procedural pool: {existing} cached, topping up {missing} more into "
          f"{pool_dir}/ (~{missing*0.062/max(workers,1):.0f}s at ~62ms/design, "
          f"{workers} worker(s))...")

    # continue numbering after whatever is already there; each task gets its
    # own distinct seed so designs don't repeat across tasks/workers
    tasks = [(existing + k, seed * 1_000_003 + existing + k, pool_dir)
             for k in range(missing)]

    if workers <= 1:
        for _ in map(_make_one_template, tasks):
            pass
    else:
        with mp.Pool(workers) as pool:
            for _ in pool.imap_unordered(_make_one_template, tasks, chunksize=4):
                pass


def load_procedural_pool(pool_dir):

    templates = []
    for f in sorted(os.listdir(pool_dir)):
        if not f.endswith(".png"):
            continue
        cache_path = os.path.join(pool_dir, f[:-4] + ".alpha.npy")
        if os.path.exists(cache_path):
            alpha = np.load(cache_path)
        else:
            alpha = extract_alpha(load_gray(os.path.join(pool_dir, f)))
            np.save(cache_path, alpha)
        templates.append((alpha, f))
    return templates


def load_backgrounds(pages_dir, crops_dir, excluded_depositors=frozenset(), recovered_dir=None):
    stamped = {crop_stem(f) for f in os.listdir(crops_dir) if f.endswith(".png")}
    out = []
    for f in sorted(os.listdir(pages_dir)):
        if not f.endswith(".png"):
            continue
        stem = page_stem(f)
        if stem in stamped:
            continue
        if depositor_of(stem) in excluded_depositors:
            continue
        out.append(load_gray(os.path.join(pages_dir, f)))

    if recovered_dir and os.path.isdir(recovered_dir):
        for f in sorted(os.listdir(recovered_dir)):
            if not f.endswith(".png"):
                continue
            # recovered pages were stamped originally, so re-check the
            # exclusion here too rather than trusting how they were made
            if depositor_of(page_stem(f)) in excluded_depositors:
                continue
            out.append(load_gray(os.path.join(recovered_dir, f)))

    return out


# ---------------------------------------------------------------------------
# template sampling: the actual "merge" logic
# ---------------------------------------------------------------------------

def sample_template(rng, cfg, real_by_dep, real_deps, proc_pool):

    if rng.random() < cfg["procedural_frac"] or not real_deps:
        alpha, fname = proc_pool[rng.integers(len(proc_pool))]
        return alpha, "procedural", fname
    dep = real_deps[rng.integers(len(real_deps))]
    alpha, fname = real_by_dep[dep][rng.integers(len(real_by_dep[dep]))]
    return alpha, "real", f"{dep}/{fname}"


# ---------------------------------------------------------------------------
# per stamp augmentation 
# ---------------------------------------------------------------------------

def degrade_alpha(a, cfg, rng):
    if rng.random() < cfg["erode_prob"]:
        a = cv2.erode(a, np.ones((2, 2), np.uint8))
    if rng.random() < cfg["dilate_prob"]:
        a = cv2.dilate(a, np.ones((2, 2), np.uint8))
        a = cv2.GaussianBlur(a, (0, 0), 0.6)

    if rng.random() < cfg["blotch_prob"]:
        h, w = a.shape
        n = rng.random(size=(max(h // 16, 2), max(w // 16, 2))).astype(np.float32)
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
        n = cv2.GaussianBlur(n, (0, 0), max(min(h, w) / 12, 1))
        n = (n - n.min()) / (np.ptp(n) + 1e-6)
        a = a * (0.45 + 0.55 * n)

    if rng.random() < cfg["ramp_prob"]:
        h, w = a.shape
        ang = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        proj = xx * np.cos(ang) + yy * np.sin(ang)
        proj = (proj - proj.min()) / (np.ptp(proj) + 1e-6)
        lo = rng.uniform(0.10, 0.50)
        a = a * (lo + (1.0 - lo) * proj)

    if rng.random() < cfg["fade_prob"]:
        a = a * rng.uniform(*cfg["fade_range"])

    return np.clip(a, 0, 1)


def place_stamp(canvas, occupancy, alpha, cfg, rng, existing):
    H, W = canvas.shape

    ang = rng.uniform(*cfg["rotation_deg"]) * rng.choice([-1, 1])
    ah, aw = alpha.shape
    diag = int(np.ceil(np.hypot(ah, aw)))
    pad = np.zeros((diag, diag), np.float32)
    oy, ox = (diag - ah) // 2, (diag - aw) // 2
    pad[oy:oy + ah, ox:ox + aw] = alpha
    M = cv2.getRotationMatrix2D((diag / 2, diag / 2), ang, 1.0)
    rot = cv2.warpAffine(pad, M, (diag, diag), flags=cv2.INTER_LINEAR,
                         borderValue=0)

    ys, xs = np.nonzero(rot > 0.02)
    if len(xs) < 20:
        return None
    rot = rot[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    target_area = rng.uniform(*cfg["area_frac"]) * H * W
    scale = np.sqrt(target_area / max(rot.shape[0] * rot.shape[1], 1))
    nw, nh = max(int(rot.shape[1] * scale), 8), max(int(rot.shape[0] * scale), 8)
    if nw >= W or nh >= H:
        return None
    rot = cv2.resize(rot, (nw, nh), interpolation=cv2.INTER_AREA)

    if existing and rng.random() < cfg["overlap_prob"]:
        bx0, by0, bx1, by1 = existing[rng.integers(len(existing))]
        off = rng.uniform(*cfg["overlap_offset"]) * rng.choice([-1, 1])
        x = int(bx0 + off * (bx1 - bx0))
        y = int(by0 + rng.uniform(*cfg["overlap_offset"]) * rng.choice([-1, 1]) * (by1 - by0))
        x = int(np.clip(x, 0, W - nw))
        y = int(np.clip(y, 0, H - nh))
    else:
        x = int(rng.integers(0, max(W - nw, 1)))
        y = int(rng.integers(0, max(H - nh, 1)))

        if existing and cfg.get("max_incidental_iou", 1.0) < 1.0:
            for _ in range(40):     
                worst = 0.0
                for bx0, by0, bx1, by1 in existing:
                    ix = max(0, min(x + nw, bx1) - max(x, bx0))
                    iy = max(0, min(y + nh, by1) - max(y, by0))
                    inter = ix * iy
                    union = nw * nh + (bx1 - bx0) * (by1 - by0) - inter
                    worst = max(worst, inter / max(union, 1e-9))
                if worst <= cfg["max_incidental_iou"]:
                    break
                x = int(rng.integers(0, max(W - nw, 1)))
                y = int(rng.integers(0, max(H - nh, 1)))

    region = occupancy[y:y + nh, x:x + nw]
    ink_mask = rot > 0.25
    if ink_mask.sum() > 0:
        visible = 1.0 - (region[ink_mask] > 0).mean()
        if visible < cfg["min_visible_frac"]:
            return None

    ink = rng.uniform(*cfg["ink_darkness"])
    patch = canvas[y:y + nh, x:x + nw]
    canvas[y:y + nh, x:x + nw] = patch * (1.0 - rot + rot * ink)
    occupancy[y:y + nh, x:x + nw] = np.maximum(region, ink_mask.astype(np.uint8))


    return (x, y, x + nw, y + nh), rot, ink


def degrade_page(img_f32, cfg, rng):
    out = img_f32
    s = rng.uniform(*cfg["blur_sigma"])
    if s > 0.05:
        out = cv2.GaussianBlur(out, (0, 0), s)
    out = out + rng.normal(0, rng.uniform(*cfg["noise_sigma"]), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    q = int(rng.integers(*cfg["jpeg_quality"]))
    ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return out


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# one page's worth of work, factored out so it can run under
# multiprocessing.Pool - see _init_worker/_render_page below for how the
# shared read-only inputs (backgrounds/pool/crops) get to each worker.
# ---------------------------------------------------------------------------

_WORKER_STATE = {}


def _init_worker(cfg, real_by_dep, real_deps, proc_pool, backgrounds, out_dir,
                  save_stamps):
    _WORKER_STATE.update(
        cfg=cfg, real_by_dep=real_by_dep, real_deps=real_deps,
        proc_pool=proc_pool, backgrounds=backgrounds, out_dir=out_dir,
        save_stamps=save_stamps,
    )


def _render_page(task):
    i, page_seed = task
    st = _WORKER_STATE
    cfg, real_by_dep, real_deps = st["cfg"], st["real_by_dep"], st["real_deps"]
    proc_pool, backgrounds = st["proc_pool"], st["backgrounds"]
    out_dir, save_stamps = st["out_dir"], st["save_stamps"]

    rng = np.random.default_rng(page_seed)
    name = f"synth_{i:06d}"

    bg = backgrounds[rng.integers(len(backgrounds))].astype(np.float32).copy()
    H, W = bg.shape
    occupancy = np.zeros((H, W), np.uint8)

    want = int(rng.integers(cfg["stamps_per_page"][0], cfg["stamps_per_page"][1] + 1))
    boxes = []
    source_counts = collections.Counter()
    manifest_lines = []
    for _ in range(want * 3):
        if len(boxes) >= want:
            break
        alpha, src, src_id = sample_template(rng, cfg, real_by_dep, real_deps, proc_pool)
        a = degrade_alpha(alpha.copy(), cfg, rng)
        result = place_stamp(bg, occupancy, a, cfg, rng, boxes)
        if result is None:
            continue
        box, rot, ink = result
        boxes.append(box)
        source_counts[src] += 1

        if save_stamps:
            crop_file = f"{name}_stamp_{len(boxes)-1:02d}_{src}.png"
            preview = fake.PAPER_TONE * (1.0 - rot + rot * ink)
            Image.fromarray(np.clip(preview, 0, 255).astype(np.uint8)) \
                .save(f"{out_dir}/stamps_used/{crop_file}")
            x0, y0, x1, y1 = box
            manifest_lines.append(f"{name},{len(boxes)-1},{src},{src_id},{ink:.3f},"
                                   f"{x0},{y0},{x1},{y1},{crop_file}\n")

    shortfall = max(want - len(boxes), 0)

    img = degrade_page(bg, cfg, rng)
    cv2.imwrite(f"{out_dir}/images/{name}.png", img)
    with open(f"{out_dir}/labels/{name}.txt", "w") as f:
        for x0, y0, x1, y1 in boxes:
            f.write(f"0 {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} "
                    f"{(x1-x0)/W:.6f} {(y1-y0)/H:.6f}\n")

    return len(boxes), source_counts, shortfall, manifest_lines


def generate(n, out_dir, cfg, seed=0, save_stamps=True, split_result_path=SPLIT_RESULT_PATH,
             recovered_dir=RECOVERED_DIR, workers=1):

    val_depositors = load_val_depositors(split_result_path)
    print(f"excluding {len(val_depositors)} val depositors from synthesis")

    ensure_procedural_pool(PROC_POOL_DIR, cfg["pool_size"], seed, workers=workers)
    proc_pool = load_procedural_pool(PROC_POOL_DIR)
    real_by_dep = load_real_templates_by_depositor(CROPS_DIR, excluded_depositors=val_depositors)
    real_deps = sorted(real_by_dep)
    backgrounds = load_backgrounds(PAGES_DIR, CROPS_DIR, excluded_depositors=val_depositors,
                                    recovered_dir=recovered_dir)

    n_real_crops = sum(len(v) for v in real_by_dep.values())
    print(f"procedural pool: {len(proc_pool)}   "
          f"real: {n_real_crops} crops / {len(real_deps)} depositors   "
          f"backgrounds: {len(backgrounds)}   workers: {workers}")

    os.makedirs(f"{out_dir}/images", exist_ok=True)
    os.makedirs(f"{out_dir}/labels", exist_ok=True)
    if save_stamps:
        os.makedirs(f"{out_dir}/stamps_used", exist_ok=True)

    # one independent, decorrelated RNG stream per page instead of a single
    # shared sequential one - required once pages can run out of order
    # across worker processes
    tasks = list(enumerate(np.random.SeedSequence(seed).spawn(n)))

    total_boxes = 0
    source_counts = collections.Counter()
    shortfall_pages = 0
    shortfall_total = 0
    manifest_lines = []

    if workers <= 1:
        _init_worker(cfg, real_by_dep, real_deps, proc_pool, backgrounds, out_dir, save_stamps)
        page_results = map(_render_page, tasks)
        for n_boxes, counts, shortfall, lines in page_results:
            total_boxes += n_boxes
            source_counts.update(counts)
            if shortfall:
                shortfall_pages += 1
                shortfall_total += shortfall
            manifest_lines.extend(lines)
    else:
        with mp.Pool(
            workers, initializer=_init_worker,
            initargs=(cfg, real_by_dep, real_deps, proc_pool, backgrounds, out_dir,
                      save_stamps),
        ) as pool:
            for n_boxes, counts, shortfall, lines in pool.imap_unordered(
                    _render_page, tasks, chunksize=8):
                total_boxes += n_boxes
                source_counts.update(counts)
                if shortfall:
                    shortfall_pages += 1
                    shortfall_total += shortfall
                manifest_lines.extend(lines)

    if save_stamps:
        with open(f"{out_dir}/stamps_used_manifest.csv", "w") as f:
            f.write("page,stamp_index,source,source_id,ink,x0,y0,x1,y1,crop_file\n")
            f.writelines(manifest_lines)
        print(f"saved {total_boxes} used-stamp crops to {out_dir}/stamps_used/  "
              f"(manifest: {out_dir}/stamps_used_manifest.csv)")

    print(f"\nwrote {n} images, {total_boxes} boxes "
          f"({total_boxes/max(n,1):.1f} stamps/page) to {out_dir}/")
    total_src = sum(source_counts.values())
    for src, c in source_counts.most_common():
        print(f"  {src:10s}: {c:5d}  ({c/max(total_src,1):.1%})   "
              f"[target procedural={cfg['procedural_frac']:.0%}]")
    if shortfall_pages:
        print(f"note: {shortfall_pages}/{n} pages placed fewer stamps than requested "
              f"(avg shortfall {shortfall_total/shortfall_pages:.1f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default=str(REPO_ROOT / "synth_data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-size", type=int, default=None,
                    help="override CONFIG['pool_size']")
    ap.add_argument("--no-save-stamps", action="store_true",
                    help="skip saving individual used-stamp crops + manifest")
    ap.add_argument("--split-result", default=SPLIT_RESULT_PATH,
                    help="path to split_result.json from splitbybank.py")
    ap.add_argument("--recovered-dir", default=RECOVERED_DIR,
                    help="dir of inpainted backgrounds from cleanbackgrounds.py "
                         "(must have been built with the same --split-result); "
                         "pass '' to disable")
    ap.add_argument("--workers", type=int, default=1,
                    help="worker processes for page rendering; 1 = sequential "
                         "(default), os.cpu_count() is a reasonable upper bound")
    a = ap.parse_args()
    if a.pool_size:
        CONFIG["pool_size"] = a.pool_size
    generate(a.n, a.out, CONFIG, a.seed, save_stamps=not a.no_save_stamps,
              split_result_path=a.split_result, recovered_dir=a.recovered_dir or None,
              workers=a.workers)