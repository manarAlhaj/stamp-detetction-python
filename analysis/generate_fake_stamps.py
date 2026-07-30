# python3 generate_fake_stamps.py --n 12 --out proc_out
#"/home/manaralhajyousef/Desktop/stamp-detetction-python/analysis/fake_stamps"


import argparse
import calendar
import glob
import os
import random
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PAPER_TONE = 246          # matches the median paper brightness in real scans
INK_RANGE = (22, 40)      # darker = fresher ink; randomized per stamp below

EDGE_WEAR_PROB = 0.45     # chance a given border/ring gets ragged, missing chunks

LINE_INK_JITTER_PROB = 0.5    # chance a single line/word gets its own ink density
LINE_INK_JITTER = (-6, 55)    # (min, max) offset added to the stamp's base ink;
                               # skewed lighter since uneven stamp pressure tends
                               # to under-ink a word more often than over-ink it

SUPERSAMPLE = 4

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",

]
FONT_PATHS = [path for path in _FONT_CANDIDATES if os.path.isfile(path)]
if not FONT_PATHS:
    raise RuntimeError(
        "No Arabic-capable font was found. Install DejaVu Sans/Serif or "
        "update _FONT_CANDIDATES with an available Arabic font."
    )


def _cmap(path):
    from fontTools.ttLib import TTFont
    cm = set()
    for t in TTFont(path, fontNumber=0)["cmap"].tables:
        cm |= set(t.cmap.keys())
    return cm


CMAPS = {p: _cmap(p) for p in FONT_PATHS}


# entity names stay paired so a bilingual stamp never mixes two companies.
ENTITY_RECORDS = [
    ("فلسطين للتأمين", "Palestine Insurance"),
    ("شركة التأمين الوطنية", "National Insurance Company"),
    ("شركة ترست العالمية للتأمين", "Trust International Insurance Co."),
    ("شركة القدس للتأمين", "Jerusalem Insurance"),
    ("الشركة الفلسطينية للتأجير التمويلي", "Palestine Leasing Company"),
    ("شركة دار الشفاء", "PharmaCare Ltd."),
    ("شركة الكرمل للخدمات التسويقية", "Al Karmel Marketing Services"),
    ("شركة باماك لخدمات الدفع الإلكتروني", "BAMAK Payment Services"),
    ("شركة بالكو للاستيراد والتوزيع", "PALCO Import & Distribution"),
    ("بلدية نابلس", "Nablus Municipality"),
    ("بلدية سلفيت", "Salfit Municipality"),
    ("شركة الهدايا ومستحضرات التجميل", "AL THAHER GIFTS & COSMETICS CO."),
    ("مؤسسة الأرض السعيدة", "Merry Land"),
    ("صيدلية ماكسي كير", "MAXICARE"),
    ("شركة قطع السيارات", "ALL PARTS CO."),
    ("بنك فلسطين", "Bank of Palestine"),
    ("بنك الأردن", "Bank of Jordan"),
    ("البنك الأهلي الأردني", "Jordan Ahli Bank"),
    ("بنك السلام", "Al Salam Bank"),
    ("البنك الخليجي", "Khaleeji Bank"),
    ("بنك إتش إس بي سي الشرق الأوسط", "HSBC Bank Middle East Ltd."),
]

ENTITY = [arabic for arabic, _ in ENTITY_RECORDS]
LATIN = [latin for _, latin in ENTITY_RECORDS]

BANK_AR = [
    "بنك فلسطين",
    "بنك الأردن",
    "البنك الأهلي الأردني",
    "بنك السلام",
    "البنك الخليجي",
    "بنك إتش إس بي سي الشرق الأوسط",
]

BANK_LATIN = [
    "Bank of Palestine",
    "Bank of Jordan",
    "Jordan Ahli Bank",
    "Al Salam Bank",
    "HSBC Bank Middle East Ltd.",
    "Khaleeji Bank",
]

BRANCH_AR = [
    "فرع رام الله",
    "فرع نابلس",
    "فرع الخليل",
    "فرع بيت لحم",
    "فرع طولكرم",
    "فرع جنين",
    "المكتب الرئيسي - رام الله",
    "الإدارة العامة",
]

BRANCH_LATIN = [
    "Ramallah Branch",
    "Nablus Branch",
    "Hebron Branch",
    "Bethlehem Branch",
    "Tulkarem Branch",
    "Jenin Branch",
    "Head Office - Ramallah",
    "Zinj Branch",
    "Seef Branch",
]


BRANCH = BRANCH_AR

DEPARTMENT = [
    "قسم المحاسبة",
    "الدائرة المالية",
    "الإدارة المالية",
    "الإدارة العامة",
    "المكتب الرئيسي",
]

PURPOSE_AR = [
    "للإيداع فقط",
    "لإيداع الشيكات",
    "تظهير لغايات التحصيل",
    "تظهير لغاية التحصيل",
    "فقط لأمر البنك الأهلي الأردني",
    "لأمر البنك الأهلي الأردني",
]

PURPOSE_LATIN = [
    "FOR DEPOSIT ONLY",
    "FOR COLLECTION",
    "CLEARING",
    "Presentment",
    "CLEARING CHEQUES",
    "RETURNED / REJECTED",
]

PURPOSE = PURPOSE_AR

ROLE = [
    "أمين الصندوق",
    "التوقيع",
    "المحاسب",
    "المدير المالي",
]

ACTIVITY = [
    "للتأمين",
    "للتأجير التمويلي",
    "للاستيراد والتوزيع",
    "للخدمات التسويقية",
    "لخدمات الدفع الإلكتروني",
    "لمواد البناء",
    "للإنتاج والتسويق",
    "لاستيراد وتسويق الأدوية",
]

NUMBER_KINDS = (
    "short_id",
    "registration",
    "phone",
    "branch_dash",
    "branch_slash",
    "account_slash",
    "account_dash",
    "nis",
)


#we get some number from 0-9 with the length provided, as a string to keep any trailing zeros bs
def _digits(rng, length):
    return "".join(rng.choice("0123456789") for _ in range(length))


def random_identifier(rng, kind=None):
    kind = kind or rng.choice(NUMBER_KINDS)
    if kind == "short_id":
        return _digits(rng, rng.choice([5, 6]))
    if kind == "registration":
        return _digits(rng, 7)
    if kind == "phone":
        return "05" + _digits(rng, 8)
    if kind == "branch_dash":
        return f"{_digits(rng, 5)}-{_digits(rng, 1)}"
    if kind == "branch_slash":
        return f"{_digits(rng, 5)}/{_digits(rng, 1)}"
    if kind == "account_slash":
        return f"{_digits(rng, 7)}/{_digits(rng, 3)}"
    if kind == "account_dash":
        return f"{_digits(rng, 6)}-{_digits(rng, 3)}"
    if kind == "nis":
        return f"NIS {_digits(rng, 10)}"
    raise ValueError(f"unknown identifier kind: {kind}")


def random_number_line(rng):
    kind = rng.choice(NUMBER_KINDS)
    value = random_identifier(rng, kind)
    if kind in {"account_slash", "account_dash"}:
        return f"{rng.choice(['حساب رقم', 'رقم الحساب'])} {value}"
    if kind == "phone":
        return rng.choice([f"هاتف: {value}", f"Tel.: {value}"])
    if kind in {"short_id", "registration"} and rng.random() < 0.45:
        return f"{rng.choice(['رقم التسجيل', 'رقم المنشأة'])} {value}"
    return value


def random_stamp_date(rng, min_year=2020, max_year=2026, style=None):
    year = rng.randint(min_year, max_year)
    month = rng.randint(1, 12)
    day = rng.randint(1, calendar.monthrange(year, month)[1])
    style = style or rng.choice(["numeric_dash", "numeric_slash", "latin"])
    if style == "numeric_dash":
        return f"{day:02d}-{month:02d}-{year:04d}"
    if style == "numeric_slash":
        return f"{day:02d}/{month:02d}/{year:04d}"
    if style == "latin":
        months = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                  "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
        return f"{day:02d} {months[month - 1]} {year:04d}"
    raise ValueError(f"unknown date style: {style}")


def ar(text):
    return text


def _font(rng, size, text=None):

    if text is None:
        return ImageFont.truetype(rng.choice(FONT_PATHS), size)
    need = {ord(c) for c in text if c != " "}
    ok = [p for p in FONT_PATHS if need <= CMAPS[p]]
    if ok:
        return ImageFont.truetype(rng.choice(ok), size)
    best = max(FONT_PATHS, key=lambda p: len(need & CMAPS[p]))
    return ImageFont.truetype(best, size)



def _fit(draw, text, max_w, max_h, rng, start):

    size = max(int(start), 7)
    while size > 7:
        f = _font(rng, size, text)
        #box returns tuple of 4 values left top right bottom
        box = draw.textbbox((0, 0), text, font=f) #hoon we draw a pixel exact box around the text to get its h/w
        if (box[2] - box[0]) <= max_w and (box[3] - box[1]) <= max_h:
            return f
        size -= 1
    return _font(rng, 7, text)


def _draw_arc_text(draw, text, cx, cy, radius, font, rng, ink, flip=False):

    shaped = ar(text)
    probe = ImageDraw.Draw(Image.new("L", (8, 8)))
    b = probe.textbbox((0, 0), shaped, font=font)
    sw, sh = max(b[2] - b[0], 1), max(b[3] - b[1], 1)

    # Strip background is PAPER_TONE, not 0: outside the glyphs (and outside
    # the valid warp region below) this must read as "no ink here", and with
    # a paper-toned canvas "no ink" means paper-toned, not black.
    strip = Image.new("L", (sw + 4, sh + 4), PAPER_TONE)
    ImageDraw.Draw(strip).text((2 - b[0], 2 - b[1]), shaped, fill=ink, font=font)
    src = np.asarray(strip, np.float32)
    sh4, sw4 = src.shape

    canvas = draw._image
    W, H = canvas.size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    r = np.hypot(dx, dy)

    span = min(sw4 / max(radius, 1.0), np.pi * 1.5)

    if not flip:                      # top arc, letters pointing outward
        phi = np.arctan2(dx, -dy)     # 0 at top, positive to the right
        v = (radius + sh4 / 2.0) - r
    else:                             # bottom arc, letters pointing inward
        phi = np.arctan2(-dx, dy)     # 0 at bottom, positive to the left
        v = r - (radius - sh4 / 2.0)

    u = (phi + span / 2.0) / span * sw4

    inside = (u >= 0) & (u < sw4) & (v >= 0) & (v < sh4)

    warped = np.full((H, W), PAPER_TONE, dtype=np.float32)
    valid = inside & (u < sw4 - 1) & (v < sh4 - 1)
    u0 = np.floor(u[valid]).astype(np.int32)
    v0 = np.floor(v[valid]).astype(np.int32)
    du = u[valid] - u0
    dv = v[valid] - v0
    warped[valid] = (
        src[v0, u0] * (1.0 - du) * (1.0 - dv)
        + src[v0, u0 + 1] * du * (1.0 - dv)
        + src[v0 + 1, u0] * (1.0 - du) * dv
        + src[v0 + 1, u0 + 1] * du * dv
    )


    base = np.asarray(canvas, np.float32)
    canvas.paste(Image.fromarray(np.minimum(base, warped).astype(np.uint8)), (0, 0))


def _line_ink(rng, base_ink, prob=LINE_INK_JITTER_PROB, spread=LINE_INK_JITTER):
    """Pick an ink value for one line/word of text.

    Real stamps rarely deposit ink evenly across the whole imprint, so each
    line gets a chance to come out lighter (occasionally darker) than the
    stamp's base density instead of every line sharing one flat `ink` value.
    """
    if rng.random() < prob:
        return int(np.clip(base_ink + rng.randint(*spread), 10, 140))
    return base_ink


def _rect_perimeter(x0, y0, x1, y1, step=6):
    pts = []

    def seg(p0, p1):
        length = max(np.hypot(p1[0] - p0[0], p1[1] - p0[1]), 1.0)
        n = max(int(length / step), 2)
        for t in np.linspace(0, 1, n, endpoint=False):
            pts.append((p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t))

    seg((x0, y0), (x1, y0))
    seg((x1, y0), (x1, y1))
    seg((x1, y1), (x0, y1))
    seg((x0, y1), (x0, y0))
    return pts


def _ellipse_perimeter(cx, cy, rx, ry, step=6):
    # Ramanujan's approximation is plenty accurate for spacing purposes here.
    circumference = np.pi * (3 * (rx + ry) - np.sqrt((3 * rx + ry) * (rx + 3 * ry)))
    n = max(int(circumference / step), 24)
    return [
        (cx + rx * np.cos(t), cy + ry * np.sin(t))
        for t in np.linspace(0, 2 * np.pi, n, endpoint=False)
    ]


def _worn_line_loop(d, points, ink, width, rng, n_gaps=(2, 5), gap_frac=(0.03, 0.14)):
    """Draw a closed polyline through `points`, skipping random arcs of it
    to mimic a worn rubber stamp that no longer makes full edge contact."""
    n = len(points)
    keep = np.ones(n, dtype=bool)
    for _ in range(rng.randint(*n_gaps)):
        glen = max(int(n * rng.uniform(*gap_frac)), 2)
        start = rng.randrange(n)
        for k in range(glen):
            keep[(start + k) % n] = False

    i = 0
    while i < n:
        if not keep[i]:
            i += 1
            continue
        j = i
        run = [points[i]]
        while j + 1 < n and keep[j + 1]:
            j += 1
            run.append(points[j])
        if len(run) >= 2:
            d.line(run, fill=ink, width=width, joint="curve")
        else:
            r = width / 2.0
            px, py = run[0]
            d.ellipse([px - r, py - r, px + r, py + r], fill=ink)
        i = j + 1


def _maybe_worn_rectangle(d, box, ink, width, rng, wear_prob=EDGE_WEAR_PROB):
    x0, y0, x1, y1 = box
    if rng.random() < wear_prob:
        _worn_line_loop(d, _rect_perimeter(x0, y0, x1, y1), ink, width, rng)
    else:
        d.rectangle(box, outline=ink, width=width)


def _maybe_worn_ellipse(d, box, ink, width, rng, wear_prob=EDGE_WEAR_PROB):
    x0, y0, x1, y1 = box
    if rng.random() < wear_prob:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
        _worn_line_loop(d, _ellipse_perimeter(cx, cy, rx, ry), ink, width, rng)
    else:
        d.ellipse(box, outline=ink, width=width)


def make_rect_stamp(rng, S=320):
    ink = rng.randint(*INK_RANGE)
    img = Image.new("L", (S, S), PAPER_TONE)
    d = ImageDraw.Draw(img)
    w, h = int(S * rng.uniform(0.72, 0.94)), int(S * rng.uniform(0.36, 0.72))
    x0, y0 = (S - w) // 2, (S - h) // 2
    bw = rng.choice([2, 3, 4])
    _maybe_worn_rectangle(d, [x0, y0, x0 + w, y0 + h], ink, bw, rng)
    if rng.random() < 0.35:                        # double border
        g = rng.randint(4, 8)
        _maybe_worn_rectangle(d, [x0 + g, y0 + g, x0 + w - g, y0 + h - g], ink, 1,
                               rng, wear_prob=EDGE_WEAR_PROB * 0.7)

    entity_ar, entity_latin = rng.choice(ENTITY_RECORDS)
    lines = [ar(entity_ar)]
    optional_lines = []
    if rng.random() < 0.52:
        optional_lines.append(entity_latin)
    if rng.random() < 0.50:
        optional_lines.append(ar(rng.choice(DEPARTMENT)))
    if rng.random() < 0.62:
        optional_lines.append(ar(rng.choice(PURPOSE_AR)))
    if rng.random() < 0.16:
        optional_lines.append(rng.choice(PURPOSE_LATIN))
    if rng.random() < 0.38:
        optional_lines.append(ar(rng.choice(BRANCH_AR)))
    if rng.random() < 0.18:
        optional_lines.append(ar(rng.choice(ROLE)))
    if rng.random() < 0.24:
        optional_lines.append(ar(rng.choice(ACTIVITY)))
    if rng.random() < 0.75:
        optional_lines.append(ar(random_number_line(rng)))
    if rng.random() < 0.18:
        optional_lines.append(random_stamp_date(rng))
    if not optional_lines:
        optional_lines.append(ar(rng.choice(PURPOSE_AR)))
    rng.shuffle(optional_lines)
    lines.extend(optional_lines[:rng.randint(1, min(4, len(optional_lines)))])

    pad = bw + 6
    fs = max(int(h / (len(lines) + 1.2)), 10)
    fonts = [_fit(d, ln, w - 2 * pad, h / len(lines) - 2, rng, fs) for ln in lines]
    step = max(f.size for f in fonts) * 1.24
    cy = y0 + h / 2 - (len(lines) - 1) * step / 2
    for ln, f in zip(lines, fonts):
        d.text((S / 2, cy), ln, fill=_line_ink(rng, ink), font=f, anchor="mm")
        cy += step
    return img


def make_round_stamp(rng, S=320):
    ink = rng.randint(*INK_RANGE)
    img = Image.new("L", (S, S), PAPER_TONE)
    d = ImageDraw.Draw(img)
    cx = cy = S / 2
    R = S * rng.uniform(0.40, 0.47)
    _maybe_worn_ellipse(d, [cx - R, cy - R, cx + R, cy + R], ink,
                         rng.choice([2, 3, 4]), rng)
    R2 = R * rng.uniform(0.74, 0.86)
    _maybe_worn_ellipse(d, [cx - R2, cy - R2, cx + R2, cy + R2], ink, 2, rng,
                         wear_prob=EDGE_WEAR_PROB * 0.7)

    entity_ar, entity_latin = rng.choice(ENTITY_RECORDS)
    branch_ar = rng.choice(BRANCH_AR)
    f_out = _font(rng, max(int(R * 0.16), 9), ar(entity_ar + branch_ar))
    _draw_arc_text(d, entity_ar, cx, cy, (R + R2) / 2, f_out, rng, _line_ink(rng, ink))
    if rng.random() < 0.7:
        _draw_arc_text(d, branch_ar, cx, cy, (R + R2) / 2, f_out,
                       rng, _line_ink(rng, ink), flip=True)

    inner = [ar(branch_ar)]
    if rng.random() < 0.34:
        inner[0] = entity_latin
    if rng.random() < 0.74:
        inner.append(random_identifier(
            rng, rng.choice(["short_id", "branch_dash", "branch_slash", "nis"])
        ))
    elif rng.random() < 0.55:
        inner.append(ar(rng.choice(DEPARTMENT)))
    inner_w = R2 * 1.55                      # chord that fits inside the ring
    fonts = [_fit(d, ln, inner_w, R2, rng, max(int(R * 0.18), 9)) for ln in inner]
    step = max(f.size for f in fonts) * 1.2
    yy = cy - (len(inner) - 1) * step / 2
    for ln, f in zip(inner, fonts):
        d.text((cx, yy), ln, fill=_line_ink(rng, ink), font=f, anchor="mm")
        yy += step

    if rng.random() < 0.3:                        # star separators
        for k in range(rng.choice([2, 4])):
            a = np.pi / 2 + k * 2 * np.pi / rng.choice([2, 4])
            px, py = cx + (R + R2) / 2 * np.cos(a), cy + (R + R2) / 2 * np.sin(a)
            d.text((px, py), "*", fill=ink, font=f_out, anchor="mm")
    return img


def make_oval_stamp(rng, S=320):
    ink = rng.randint(*INK_RANGE)
    img = Image.new("L", (S, S), PAPER_TONE)
    d = ImageDraw.Draw(img)
    w, h = S * rng.uniform(0.80, 0.94), S * rng.uniform(0.36, 0.56)
    x0, y0 = (S - w) / 2, (S - h) / 2
    _maybe_worn_ellipse(d, [x0, y0, x0 + w, y0 + h], ink, rng.choice([2, 3]), rng)
    lines = [
        ar(rng.choice(PURPOSE_AR + PURPOSE_LATIN)),
        random_stamp_date(rng),
    ]
    if rng.random() < 0.58:
        lines.append(ar(rng.choice(BRANCH_AR + BRANCH_LATIN)))
    if rng.random() < 0.35:
        lines.append(random_identifier(
            rng, rng.choice(["short_id", "branch_dash", "account_slash"])
        ))
    fonts = [_fit(d, ln, w * 0.72, h / len(lines) - 2, rng, max(int(h * 0.20), 10))
             for ln in lines]
    step = max(f.size for f in fonts) * 1.24
    cy = S / 2 - (len(lines) - 1) * step / 2
    for ln, f in zip(lines, fonts):
        d.text((S / 2, cy), ln, fill=_line_ink(rng, ink), font=f, anchor="mm")
        cy += step
    return img


MAKERS = [make_rect_stamp, make_rect_stamp, make_round_stamp, make_oval_stamp]


def make_template(rng, S=320, margin=6):
   
    img = rng.choice(MAKERS)(rng, S * SUPERSAMPLE)
    img = img.resize((S, round(img.height / SUPERSAMPLE)), Image.LANCZOS)


    if rng.random() < 0.42:

        img = img.rotate(90, expand=True, fillcolor=PAPER_TONE)

    g = np.asarray(img, np.float32)
    deviation = PAPER_TONE - g                 # positive where ink was drawn
    ys, xs = np.nonzero(deviation > 8)
    if len(xs) < 20:
        return None

    y0 = max(ys.min() - margin, 0)
    x0 = max(xs.min() - margin, 0)
    y1 = min(ys.max() + margin + 1, g.shape[0])
    x1 = min(xs.max() + margin + 1, g.shape[1])
    return g[y0:y1, x0:x1].astype(np.uint8)


def _next_start_index(out_dir):
    existing = glob.glob(os.path.join(out_dir, "proc_*.png"))
    if not existing:
        return 0
    indices = []
    for path in existing:
        m = re.search(r"proc_(\d+)\.png$", os.path.basename(path))
        if m:
            indices.append(int(m.group(1)))
    return max(indices) + 1 if indices else 0


def generate(n, out_dir, seed=0):
    rng = random.Random(seed)
    rng.randint, rng.uniform, rng.choice, rng.random  # noqa
    os.makedirs(out_dir, exist_ok=True)
    start = _next_start_index(out_dir)
    made = 0
    while made < n:
        img = make_template(rng)          # already a finished uint8 image now
        if img is None:
            continue
        Image.fromarray(img).save(f"{out_dir}/proc_{start + made:05d}.png")
        made += 1
    print(f"wrote {made} procedural templates to {out_dir}/ "
          f"(indices {start:05d}-{start + made - 1:05d})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="proc_out")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate(a.n, a.out, a.seed)