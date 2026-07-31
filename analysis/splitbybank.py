import re
from pathlib import Path

from banksanalysis import scan
import json

valfrac = 0.1
excluded = {"TNBCPS20004", "MANUALALSALAM001"}

REPO_ROOT = Path(__file__).resolve().parent.parent
unst_dir = str(REPO_ROOT / "images_no_stamps")
st_dir = str(REPO_ROOT / "images_with_stamps")

stamped = scan(st_dir)
unstamped = scan(unst_dir)


total_images = (
    sum(len(v) for v in stamped.values())
    + sum(len(v) for v in unstamped.values())
)

val_image_count = round(valfrac * total_images)


all_depositers = set(stamped) | set(unstamped)
banks_for_val = all_depositers - excluded


banks_with_stamped_or_both = [s for s in banks_for_val if stamped.get(s)]
banks_no_stamps = [s for s in banks_for_val if not stamped.get(s)]


def sort_key(d):
    total = len(stamped.get(d, [])) + len(unstamped.get(d, []))
    return (total, d)  

ranked = sorted(banks_with_stamped_or_both, key=sort_key)


# greedy
def select_val(ranked_banks, val_images_cnt, stamped, unstamped):
    selected = []
    selected_images = 0
    for pick in ranked_banks:
        if selected_images >= val_images_cnt:
            break
        total = len(stamped.get(pick, [])) + len(unstamped.get(pick, []))
        selected.append(pick)
        selected_images += total
    return selected, selected_images


def select_background_slice(banks_no_stamps, unstamped, n=4):
    ranked_bg = sorted(
        banks_no_stamps,
        key=lambda d: (-len(unstamped.get(d, [])), d)
    )
    picks = ranked_bg[:n]
    total_bg_images = sum(len(unstamped[d]) for d in picks)
    return picks, total_bg_images


val_depositors, val_images = select_val(ranked, val_image_count, stamped, unstamped)
bg_depositors, bg_images = select_background_slice(banks_no_stamps, unstamped, n=4)

val_depositors = val_depositors + bg_depositors
val_images = val_images + bg_images
train_depositors = sorted(all_depositers - set(val_depositors))

output = {
    "val_depositors": val_depositors,
    "val_images": val_images,
    "train_depositors": train_depositors,
}

with open("split_result.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"saved {len(val_depositors)} val depositors ({val_images} images) to split_result.json")