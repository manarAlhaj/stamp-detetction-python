
#this is for crop numbers analysis

import os
import re

PAGES_DIR = "/home/manaralhajyousef/Desktop/stamp-detetction-python/data27jul/train"
CROPS_DIR = "/home/manaralhajyousef/Desktop/stamp-detetction-python/stamp_crops_from_labels"


def page_stem(fname):
   #'TNBCPS20004_240625Q001191_back.png' = 'TNBCPS20004_240625Q001191'
    m = re.match(r"^(.*)_back\.png$", fname)
    if m:
        return m.group(1)
    return fname[:-4]   # handle "image (1).png" files


def crop_stem(fname):
    #TNBCPS20004_240625Q001191_back_stamp_1.png' = same stem as above
    m = re.match(r"^(.*)_back_stamp_\d+\.png$", fname)
    if m:
        return m.group(1)
    m = re.match(r"^(.*)_stamp_\d+\.png$", fname)      # "image (1)_stamp_1.png"
    return m.group(1)


# --- load filenames ---------------------------------------------------
pages = sorted(f for f in os.listdir(PAGES_DIR) if f.endswith(".png"))
crops = sorted(f for f in os.listdir(CROPS_DIR) if f.endswith(".png"))

# --- how many crops does each page stem have? --------------------------
crops_per_stem = {}
for c in crops:
    stem = crop_stem(c)
    crops_per_stem[stem] = crops_per_stem.get(stem, 0) + 1

stamped_stems = set(crops_per_stem)

# --- classify each page --------------------------------------------------
stamped = [p for p in pages if page_stem(p) in stamped_stems]
unstamped = [p for p in pages if page_stem(p) not in stamped_stems]

# --- how many stamped pages have 1, 2, 3 stamps? --------------------------
n_with_1 = sum(1 for n in crops_per_stem.values() if n == 1)
n_with_2 = sum(1 for n in crops_per_stem.values() if n == 2)
n_with_3 = sum(1 for n in crops_per_stem.values() if n == 3)


print(f"pages total       : {len(pages)}")
print(f"  stamped         : {len(stamped)}")
print(f"  unstamped       : {len(unstamped)}")
print(f"  split           : {len(stamped)/len(pages):.0%} / {len(unstamped)/len(pages):.0%}")
print(f"crops total       : {len(crops)}")
print(f"  across          : {len(crops_per_stem)} stamped pages")
print(f"  1 stamp         : {n_with_1} pages")
print(f"  2 stamps        : {n_with_2} pages")
print(f"  3 stamps        : {n_with_3} pages")