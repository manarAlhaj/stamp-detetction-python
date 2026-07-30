
"""
| Part    | Meaning                           |
| ------- | --------------------------------- |
| `^`     | Start of the filename             |
| `(.*)`  | Capture any characters as group 1 |
| `_back` | Match the literal text `_back`    |
| `\.`    | Match a literal period            |
| `png`   | Match `png`                       |
| `$`     | End of the filename               |

"""

import os
import argparse
import re
import collections


stamped_dir = "/home/manaralhajyousef/Desktop/stamp-detetction-python/images_with_stamps"
unstamped_dir = "/home/manaralhajyousef/Desktop/stamp-detetction-python/images_no_stamps"


def page_stem(file):
    m = re.match(r"^(.*)_back\.png$", file)
    if m:
        return m.group(1)
    else:
        os.path.splitext(file)[0]

def bank(stem):
    return stem.split("_")[0]



def scan(folder):
    by_dep = collections.defaultdict(list)
    if not os.path.isdir(folder):
        print(f"warning: {folder} does not exist, skipping")
        return by_dep
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        dep = bank(page_stem(f))
        by_dep[dep].append(f)
    return by_dep
 
 
def main(stamped_dir, unstamped_dir):
    stamped = scan(stamped_dir)
    unstamped = scan(unstamped_dir)
 
    all_deps = sorted(
        set(stamped) | set(unstamped),
        key=lambda d: -(len(stamped.get(d, [])) + len(unstamped.get(d, []))),
    )
 
    total_stamped = sum(len(v) for v in stamped.values())
    total_unstamped = sum(len(v) for v in unstamped.values())
    total = total_stamped + total_unstamped
 
    print(f"{'depositor':16s} {'stamped':>8s} {'unstamped':>10s} {'total':>7s} {'% of all':>9s} ")
    print("-" * 70)
    for dep in all_deps:
        s = len(stamped.get(dep, []))
        u = len(unstamped.get(dep, []))
        t = s + u
        if s and u:
            role = "both"
        elif s:
            role = "stamped-only"
        else:
            role = "unstamped-only"
        pct = t / total if total else 0
        print(f"{dep:16s} {s:8d} {u:10d} {t:7d} {pct:8.1%}  {role}")
 
    n_stamped_dep = sum(1 for d in all_deps if stamped.get(d))
    n_unstamped_only_dep = sum(1 for d in all_deps if not stamped.get(d))
 
    print()
    print(f"depositors total          : {len(all_deps)}")
    print(f"  with a stamped example  : {n_stamped_dep}")
    print(f"  unstamped only          : {n_unstamped_only_dep}")
    print(f"images total              : {total}  ({total_stamped} stamped / {total_unstamped} unstamped)")
    if all_deps:
        top_dep = all_deps[0]
        top_total = len(stamped.get(top_dep, [])) + len(unstamped.get(top_dep, []))
        print(f"largest depositor         : {top_dep} - {top_total} images "
              f"({top_total/total:.0%} of all {total})")
 
    return stamped, unstamped
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamped", default=stamped_dir)
    ap.add_argument("--unstamped", default=unstamped_dir)
    a = ap.parse_args()
    main(a.stamped, a.unstamped)


