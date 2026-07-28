from pathlib import Path
from collections import defaultdict
import random
import shutil

SOURCE = Path("/home/ps-ai/Desktop/manar/yolofiles")

IMAGES_DIR = SOURCE / "images"
LABELS_DIR = SOURCE / "labels"

OUTPUT = Path("/home/ps-ai/Desktop/manar/finaldataset")

TRAIN_RATIO = 0.80
SEED = 42


train_images = OUTPUT / "images" / "train"
val_images = OUTPUT / "images" / "val"

train_labels = OUTPUT / "labels" / "train"
val_labels = OUTPUT / "labels" / "val"

for folder in [
    train_images,
    val_images,
    train_labels,
    val_labels
]:
    folder.mkdir(parents=True, exist_ok=True)


image_extensions = {".png", ".jpg", ".jpeg", ".webp"}

pairs = []

for image in IMAGES_DIR.iterdir():

    if image.suffix.lower() not in image_extensions:
        continue

    label = LABELS_DIR / f"{image.stem}.txt"

    if not label.exists():
        print(f"no label for {image.name}")
        continue

    pairs.append((image, label))

print(f"Valid image/label pairs: {len(pairs)}")


def get_group(filename):

    stem = Path(filename).stem
    if "_" in stem:
        return stem.split("_")[0]
    return stem


groups = defaultdict(list)

for image, label in pairs:
    group = get_group(image.name)
    groups[group].append((image, label))

print(f"Number of groups: {len(groups)}")

for group, items in sorted(
    groups.items(),
    key=lambda x: len(x[1]),
    reverse=True
):
    print(f"{group:20s} -> {len(items)} images")


random.seed(SEED)

group_items = list(groups.items())

random.shuffle(group_items)

group_items.sort(
    key=lambda x: len(x[1]),
    reverse=True
)

total_images = len(pairs)

target_train = round(total_images * TRAIN_RATIO)

train_pairs = []
val_pairs = []

train_groups = []
val_groups = []

for group_name, group_pairs in group_items:

    group_size = len(group_pairs)

    current_train = len(train_pairs)

    distance_if_train = abs(
        target_train - (current_train + group_size)
    )

    distance_if_val = abs(
        target_train - current_train
    )

    if distance_if_train <= distance_if_val:
        train_pairs.extend(group_pairs)
        train_groups.append(group_name)
    else:
        val_pairs.extend(group_pairs)
        val_groups.append(group_name)



def copy_pairs(pairs, images_destination, labels_destination):

    for image, label in pairs:

        shutil.copy2(
            image,
            images_destination / image.name
        )

        shutil.copy2(
            label,
            labels_destination / label.name
        )


copy_pairs(
    train_pairs,
    train_images,
    train_labels
)

copy_pairs(
    val_pairs,
    val_images,
    val_labels
)


print("\n=========================")

print(
    f"Train: {len(train_pairs)} "
    f"({len(train_pairs) / total_images:.1%})"
)

print(
    f"Val:   {len(val_pairs)} "
    f"({len(val_pairs) / total_images:.1%})"
)

print(f"\nTrain groups: {len(train_groups)}")
print(f"Val groups:   {len(val_groups)}")

print("\nValidation groups:")

for group in sorted(val_groups):
    print(
        f"{group:20s} "
        f"{len(groups[group])} images"
    )