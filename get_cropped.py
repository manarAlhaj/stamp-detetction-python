from pathlib import Path
from PIL import Image
import re
import shutil

IMAGES_DIR = Path("/home/ps-ai/Desktop/manar/yolofiles/images")
LABELS_DIR = Path("/home/ps-ai/Desktop/manar/yolofiles/labels")
OUTPUT_DIR = Path("/home/ps-ai/Desktop/manar/stamp_crops_from_labels")

WITH_STAMPS_DIR = Path("/home/ps-ai/Desktop/manar/images_with_stamps")
NO_STAMPS_DIR = Path("/home/ps-ai/Desktop/manar/images_no_stamps")
NO_LABEL_DIR = Path("/home/ps-ai/Desktop/manar/images_no_label")


###################################################################


for d in (OUTPUT_DIR, WITH_STAMPS_DIR, NO_STAMPS_DIR, NO_LABEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

image_extensions = {".png", ".jpg", ".jpeg", ".webp"}


def find_label(image_path, labels_dir):
    normal_label = labels_dir / f"{image_path.stem}.txt"
    if normal_label.exists():
        return normal_label

    # handle images named like: image (1).png
    match = re.fullmatch(r"image \((\d+)\)", image_path.stem)
    if match:
        number = match.group(1)
        # image (9).png can match: d1f1e577-image_9.txt
        matches = list(labels_dir.glob(f"*-image_{number}.txt"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f" multiple possible labels found for {image_path.name}:")
            for label in matches:
                print("   ", label.name)
    return None


counts = {"with": 0, "without": 0, "missing": 0, "crops": 0}

for image_path in sorted(IMAGES_DIR.iterdir()):
    if image_path.suffix.lower() not in image_extensions:
        continue

    label_path = find_label(image_path, LABELS_DIR)

    if label_path is None:
        print(f"No label found for: {image_path.name}")
        shutil.copy2(image_path, NO_LABEL_DIR / image_path.name)
        counts["missing"] += 1
        continue

    lines = [ln for ln in label_path.read_text().splitlines() if ln.strip()]

    if not lines:
        shutil.copy2(image_path, NO_STAMPS_DIR / image_path.name)
        counts["without"] += 1
        print(f"No stamps (negative): {image_path.name}")
        continue

    shutil.copy2(image_path, WITH_STAMPS_DIR / image_path.name)
    counts["with"] += 1

    image = Image.open(image_path)
    img_w, img_h = image.size

    for i, line in enumerate(lines):
        class_id, x_center, y_center, width, height = map(float, line.split())

        x_center *= img_w
        y_center *= img_h
        width *= img_w
        height *= img_h

        x1 = max(0, int(x_center - width / 2))
        y1 = max(0, int(y_center - height / 2))
        x2 = min(img_w, int(x_center + width / 2))
        y2 = min(img_h, int(y_center + height / 2))

        if x2 <= x1 or y2 <= y1:
            print(f"  skipping degenerate box {i+1} in {label_path.name}")
            continue

        crop = image.crop((x1, y1, x2, y2))
        output_name = f"{image_path.stem}_stamp_{i+1}{image_path.suffix}"
        crop.save(OUTPUT_DIR / output_name)
        counts["crops"] += 1
        print(f"Saved: {output_name} (label: {label_path.name})")

print(
    f"\ndonez. {counts['with']} with stamps, {counts['without']} without, "
    f"{counts['missing']} with no label file, {counts['crops']} crops"
)