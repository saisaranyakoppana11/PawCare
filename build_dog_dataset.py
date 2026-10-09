from pathlib import Path
import shutil
import random

PROJECT = Path(r"C:\Users\saisa\Documents\PawCare")

SOURCE = Path(
    r"C:\Users\saisa\Downloads\object detection\dog skin disease\yolo_dataset"
)

OUTPUT = PROJECT / "dog_disease_final_dataset"

CLASS_MAP = {
    0: 0,  # demodicosis
    1: 1,  # Dermatitis
    2: 2,  # Fungal_infections
    4: 3,  # Hypersensitivity
    5: 4,  # ringworm
}

CLASS_NAMES = {
    0: "demodicosis",
    1: "Dermatitis",
    2: "Fungal_infections",
    3: "Hypersensitivity",
    4: "ringworm",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(42)


def find_labels(label_root):
    lookup = {}

    if not label_root.exists():
        return lookup

    for txt in label_root.rglob("*.txt"):
        lookup.setdefault(txt.stem.lower(), []).append(txt)

    return lookup


def read_valid_label(label_file):
    output = []

    for line in label_file.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            continue

        old_id = int(parts[0])

        if old_id not in CLASS_MAP:
            continue

        parts[0] = str(CLASS_MAP[old_id])

        output.append(" ".join(parts))

    return output


print("=" * 65)
print("BUILDING FINAL DOG DISEASE DATASET")
print("=" * 65)

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

for split in ["train", "val", "test"]:
    (OUTPUT / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )
    (OUTPUT / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# ------------------------------------------------------------
# COLLECT ALL LABELED DISEASE IMAGES
# ------------------------------------------------------------

disease_samples = []

for original_split in ["train", "valid", "val", "test"]:

    image_root = SOURCE / "images" / original_split
    label_root = SOURCE / "labels" / original_split

    if not image_root.exists():
        continue

    label_lookup = find_labels(label_root)

    for image in image_root.rglob("*"):

        if not image.is_file():
            continue

        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Healthy images are handled separately.
        if any(
            part.lower() == "healthy"
            for part in image.parts
        ):
            continue

        matches = label_lookup.get(
            image.stem.lower(),
            []
        )

        if len(matches) == 0:
            continue

        # Prefer a label that contains usable annotations.
        selected_label = None

        for label in matches:

            try:
                labels = read_valid_label(label)
            except Exception:
                continue

            if labels:
                selected_label = (image, labels)
                break

        if selected_label:
            disease_samples.append(selected_label)


print(f"\nUsable labeled disease images: {len(disease_samples)}")


# ------------------------------------------------------------
# COLLECT HEALTHY BACKGROUND IMAGES
# ------------------------------------------------------------

healthy_samples = []

for original_split in ["train", "valid", "val", "test"]:

    image_root = SOURCE / "images" / original_split

    if not image_root.exists():
        continue

    for image in image_root.rglob("*"):

        if not image.is_file():
            continue

        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if any(
            part.lower() == "healthy"
            for part in image.parts
        ):
            healthy_samples.append(image)


print(f"Healthy background images: {len(healthy_samples)}")


# ------------------------------------------------------------
# SHUFFLE
# ------------------------------------------------------------

random.shuffle(disease_samples)
random.shuffle(healthy_samples)


# ------------------------------------------------------------
# SPLIT DISEASE DATA
#
# 70% train
# 15% validation
# 15% test
# ------------------------------------------------------------

n = len(disease_samples)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

disease_train = disease_samples[:train_end]
disease_val = disease_samples[train_end:val_end]
disease_test = disease_samples[val_end:]


# ------------------------------------------------------------
# SPLIT HEALTHY DATA
# ------------------------------------------------------------

hn = len(healthy_samples)

healthy_train_end = int(hn * 0.70)
healthy_val_end = int(hn * 0.85)

healthy_train = healthy_samples[:healthy_train_end]
healthy_val = healthy_samples[
    healthy_train_end:healthy_val_end
]
healthy_test = healthy_samples[
    healthy_val_end:
]


# ------------------------------------------------------------
# COPY DISEASE IMAGES + LABELS
# ------------------------------------------------------------

def copy_disease(samples, split):

    count = 0

    for index, (image, labels) in enumerate(samples):

        filename = (
            f"disease_{index:05d}"
            + image.suffix.lower()
        )

        destination_image = (
            OUTPUT / "images" / split / filename
        )

        destination_label = (
            OUTPUT / "labels" / split /
            f"disease_{index:05d}.txt"
        )

        shutil.copy2(
            image,
            destination_image
        )

        destination_label.write_text(
            "\n".join(labels) + "\n",
            encoding="utf-8"
        )

        count += 1

    return count


# ------------------------------------------------------------
# COPY HEALTHY BACKGROUND IMAGES
# ------------------------------------------------------------

def copy_healthy(samples, split, start_index=0):

    count = 0

    for index, image in enumerate(samples):

        filename = (
            f"healthy_{start_index + index:05d}"
            + image.suffix.lower()
        )

        destination_image = (
            OUTPUT / "images" / split / filename
        )

        shutil.copy2(
            image,
            destination_image
        )

        # IMPORTANT:
        # No label file for Healthy background images.

        count += 1

    return count


train_d = copy_disease(
    disease_train,
    "train"
)

val_d = copy_disease(
    disease_val,
    "val"
)

test_d = copy_disease(
    disease_test,
    "test"
)

train_h = copy_healthy(
    healthy_train,
    "train"
)

val_h = copy_healthy(
    healthy_val,
    "val"
)

test_h = copy_healthy(
    healthy_test,
    "test"
)


# ------------------------------------------------------------
# WRITE YAML
# ------------------------------------------------------------

yaml_path = OUTPUT / "dataset.yaml"

yaml_text = f"""path: {OUTPUT.as_posix()}

train: images/train
val: images/val
test: images/test

names:
  0: demodicosis
  1: Dermatitis
  2: Fungal_infections
  3: Hypersensitivity
  4: ringworm
"""

yaml_path.write_text(
    yaml_text,
    encoding="utf-8"
)


# ------------------------------------------------------------
# VERIFY DATASET BEFORE TRAINING
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FINAL DATASET")
print("=" * 65)

print(f"\nTRAIN")
print(f"  Disease : {train_d}")
print(f"  Healthy : {train_h}")

print(f"\nVALIDATION")
print(f"  Disease : {val_d}")
print(f"  Healthy : {val_h}")

print(f"\nTEST")
print(f"  Disease : {test_d}")
print(f"  Healthy : {test_h}")


# Count label files
for split in ["train", "val", "test"]:

    images = list(
        (OUTPUT / "images" / split).glob("*")
    )

    labels = list(
        (OUTPUT / "labels" / split).glob("*.txt")
    )

    print(
        f"\n{split}: "
        f"{len(images)} images, "
        f"{len(labels)} labeled images"
    )


print("\n" + "=" * 65)
print("DATASET READY")
print("=" * 65)

print(f"\nDataset:")
print(OUTPUT)

print(f"\nYAML:")
print(yaml_path)

print(
    "\nDO NOT START TRAINING YET."
)

print(
    "Check that VALIDATION shows a non-zero Disease count "
    "and a non-zero labeled-image count."
)