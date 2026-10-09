from pathlib import Path
import shutil
from ultralytics import YOLO

# ============================================================
# PAWCARE - CLEAN DOG DISEASE DATASET + TRAINING
# ============================================================

PROJECT = Path(r"C:\Users\saisa\Documents\PawCare")

# ORIGINAL DATASET - NEVER MODIFY THIS
SOURCE = Path(
    r"C:\Users\saisa\Downloads\object detection\dog skin disease\yolo_dataset"
)

# NEW CLEAN DATASET
DATASET = PROJECT / "dog_disease_clean_v2"

# NEW TRAINING OUTPUT
TRAINING_OUTPUT = PROJECT / "disease_training_outputs"

BASE_MODEL = PROJECT / "disease_models" / "yolo26n.pt"

# Old class IDs -> new class IDs
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

print("=" * 65)
print("PAWCARE DOG DISEASE - CLEAN DATASET PREPARATION")
print("=" * 65)

# ------------------------------------------------------------
# 1. Check source
# ------------------------------------------------------------

if not SOURCE.exists():
    raise FileNotFoundError(f"Original dataset not found:\n{SOURCE}")

if not BASE_MODEL.exists():
    raise FileNotFoundError(f"Base model not found:\n{BASE_MODEL}")

print(f"\nOriginal dataset:")
print(SOURCE)

print("\nThe original dataset will NOT be modified.")

# ------------------------------------------------------------
# 2. Remove only our previous clean copy
# ------------------------------------------------------------

if DATASET.exists():
    print("\nRemoving previous clean dataset...")
    shutil.rmtree(DATASET)

# Create clean YOLO folders
for split in ["train", "valid", "test"]:
    (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
    (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 3. Build label lookup
# ------------------------------------------------------------

def build_label_lookup(label_root):
    lookup = {}

    if not label_root.exists():
        return lookup

    for txt in label_root.rglob("*.txt"):
        key = txt.stem.lower()

        if key not in lookup:
            lookup[key] = txt

    return lookup


# ------------------------------------------------------------
# 4. Process each split
# ------------------------------------------------------------

total_copied = 0
total_positive = 0
total_healthy = 0
total_skipped = 0

for split in ["train", "valid", "test"]:

    source_images = SOURCE / "images" / split
    source_labels = SOURCE / "labels" / split

    # Some datasets use "valid" instead of "val"
    if not source_images.exists():
        print(f"\n{split}: image folder not found - skipped")
        continue

    label_lookup = build_label_lookup(source_labels)

    print(f"\nProcessing {split}...")

    split_copied = 0
    split_positive = 0
    split_healthy = 0
    split_skipped = 0

    images = [
        p for p in source_images.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for index, image_path in enumerate(images):

        # ----------------------------------------------------
        # Determine whether this is a Healthy image
        # ----------------------------------------------------

        is_healthy = any(
            part.lower() == "healthy"
            for part in image_path.parts
        )

        # ----------------------------------------------------
        # Find corresponding label
        # ----------------------------------------------------

        label_path = label_lookup.get(image_path.stem.lower())

        # ----------------------------------------------------
        # Healthy = background / negative image
        # ----------------------------------------------------

        if is_healthy:

            destination_name = f"{split}_healthy_{index:05d}{image_path.suffix.lower()}"

            destination_image = (
                DATASET / "images" / split / destination_name
            )

            shutil.copy2(image_path, destination_image)

            split_copied += 1
            split_healthy += 1

            # IMPORTANT:
            # No .txt file is created for Healthy.
            # This makes it a background/negative image.

            continue

        # ----------------------------------------------------
        # Non-Healthy image must have a real label
        # ----------------------------------------------------

        if label_path is None:
            split_skipped += 1
            continue

        try:
            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()
        except Exception:
            split_skipped += 1
            continue

        new_lines = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            old_class = int(parts[0])

            # Healthy must never become a bounding box
            if old_class == 3:
                continue

            if old_class not in CLASS_MAP:
                continue

            parts[0] = str(CLASS_MAP[old_class])

            new_lines.append(" ".join(parts))

        # If there is no usable disease annotation,
        # do NOT treat this disease image as Healthy.
        if not new_lines:
            split_skipped += 1
            continue

        destination_name = (
            f"{split}_disease_{index:05d}{image_path.suffix.lower()}"
        )

        destination_image = (
            DATASET / "images" / split / destination_name
        )

        destination_label = (
            DATASET / "labels" / split /
            f"{split}_disease_{index:05d}.txt"
        )

        shutil.copy2(image_path, destination_image)

        destination_label.write_text(
            "\n".join(new_lines) + "\n",
            encoding="utf-8"
        )

        split_copied += 1
        split_positive += 1

    total_copied += split_copied
    total_positive += split_positive
    total_healthy += split_healthy
    total_skipped += split_skipped

    print(f"  Images copied : {split_copied}")
    print(f"  Disease images: {split_positive}")
    print(f"  Healthy images: {split_healthy}")
    print(f"  Skipped       : {split_skipped}")


# ------------------------------------------------------------
# 5. Create correct dataset.yaml
# ------------------------------------------------------------

yaml_path = DATASET / "dataset.yaml"

yaml_text = f"""path: {DATASET.as_posix()}

train: images/train
val: images/valid
test: images/test

names:
  0: demodicosis
  1: Dermatitis
  2: Fungal_infections
  3: Hypersensitivity
  4: ringworm
"""

yaml_path.write_text(yaml_text, encoding="utf-8")

print("\n" + "=" * 65)
print("CLEAN DATASET CREATED")
print("=" * 65)

print(f"\nTotal images copied : {total_copied}")
print(f"Disease images      : {total_positive}")
print(f"Healthy background  : {total_healthy}")
print(f"Skipped             : {total_skipped}")

print(f"\nDataset:")
print(DATASET)

print(f"\nYAML:")
print(yaml_path)

# ------------------------------------------------------------
# 6. TRAIN
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("STARTING DOG DISEASE TRAINING")
print("=" * 65)

model = YOLO(str(BASE_MODEL))

model.train(
    data=str(yaml_path),

    # Faster CPU training
    epochs=20,
    imgsz=416,
    batch=8,

    # Windows CPU
    workers=0,
    device="cpu",
    cache=False,

    # Stop early if validation stops improving
    patience=7,

    # Save separately
    project=str(TRAINING_OUTPUT),
    name="dog_v2_clean",
    exist_ok=True,

    save=True,
    save_period=5,

    verbose=True
)

# ------------------------------------------------------------
# 7. Locate best model
# ------------------------------------------------------------

best_model = (
    TRAINING_OUTPUT /
    "dog_v2_clean" /
    "weights" /
    "best.pt"
)

print("\n" + "=" * 65)
print("TRAINING FINISHED")
print("=" * 65)

if not best_model.exists():
    print("\nERROR: best.pt was not created.")
    raise SystemExit

print(f"\nBest model:")
print(best_model)

# ------------------------------------------------------------
# 8. TEST EVALUATION
# ------------------------------------------------------------

print("\nRunning TEST evaluation...")

trained_model = YOLO(str(best_model))

metrics = trained_model.val(
    data=str(yaml_path),
    split="test",
    imgsz=416,
    batch=8,
    workers=0,
    device="cpu"
)

print("\n" + "=" * 65)
print("FINAL TEST RESULTS")
print("=" * 65)

print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")
print(f"mAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")

print("\nNew model:")
print(best_model)

print("\nIMPORTANT:")
print("Existing disease_models/dog/best.pt was NOT changed.")