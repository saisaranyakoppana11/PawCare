from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = Path("data/Cat breeds dataset")
OUTPUT_DIR = Path("data/cat_breed_split")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRAIN_SIZE = 0.80
VALIDATION_SIZE = 0.10
TEST_SIZE = 0.10

RANDOM_STATE = 42


# ============================================================
# CHECK SOURCE DATASET
# ============================================================

if not SOURCE_DIR.exists():
    print(f"ERROR: Dataset folder not found:")
    print(SOURCE_DIR)
    raise SystemExit


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

for split in ["train", "validation", "test"]:
    (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)


# ============================================================
# GET BREED FOLDERS
# ============================================================

breed_folders = sorted(
    folder for folder in SOURCE_DIR.iterdir()
    if folder.is_dir()
)

print("=" * 70)
print("PREPARING CAT BREED DATASET")
print("=" * 70)

print(f"\nNumber of classes: {len(breed_folders)}")
print(f"Train: {TRAIN_SIZE * 100:.0f}%")
print(f"Validation: {VALIDATION_SIZE * 100:.0f}%")
print(f"Test: {TEST_SIZE * 100:.0f}%")
print()


# ============================================================
# PROCESS EACH BREED
# ============================================================

total_train = 0
total_validation = 0
total_test = 0

for breed_folder in breed_folders:

    breed = breed_folder.name

    images = [
        file for file in breed_folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # First split: 80% train, 20% temporary
    train_files, temp_files = train_test_split(
        images,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    # Second split: temporary 20% into 10% validation + 10% test
    validation_files, test_files = train_test_split(
        temp_files,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    # Create breed directories
    train_dir = OUTPUT_DIR / "train" / breed
    validation_dir = OUTPUT_DIR / "validation" / breed
    test_dir = OUTPUT_DIR / "test" / breed

    train_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Copy images
    for image in train_files:
        shutil.copy2(image, train_dir / image.name)

    for image in validation_files:
        shutil.copy2(image, validation_dir / image.name)

    for image in test_files:
        shutil.copy2(image, test_dir / image.name)

    # Counts
    total_train += len(train_files)
    total_validation += len(validation_files)
    total_test += len(test_files)

    print(
        f"{breed:20} "
        f"Train: {len(train_files):3} | "
        f"Validation: {len(validation_files):2} | "
        f"Test: {len(test_files):2}"
    )


# ============================================================
# SUMMARY
# ============================================================

total = total_train + total_validation + total_test

print("\n" + "=" * 70)
print("DATASET SPLIT COMPLETE")
print("=" * 70)

print(f"Training images   : {total_train}")
print(f"Validation images : {total_validation}")
print(f"Testing images    : {total_test}")
print(f"Total images      : {total}")

print("\nDataset location:")
print(OUTPUT_DIR)

print("\nSTATUS: Dataset split completed successfully.")