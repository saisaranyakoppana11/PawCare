from pathlib import Path
from PIL import Image

# Dataset path
DATASET_DIR = Path("data/Cat breeds dataset")

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("=" * 60)
print("CAT BREED DATASET VERIFICATION")
print("=" * 60)

if not DATASET_DIR.exists():
    print(f"\nERROR: Dataset folder not found:")
    print(DATASET_DIR)
    raise SystemExit

breed_folders = sorted(
    folder for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
)

print(f"\nNumber of breed classes: {len(breed_folders)}\n")

total_images = 0
total_valid = 0
total_invalid = 0

for breed_folder in breed_folders:

    images = [
        file for file in breed_folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    valid = 0
    invalid = 0

    for image_file in images:
        try:
            with Image.open(image_file) as img:
                img.verify()
            valid += 1

        except Exception:
            invalid += 1
            print(f"Invalid image: {image_file}")

    total = valid + invalid

    total_images += total
    total_valid += valid
    total_invalid += invalid

    print(
        f"{breed_folder.name:20} "
        f"Total: {total:3} | "
        f"Valid: {valid:3} | "
        f"Invalid: {invalid:3}"
    )

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"Breed classes   : {len(breed_folders)}")
print(f"Total images    : {total_images}")
print(f"Valid images    : {total_valid}")
print(f"Invalid images  : {total_invalid}")

if total_invalid == 0:
    print("\nSTATUS: Dataset is clean and ready for classification.")
else:
    print("\nSTATUS: Invalid images found. We should clean them before training.")