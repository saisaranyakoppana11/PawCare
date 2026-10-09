from pathlib import Path
from PIL import Image


# ============================================================
# PawCare - Dog Detector Dataset Verification
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\saisa\OneDrive\Documents\PawCare")

DATASET_DIR = (
    PROJECT_ROOT
    / "dog_breed"
    / "dog_detector"
    / "dataset"
)

SPLITS = ["train", "validation", "test"]
EXPECTED_CLASSES = ["dog", "non_dog"]

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def get_image_files(folder):
    """Return all supported image files inside a folder."""
    return [
        file
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    ]


def verify_image(image_path):
    """Check whether an image can be opened and verified."""
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except Exception:
        return False


def count_images():
    """Count images for every split and class."""
    total_images = 0
    invalid_images = []

    print("=" * 70)
    print("PAWCARE - DOG DETECTOR DATASET VERIFICATION")
    print("=" * 70)

    print(f"\nDataset path:")
    print(DATASET_DIR)

    if not DATASET_DIR.exists():
        print("\nERROR: Dataset directory does not exist.")
        return False

    print("\nChecking dataset structure...")

    # --------------------------------------------------------
    # Check split directories
    # --------------------------------------------------------

    for split in SPLITS:
        split_dir = DATASET_DIR / split

        if not split_dir.exists():
            print(f"ERROR: Missing split directory: {split}")
            return False

        print(f"✓ Found: {split}")

    # --------------------------------------------------------
    # Check class directories
    # --------------------------------------------------------

    print("\nChecking class directories...")

    for split in SPLITS:

        split_dir = DATASET_DIR / split

        classes = sorted(
            [
                folder.name
                for folder in split_dir.iterdir()
                if folder.is_dir()
            ]
        )

        if classes != EXPECTED_CLASSES:
            print(
                f"ERROR: Class mismatch in {split}. "
                f"Found: {classes}"
            )
            print(
                f"Expected: {EXPECTED_CLASSES}"
            )
            return False

        print(
            f"✓ {split}: classes = {classes}"
        )

    # --------------------------------------------------------
    # Count images
    # --------------------------------------------------------

    print("\nImage counts:")
    print("-" * 70)

    all_counts = {}

    for split in SPLITS:

        split_total = 0
        all_counts[split] = {}

        print(f"\n{split.upper()}")

        for class_name in EXPECTED_CLASSES:

            class_dir = DATASET_DIR / split / class_name

            images = get_image_files(class_dir)

            count = len(images)

            all_counts[split][class_name] = count
            split_total += count
            total_images += count

            print(
                f"  {class_name:<10}: {count:>5} images"
            )

            # ------------------------------------------------
            # Verify image files
            # ------------------------------------------------

            for image_path in images:

                if not verify_image(image_path):
                    invalid_images.append(image_path)

        print(
            f"  {'TOTAL':<10}: {split_total:>5} images"
        )

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    for split in SPLITS:

        split_total = sum(
            all_counts[split].values()
        )

        print(
            f"{split:<12}: {split_total} images"
        )

    print(
        f"{'TOTAL':<12}: {total_images} images"
    )

    # --------------------------------------------------------
    # Invalid image report
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    if invalid_images:

        print(
            f"ERROR: Found {len(invalid_images)} invalid/corrupt images."
        )

        print("\nInvalid files:")

        for image_path in invalid_images:
            print(f"  {image_path}")

        return False

    else:

        print("✓ No invalid or corrupt images found.")

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    print("\n" + "-" * 70)

    class_consistency = True

    for split in SPLITS:

        for class_name in EXPECTED_CLASSES:

            if all_counts[split][class_name] == 0:

                print(
                    f"ERROR: {split}/{class_name} contains no images."
                )

                class_consistency = False

    if not class_consistency:
        return False

    print("✓ All expected classes contain images.")

    print("\n" + "=" * 70)
    print("STATUS: DATASET VERIFICATION PASSED")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = count_images()

    if not success:
        raise SystemExit(1)