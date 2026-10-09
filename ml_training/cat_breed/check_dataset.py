from pathlib import Path

from PIL import Image, UnidentifiedImageError

from config import (
    DATASET_DIR,
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR,
    NUM_CLASSES,
)


# ============================================================
# PAWCARE - CAT BREED CLASSIFIER
# DATASET VERIFICATION
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_class_folders(directory):
    """Return sorted class-folder names from a dataset split."""

    if not directory.exists():
        return []

    return sorted(
        folder.name
        for folder in directory.iterdir()
        if folder.is_dir()
    )


def get_image_files(directory):
    """Return all supported image files recursively."""

    if not directory.exists():
        return []

    return [
        file
        for file in directory.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def check_image_file(image_path):
    """Check whether an image can be opened and verified."""

    try:
        with Image.open(image_path) as image:
            image.verify()

        return True, ""

    except (UnidentifiedImageError, OSError, ValueError) as error:
        return False, str(error)


def check_split(split_name, split_dir):
    """Verify one dataset split."""

    print("\n" + "=" * 70)
    print(f"{split_name.upper()} DATASET")
    print("=" * 70)

    if not split_dir.exists():
        print(f"ERROR: {split_name} directory does not exist:")
        print(split_dir)
        return False, 0

    classes = get_class_folders(split_dir)

    print(f"\nLocation: {split_dir}")
    print(f"Classes found: {len(classes)}")

    if len(classes) != NUM_CLASSES:
        print(
            f"WARNING: Expected {NUM_CLASSES} classes "
            f"but found {len(classes)}."
        )

    print("\nClass image counts:")

    total_images = 0
    invalid_images = []

    for class_name in classes:

        class_dir = split_dir / class_name

        images = get_image_files(class_dir)

        print(f"{class_name:25} : {len(images):4}")

        total_images += len(images)

        for image_path in images:

            valid, error = check_image_file(image_path)

            if not valid:
                invalid_images.append(
                    (image_path, error)
                )

    print(f"\nTotal images: {total_images}")

    if invalid_images:

        print("\nINVALID / CORRUPTED IMAGES")
        print("-" * 70)

        for image_path, error in invalid_images:

            print(f"File : {image_path}")
            print(f"Error: {error}")
            print()

    else:

        print("\nNo invalid or corrupted images detected.")

    return len(classes) == NUM_CLASSES and not invalid_images, total_images


def main():

    print("\n" + "=" * 70)
    print("PAWCARE - CAT BREED CLASSIFIER")
    print("DATASET VERIFICATION")
    print("=" * 70)

    print("\nDataset directory:")
    print(DATASET_DIR)

    # --------------------------------------------------------
    # Check main dataset directory
    # --------------------------------------------------------

    if not DATASET_DIR.exists():

        print("\nERROR: Dataset directory does not exist.")

        raise SystemExit(1)

    print("\nDataset directory exists: YES")

    # --------------------------------------------------------
    # Verify each split
    # --------------------------------------------------------

    train_ok, train_count = check_split(
        "Training",
        TRAIN_DIR,
    )

    validation_ok, validation_count = check_split(
        "Validation",
        VALIDATION_DIR,
    )

    test_ok, test_count = check_split(
        "Test",
        TEST_DIR,
    )

    # --------------------------------------------------------
    # Verify class consistency
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLASS CONSISTENCY CHECK")
    print("=" * 70)

    train_classes = get_class_folders(TRAIN_DIR)
    validation_classes = get_class_folders(VALIDATION_DIR)
    test_classes = get_class_folders(TEST_DIR)

    print("\nTraining classes:")
    print(train_classes)

    print("\nValidation classes:")
    print(validation_classes)

    print("\nTest classes:")
    print(test_classes)

    classes_match = (
        train_classes
        == validation_classes
        == test_classes
    )

    if classes_match:

        print("\nClass consistency: PASSED")

    else:

        print("\nClass consistency: FAILED")

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    total_images = (
        train_count
        + validation_count
        + test_count
    )

    all_checks_passed = (
        train_ok
        and validation_ok
        and test_ok
        and classes_match
    )

    print("\n" + "=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)

    print(f"\nNumber of classes : {len(train_classes)}")
    print(f"Training images   : {train_count}")
    print(f"Validation images : {validation_count}")
    print(f"Test images       : {test_count}")
    print(f"Total images      : {total_images}")

    print("\nExpected classes  :", NUM_CLASSES)

    print("\nDataset checks:")

    print(
        f"Dataset directory : "
        f"{'PASSED' if DATASET_DIR.exists() else 'FAILED'}"
    )

    print(
        f"Training split    : "
        f"{'PASSED' if train_ok else 'FAILED'}"
    )

    print(
        f"Validation split  : "
        f"{'PASSED' if validation_ok else 'FAILED'}"
    )

    print(
        f"Test split        : "
        f"{'PASSED' if test_ok else 'FAILED'}"
    )

    print(
        f"Class consistency : "
        f"{'PASSED' if classes_match else 'FAILED'}"
    )

    print("\n" + "=" * 70)

    if all_checks_passed:

        print("STATUS: DATASET VERIFICATION PASSED")
        print("=" * 70)

    else:

        print("STATUS: DATASET VERIFICATION FAILED")
        print("Please fix the reported problems before training.")
        print("=" * 70)

        raise SystemExit(1)


if __name__ == "__main__":
    main()