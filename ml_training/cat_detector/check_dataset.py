from pathlib import Path

from PIL import Image

import config


# ============================================================
# PAWCARE - CAT VS NON-CAT DATASET CHECK
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def get_image_files(folder):
    """
    Return all supported image files inside a folder.
    """
    if not folder.exists():
        return []

    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def check_image(image_path):
    """
    Check whether an image can be opened and verified.
    """
    try:
        with Image.open(image_path) as image:
            image.verify()

        return True

    except Exception:
        return False


def check_split(split_name, split_dir):
    """
    Check one dataset split.
    """

    print("\n" + "=" * 70)
    print(f"{split_name.upper()} DATASET")
    print("=" * 70)

    if not split_dir.exists():
        print(f"ERROR: Directory not found:")
        print(split_dir)
        return 0, 0, False

    class_directories = sorted(
        [
            folder
            for folder in split_dir.iterdir()
            if folder.is_dir()
        ]
    )

    detected_classes = [
        folder.name
        for folder in class_directories
    ]

    print(f"\nDetected classes: {len(detected_classes)}")

    for class_name in detected_classes:
        print(f"  - {class_name}")

    expected_classes = set(config.EXPECTED_CLASSES)
    detected_classes_set = set(detected_classes)

    if detected_classes_set == expected_classes:
        print("\n✓ Class folders match expected classes")

    else:
        print("\n✗ CLASS FOLDER MISMATCH")

        missing_classes = expected_classes - detected_classes_set
        unexpected_classes = detected_classes_set - expected_classes

        if missing_classes:
            print("Missing classes:")
            for class_name in sorted(missing_classes):
                print(f"  - {class_name}")

        if unexpected_classes:
            print("Unexpected classes:")
            for class_name in sorted(unexpected_classes):
                print(f"  - {class_name}")

        return 0, 0, False

    total_images = 0
    invalid_images = 0

    print("\nImage counts:")
    print("-" * 70)

    for class_name in config.EXPECTED_CLASSES:

        class_dir = split_dir / class_name

        image_files = get_image_files(class_dir)

        print(
            f"{class_name:<15} : "
            f"{len(image_files):>5} images"
        )

        total_images += len(image_files)

        for image_path in image_files:

            if not check_image(image_path):

                invalid_images += 1

                print(
                    f"  ✗ Invalid image: "
                    f"{image_path.name}"
                )

    print("-" * 70)
    print(f"Total images       : {total_images}")
    print(f"Invalid images     : {invalid_images}")

    if invalid_images == 0:
        print("✓ No corrupted/unreadable images found")

    else:
        print("✗ Invalid images detected")

    return total_images, invalid_images, True


def verify_class_consistency():

    print("\n" + "=" * 70)
    print("CLASS CONSISTENCY CHECK")
    print("=" * 70)

    split_directories = [
        config.TRAIN_DIR,
        config.VALIDATION_DIR,
        config.TEST_DIR
    ]

    class_sets = []

    for split_dir in split_directories:

        if not split_dir.exists():
            print(
                f"✗ Missing split directory: "
                f"{split_dir}"
            )

            return False

        classes = {
            folder.name
            for folder in split_dir.iterdir()
            if folder.is_dir()
        }

        class_sets.append(classes)

    if (
        class_sets[0]
        == class_sets[1]
        == class_sets[2]
        == set(config.EXPECTED_CLASSES)
    ):

        print("✓ All splits contain the same classes")

        print(
            f"✓ Expected classes: "
            f"{config.EXPECTED_CLASSES}"
        )

        return True

    print("✗ Class consistency check failed")

    print(f"Training classes   : {sorted(class_sets[0])}")
    print(f"Validation classes : {sorted(class_sets[1])}")
    print(f"Test classes       : {sorted(class_sets[2])}")

    return False


def main():

    print("=" * 70)
    print("PAWCARE - CAT VS NON-CAT DATASET VERIFICATION")
    print("=" * 70)

    print("\nDataset:")
    print(config.DATASET_DIR)

    print("\nExpected classes:")
    for index, class_name in enumerate(config.EXPECTED_CLASSES):
        print(f"  {index}: {class_name}")

    # --------------------------------------------------------
    # Check dataset directory
    # --------------------------------------------------------

    if not config.DATASET_DIR.exists():

        print("\n✗ DATASET DIRECTORY NOT FOUND")

        print(
            f"Expected path: "
            f"{config.DATASET_DIR}"
        )

        raise SystemExit(1)

    print("\n✓ Dataset directory exists")

    # --------------------------------------------------------
    # Check train dataset
    # --------------------------------------------------------

    train_total, train_invalid, train_ok = check_split(
        "Training",
        config.TRAIN_DIR
    )

    # --------------------------------------------------------
    # Check validation dataset
    # --------------------------------------------------------

    validation_total, validation_invalid, validation_ok = check_split(
        "Validation",
        config.VALIDATION_DIR
    )

    # --------------------------------------------------------
    # Check test dataset
    # --------------------------------------------------------

    test_total, test_invalid, test_ok = check_split(
        "Test",
        config.TEST_DIR
    )

    # --------------------------------------------------------
    # Class consistency
    # --------------------------------------------------------

    classes_consistent = verify_class_consistency()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    total_images = (
        train_total
        + validation_total
        + test_total
    )

    total_invalid = (
        train_invalid
        + validation_invalid
        + test_invalid
    )

    print("\n" + "=" * 70)
    print("DATASET VERIFICATION SUMMARY")
    print("=" * 70)

    print(f"\nTraining images   : {train_total}")
    print(f"Validation images : {validation_total}")
    print(f"Test images       : {test_total}")
    print(f"Total images      : {total_images}")

    print(f"\nInvalid images    : {total_invalid}")

    print(
        f"\nExpected classes  : "
        f"{config.NUM_CLASSES}"
    )

    print(
        f"Class consistency : "
        f"{'PASSED' if classes_consistent else 'FAILED'}"
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    all_checks_passed = (
        train_ok
        and validation_ok
        and test_ok
        and classes_consistent
        and total_invalid == 0
    )

    print("\n" + "=" * 70)

    if all_checks_passed:

        print("STATUS: DATASET VERIFICATION PASSED")

    else:

        print("STATUS: DATASET VERIFICATION FAILED")

    print("=" * 70)

    if not all_checks_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()