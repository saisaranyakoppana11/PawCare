from pathlib import Path

from PIL import Image


# ============================================================
# PawCare - Dog Breed Classifier
# Dataset Verification
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\saisa\OneDrive\Documents\PawCare"
)

DATASET_DIR = (
    PROJECT_ROOT
    / "dog_breed"
    / "dataset"
)

TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"


EXPECTED_CLASSES = [
    "beagle",
    "boxer",
    "German_shepherd",
    "golden_retriever",
    "pug",
    "shitzu",
    "Siberian_husky",
    "standard_poodle",
    "whippet",
    "Yorkshire_terrier"
]


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# Helper Functions
# ============================================================


def get_image_files(folder):

    if not folder.exists():
        return []

    return [
        file
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in VALID_EXTENSIONS
    ]


def verify_images(files):

    invalid_files = []

    for file in files:

        try:

            with Image.open(file) as image:
                image.verify()

        except Exception:
            invalid_files.append(file)

    return invalid_files


def verify_split(split_name, split_dir):

    print("\n" + "-" * 70)
    print(f"{split_name.upper()} DATASET")
    print("-" * 70)

    if not split_dir.exists():

        raise FileNotFoundError(
            f"Dataset split not found:\n{split_dir}"
        )

    total_images = 0

    split_class_names = sorted(
        [
            folder.name
            for folder in split_dir.iterdir()
            if folder.is_dir()
        ]
    )

    print(
        f"\nClasses found:\n"
        f"{split_class_names}"
    )

    if split_class_names != sorted(EXPECTED_CLASSES):

        raise ValueError(
            "\nClass mismatch detected!\n"
            f"Expected: {sorted(EXPECTED_CLASSES)}\n"
            f"Found:    {split_class_names}"
        )

    for class_name in EXPECTED_CLASSES:

        class_dir = split_dir / class_name

        image_files = get_image_files(
            class_dir
        )

        count = len(image_files)

        total_images += count

        print(
            f"{class_name:<25} : {count}"
        )

        if count == 0:

            raise ValueError(
                f"No images found for class: "
                f"{class_name}"
            )

        invalid_files = verify_images(
            image_files
        )

        if invalid_files:

            print(
                f"\nInvalid/corrupt images "
                f"found in {class_name}:"
            )

            for file in invalid_files:
                print(file)

            raise ValueError(
                f"Dataset contains invalid "
                f"images in class: {class_name}"
            )

    print(
        f"\nTotal {split_name.lower()} images: "
        f"{total_images}"
    )

    return total_images


# ============================================================
# Main Verification
# ============================================================


def main():

    print("=" * 70)
    print("PAWCARE - DOG BREED CLASSIFIER")
    print("DATASET VERIFICATION")
    print("=" * 70)

    print(
        f"\nDataset location:\n"
        f"{DATASET_DIR}"
    )

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"\nDataset directory not found:\n"
            f"{DATASET_DIR}"
        )

    print("\n✓ Dataset directory exists")

    # --------------------------------------------------------
    # Verify all three splits
    # --------------------------------------------------------

    train_count = verify_split(
        "Training",
        TRAIN_DIR
    )

    validation_count = verify_split(
        "Validation",
        VALIDATION_DIR
    )

    test_count = verify_split(
        "Test",
        TEST_DIR
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    total_count = (
        train_count
        + validation_count
        + test_count
    )

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        f"\nTraining images:   {train_count}"
    )

    print(
        f"Validation images: {validation_count}"
    )

    print(
        f"Test images:       {test_count}"
    )

    print(
        f"Total images:      {total_count}"
    )

    print(
        f"Number of classes: "
        f"{len(EXPECTED_CLASSES)}"
    )

    print(
        f"\nClasses:\n"
        f"{EXPECTED_CLASSES}"
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STATUS: DATASET VERIFICATION PASSED")
    print("=" * 70)


if __name__ == "__main__":

    main()