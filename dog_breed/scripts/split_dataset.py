import shutil
import random
from pathlib import Path


# ============================================================
# PAWCARE - DOG BREED DATASET SPLITTER
# ============================================================

# ------------------------------------------------------------
# 1. ORIGINAL DATASET LOCATION
# ------------------------------------------------------------

SOURCE_DIR = Path(
    r"C:\Users\saisa\Downloads\Dog_Breed_Classification (2)"
    r"\Dog_Breed_Classification\dataset"
)


# ------------------------------------------------------------
# 2. NEW PAWCARE DATASET LOCATION
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    r"C:\Users\saisa\OneDrive\Documents\PawCare"
    r"\dog_breed\dataset"
)


# ------------------------------------------------------------
# 3. DOG BREEDS
# ------------------------------------------------------------

BREEDS = [
    "beagle",
    "boxer",
    "German_shepherd",
    "golden_retriever",
    "pug",
    "shitzu",
    "Siberian_husky",
    "standard_poodle",
    "whippet",
    "Yorkshire_terrier",
]


# ------------------------------------------------------------
# 4. SETTINGS
# ------------------------------------------------------------

RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_images(folder):
    """
    Get all supported image files from a folder.
    """

    if not folder.exists():
        return []

    return [
        file
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def create_output_folders():
    """
    Create train, validation and test folders
    for all dog breeds.
    """

    for split in ["train", "validation", "test"]:

        for breed in BREEDS:

            folder = OUTPUT_DIR / split / breed

            folder.mkdir(
                parents=True,
                exist_ok=True
            )


def clear_existing_dataset():
    """
    Remove the previously generated PawCare dog dataset
    if it already exists.

    This prevents duplicate images if the script is
    executed more than once.
    """

    if OUTPUT_DIR.exists():

        print("\nRemoving existing PawCare dog dataset...")

        shutil.rmtree(OUTPUT_DIR)

        print("Existing dataset removed.")


def split_images(images):
    """
    Randomly split images into:
        80% training
        10% validation
        10% testing
    """

    images = images.copy()

    random.shuffle(images)

    total_images = len(images)

    train_count = int(
        total_images * TRAIN_RATIO
    )

    validation_count = int(
        total_images * VALIDATION_RATIO
    )

    train_images = images[
        :train_count
    ]

    validation_images = images[
        train_count:
        train_count + validation_count
    ]

    test_images = images[
        train_count + validation_count:
    ]

    return (
        train_images,
        validation_images,
        test_images
    )


def copy_images(images, destination):
    """
    Copy images to the destination folder.
    """

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    for image in images:

        destination_file = (
            destination / image.name
        )

        shutil.copy2(
            image,
            destination_file
        )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print()
    print("=" * 65)
    print("       PAWCARE DOG BREED DATASET SPLITTER")
    print("=" * 65)

    print("\nSource dataset:")
    print(SOURCE_DIR)

    print("\nOutput dataset:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # Check source dataset
    # --------------------------------------------------------

    if not SOURCE_DIR.exists():

        print("\nERROR: Source dataset was not found.")

        print("\nExpected location:")
        print(SOURCE_DIR)

        return

    # --------------------------------------------------------
    # Check train and validation folders
    # --------------------------------------------------------

    original_train_dir = (
        SOURCE_DIR / "train"
    )

    original_validation_dir = (
        SOURCE_DIR / "validation"
    )

    if not original_train_dir.exists():

        print(
            "\nERROR: Original train folder "
            "was not found."
        )

        print(original_train_dir)

        return

    if not original_validation_dir.exists():

        print(
            "\nERROR: Original validation folder "
            "was not found."
        )

        print(original_validation_dir)

        return

    # --------------------------------------------------------
    # Remove previously generated dataset
    # --------------------------------------------------------

    clear_existing_dataset()

    # --------------------------------------------------------
    # Create new folder structure
    # --------------------------------------------------------

    create_output_folders()

    print("\nNew dataset folders created.")

    print()
    print("=" * 65)
    print("PROCESSING DOG BREEDS")
    print("=" * 65)

    # --------------------------------------------------------
    # Set random seed
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    # --------------------------------------------------------
    # Overall counters
    # --------------------------------------------------------

    total_original = 0
    total_train = 0
    total_validation = 0
    total_test = 0

    # ========================================================
    # PROCESS EACH BREED
    # ========================================================

    for breed in BREEDS:

        print()
        print("-" * 65)
        print(f"Breed: {breed}")
        print("-" * 65)

        # ----------------------------------------------------
        # Original train folder
        # ----------------------------------------------------

        breed_train_dir = (
            original_train_dir / breed
        )

        # ----------------------------------------------------
        # Original validation folder
        # ----------------------------------------------------

        breed_validation_dir = (
            original_validation_dir / breed
        )

        # ----------------------------------------------------
        # Check breed folders
        # ----------------------------------------------------

        if not breed_train_dir.exists():

            print(
                f"ERROR: Train folder missing for {breed}"
            )

            return

        if not breed_validation_dir.exists():

            print(
                f"ERROR: Validation folder missing for {breed}"
            )

            return

        # ----------------------------------------------------
        # Read images
        # ----------------------------------------------------

        train_images = get_images(
            breed_train_dir
        )

        validation_images = get_images(
            breed_validation_dir
        )

        # ----------------------------------------------------
        # Combine original train + validation
        # ----------------------------------------------------

        all_images = (
            train_images
            + validation_images
        )

        print(
            f"Original train images: "
            f"{len(train_images)}"
        )

        print(
            f"Original validation images: "
            f"{len(validation_images)}"
        )

        print(
            f"Total images available: "
            f"{len(all_images)}"
        )

        # ----------------------------------------------------
        # Check image count
        # ----------------------------------------------------

        if len(all_images) == 0:

            print(
                f"ERROR: No images found for {breed}"
            )

            return

        # ----------------------------------------------------
        # Split dataset
        # ----------------------------------------------------

        (
            new_train,
            new_validation,
            new_test
        ) = split_images(all_images)

        print(
            f"New training images: "
            f"{len(new_train)}"
        )

        print(
            f"New validation images: "
            f"{len(new_validation)}"
        )

        print(
            f"New test images: "
            f"{len(new_test)}"
        )

        # ----------------------------------------------------
        # Destination folders
        # ----------------------------------------------------

        train_destination = (
            OUTPUT_DIR
            / "train"
            / breed
        )

        validation_destination = (
            OUTPUT_DIR
            / "validation"
            / breed
        )

        test_destination = (
            OUTPUT_DIR
            / "test"
            / breed
        )

        # ----------------------------------------------------
        # Copy images
        # ----------------------------------------------------

        copy_images(
            new_train,
            train_destination
        )

        copy_images(
            new_validation,
            validation_destination
        )

        copy_images(
            new_test,
            test_destination
        )

        # ----------------------------------------------------
        # Update counters
        # ----------------------------------------------------

        total_original += len(all_images)

        total_train += len(new_train)

        total_validation += len(
            new_validation
        )

        total_test += len(new_test)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_images = (
        total_train
        + total_validation
        + total_test
    )

    print()
    print("=" * 65)
    print("          DATASET SPLIT COMPLETED")
    print("=" * 65)

    print()
    print(f"Total original images : {total_original}")

    print(
        f"Total training images : {total_train}"
    )

    print(
        f"Total validation      : {total_validation}"
    )

    print(
        f"Total test images     : {total_test}"
    )

    print(
        f"Total final images    : {total_images}"
    )

    print()
    print("Output location:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # Expected percentage
    # --------------------------------------------------------

    if total_images > 0:

        train_percentage = (
            total_train / total_images
        ) * 100

        validation_percentage = (
            total_validation / total_images
        ) * 100

        test_percentage = (
            total_test / total_images
        ) * 100

        print()
        print("Final split percentages:")

        print(
            f"Train      : "
            f"{train_percentage:.2f}%"
        )

        print(
            f"Validation : "
            f"{validation_percentage:.2f}%"
        )

        print(
            f"Test       : "
            f"{test_percentage:.2f}%"
        )

    # ========================================================
    # FINAL DIRECTORY STRUCTURE
    # ========================================================

    print()
    print("=" * 65)
    print("FINAL DATASET STRUCTURE")
    print("=" * 65)

    print(
        """
dog_breed/
│
└── dataset/
    │
    ├── train/
    │   ├── beagle/
    │   ├── boxer/
    │   ├── German_shepherd/
    │   ├── golden_retriever/
    │   ├── pug/
    │   ├── shitzu/
    │   ├── Siberian_husky/
    │   ├── standard_poodle/
    │   ├── whippet/
    │   └── Yorkshire_terrier/
    │
    ├── validation/
    │   └── same 10 breeds
    │
    └── test/
        └── same 10 breeds
"""
    )

    print("=" * 65)
    print("DOG DATASET PREPARATION FINISHED SUCCESSFULLY!")
    print("=" * 65)
    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()