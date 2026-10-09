import os
import shutil
import random


# ============================================================
# CONFIGURATION
# ============================================================

# Original 500 non-dog images
NON_DOG_SOURCE = r"C:\Users\saisa\Downloads\non_dog dataset"

# Already prepared 500 dog images
DOG_SOURCE = r"C:\Users\saisa\OneDrive\Documents\PawCare\dog_breed\dataset"

# Output detector dataset
OUTPUT_DIR = r"C:\Users\saisa\OneDrive\Documents\PawCare\dog_breed\dog_detector\dataset"

# Reproducibility
RANDOM_SEED = 42

# Image extensions
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# FUNCTIONS
# ============================================================

def get_images_recursive(folder):
    """
    Find all image files inside a folder and its subfolders.
    """

    images = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            extension = os.path.splitext(file)[1].lower()

            if extension in IMAGE_EXTENSIONS:

                images.append(
                    os.path.join(root, file)
                )

    return images


def clean_output_directory():

    if os.path.exists(OUTPUT_DIR):

        print("\nRemoving previous detector dataset...")

        shutil.rmtree(OUTPUT_DIR)

    print("Creating detector dataset folders...")


def create_directories():

    for split in ["train", "validation", "test"]:

        for category in ["dog", "non_dog"]:

            path = os.path.join(
                OUTPUT_DIR,
                split,
                category
            )

            os.makedirs(
                path,
                exist_ok=True
            )


def copy_images(images, destination):

    for index, source_path in enumerate(images):

        extension = os.path.splitext(source_path)[1].lower()

        filename = f"{index + 1:04d}{extension}"

        destination_path = os.path.join(
            destination,
            filename
        )

        shutil.copy2(
            source_path,
            destination_path
        )


def split_images(images):

    random.shuffle(images)

    total = len(images)

    train_count = int(total * 0.80)
    validation_count = int(total * 0.10)

    train_images = images[:train_count]

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


# ============================================================
# START
# ============================================================

print("=" * 60)
print("PAWCARE DOG VS NON-DOG DATASET PREPARATION")
print("=" * 60)


# ============================================================
# CHECK SOURCE DIRECTORIES
# ============================================================

print("\nChecking source directories...")

if not os.path.exists(NON_DOG_SOURCE):

    raise FileNotFoundError(
        f"Non-dog dataset not found:\n{NON_DOG_SOURCE}"
    )

if not os.path.exists(DOG_SOURCE):

    raise FileNotFoundError(
        f"Dog dataset not found:\n{DOG_SOURCE}"
    )

print("Non-dog source found.")
print("Dog source found.")


# ============================================================
# GET NON-DOG IMAGES
# ============================================================

print("\nCollecting non-dog images...")

non_dog_images = get_images_recursive(
    NON_DOG_SOURCE
)

print(
    f"Non-dog images found: {len(non_dog_images)}"
)


# ============================================================
# GET DOG IMAGES
# ============================================================

print("\nCollecting dog images...")

dog_images = get_images_recursive(
    DOG_SOURCE
)

print(
    f"Dog images found: {len(dog_images)}"
)


# ============================================================
# VERIFY COUNTS
# ============================================================

if len(non_dog_images) != 500:

    raise ValueError(
        f"Expected 500 non-dog images, "
        f"but found {len(non_dog_images)}."
    )

if len(dog_images) != 500:

    raise ValueError(
        f"Expected 500 dog images, "
        f"but found {len(dog_images)}."
    )


print("\nSource dataset verification passed!")
print("Dog images     :", len(dog_images))
print("Non-dog images :", len(non_dog_images))
print("Total images   :", len(dog_images) + len(non_dog_images))


# ============================================================
# SHUFFLE
# ============================================================

random.seed(RANDOM_SEED)

random.shuffle(dog_images)
random.shuffle(non_dog_images)


# ============================================================
# SPLIT DATA
# ============================================================

dog_train, dog_validation, dog_test = split_images(
    dog_images
)

non_dog_train, non_dog_validation, non_dog_test = split_images(
    non_dog_images
)


# ============================================================
# PRINT SPLIT COUNTS
# ============================================================

print("\nDataset split:")

print("\nDOG")
print("Train      :", len(dog_train))
print("Validation :", len(dog_validation))
print("Test       :", len(dog_test))

print("\nNON-DOG")
print("Train      :", len(non_dog_train))
print("Validation :", len(non_dog_validation))
print("Test       :", len(non_dog_test))


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

clean_output_directory()

create_directories()


# ============================================================
# COPY DOG IMAGES
# ============================================================

print("\nCopying dog images...")

copy_images(
    dog_train,
    os.path.join(
        OUTPUT_DIR,
        "train",
        "dog"
    )
)

copy_images(
    dog_validation,
    os.path.join(
        OUTPUT_DIR,
        "validation",
        "dog"
    )
)

copy_images(
    dog_test,
    os.path.join(
        OUTPUT_DIR,
        "test",
        "dog"
    )
)


# ============================================================
# COPY NON-DOG IMAGES
# ============================================================

print("Copying non-dog images...")

copy_images(
    non_dog_train,
    os.path.join(
        OUTPUT_DIR,
        "train",
        "non_dog"
    )
)

copy_images(
    non_dog_validation,
    os.path.join(
        OUTPUT_DIR,
        "validation",
        "non_dog"
    )
)

copy_images(
    non_dog_test,
    os.path.join(
        OUTPUT_DIR,
        "test",
        "non_dog"
    )
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET VERIFICATION")
print("=" * 60)


for split in ["train", "validation", "test"]:

    print(f"\n{split.upper()}")

    for category in ["dog", "non_dog"]:

        folder = os.path.join(
            OUTPUT_DIR,
            split,
            category
        )

        count = len(
            get_images_recursive(folder)
        )

        print(
            f"{category:8s}: {count}"
        )


print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nDataset location:")
print(OUTPUT_DIR)