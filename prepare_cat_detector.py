import os
import shutil
import random

# --------------------------------------------------
# PATHS
# --------------------------------------------------

SOURCE_DIR = r"C:\PawCareData\cat_breed_split"
DEST_DIR = r"C:\PawCareData\cat_detector"

# Reproducible split
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

SPLIT_RATIOS = {
    "train": 0.80,
    "validation": 0.10,
    "test": 0.10
}

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)

# --------------------------------------------------
# COLLECT ALL CAT IMAGES
# --------------------------------------------------

print("=" * 60)
print("PAWCARE CAT DETECTOR - PREPARING CAT IMAGES")
print("=" * 60)

all_images = []

for split in ["train", "validation", "test"]:

    split_path = os.path.join(SOURCE_DIR, split)

    if not os.path.exists(split_path):
        print(f"ERROR: Missing folder: {split_path}")
        exit()

    for breed in os.listdir(split_path):

        breed_path = os.path.join(split_path, breed)

        if not os.path.isdir(breed_path):
            continue

        for filename in os.listdir(breed_path):

            if filename.lower().endswith(IMAGE_EXTENSIONS):

                image_path = os.path.join(
                    breed_path,
                    filename
                )

                all_images.append(image_path)

print(f"\nTotal cat images found: {len(all_images)}")

# --------------------------------------------------
# SHUFFLE
# --------------------------------------------------

random.shuffle(all_images)

# --------------------------------------------------
# CALCULATE SPLIT
# --------------------------------------------------

total = len(all_images)

train_end = int(total * SPLIT_RATIOS["train"])

validation_end = train_end + int(
    total * SPLIT_RATIOS["validation"]
)

train_images = all_images[:train_end]

validation_images = all_images[
    train_end:validation_end
]

test_images = all_images[
    validation_end:
]

# --------------------------------------------------
# COPY FUNCTION
# --------------------------------------------------

def copy_images(images, split):

    destination = os.path.join(
        DEST_DIR,
        split,
        "cat"
    )

    os.makedirs(
        destination,
        exist_ok=True
    )

    copied = 0

    for index, source_path in enumerate(images):

        extension = os.path.splitext(
            source_path
        )[1].lower()

        new_filename = (
            f"cat_{index:04d}{extension}"
        )

        destination_path = os.path.join(
            destination,
            new_filename
        )

        shutil.copy2(
            source_path,
            destination_path
        )

        copied += 1

    return copied


# --------------------------------------------------
# COPY DATA
# --------------------------------------------------

train_count = copy_images(
    train_images,
    "train"
)

validation_count = copy_images(
    validation_images,
    "validation"
)

test_count = copy_images(
    test_images,
    "test"
)

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("CAT IMAGES COPIED SUCCESSFULLY")
print("=" * 60)

print(f"\nTraining cat images:    {train_count}")
print(f"Validation cat images:  {validation_count}")
print(f"Test cat images:        {test_count}")

print(
    f"\nTotal copied:            "
    f"{train_count + validation_count + test_count}"
)

print("\nDestination:")
print(DEST_DIR)

print("\nIMPORTANT:")
print("Original cat breed dataset was NOT modified.")
print("Only copies were created.")