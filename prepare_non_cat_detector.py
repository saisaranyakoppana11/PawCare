import os
import shutil
import random

# ============================================================
# PATHS
# ============================================================

SOURCE_DIR = r"C:\Users\saisa\Downloads\non_cat dataset"

DEST_DIR = r"C:\PawCareData\cat_detector"

# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

TRAIN_COUNT = 842
VALIDATION_COUNT = 105
TEST_COUNT = 106

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

random.seed(RANDOM_SEED)

# ============================================================
# CHECK SOURCE
# ============================================================

print("=" * 60)
print("PAWCARE NON-CAT DATASET PREPARATION")
print("=" * 60)

if not os.path.exists(SOURCE_DIR):

    print("\nERROR: Source dataset not found!")
    print(SOURCE_DIR)
    exit()

print("\nSource dataset:")
print(SOURCE_DIR)

# ============================================================
# COLLECT ALL IMAGES
# ============================================================

all_images = []

for root, folders, files in os.walk(SOURCE_DIR):

    for filename in files:

        if filename.lower().endswith(IMAGE_EXTENSIONS):

            image_path = os.path.join(
                root,
                filename
            )

            all_images.append(image_path)

print(f"\nTotal images found: {len(all_images)}")

# ============================================================
# CHECK EXACT COUNT
# ============================================================

EXPECTED_TOTAL = (
    TRAIN_COUNT +
    VALIDATION_COUNT +
    TEST_COUNT
)

print(f"Expected images:   {EXPECTED_TOTAL}")
print(f"Found images:      {len(all_images)}")

if len(all_images) != EXPECTED_TOTAL:

    print("\nERROR:")
    print(
        f"We expected exactly {EXPECTED_TOTAL} images "
        f"but found {len(all_images)}."
    )

    print("\nPlease check the dataset before continuing.")
    exit()

print("\nImage count is correct!")

# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(all_images)

# ============================================================
# SPLIT
# ============================================================

train_images = all_images[
    :TRAIN_COUNT
]

validation_images = all_images[
    TRAIN_COUNT:
    TRAIN_COUNT + VALIDATION_COUNT
]

test_images = all_images[
    TRAIN_COUNT + VALIDATION_COUNT:
]

# ============================================================
# COPY FUNCTION
# ============================================================

def copy_images(images, split):

    destination = os.path.join(
        DEST_DIR,
        split,
        "non_cat"
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
            f"non_cat_{index:04d}{extension}"
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

# ============================================================
# COPY IMAGES
# ============================================================

print("\nCopying training images...")

train_count = copy_images(
    train_images,
    "train"
)

print("Copying validation images...")

validation_count = copy_images(
    validation_images,
    "validation"
)

print("Copying test images...")

test_count = copy_images(
    test_images,
    "test"
)

# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("NON-CAT DATASET PREPARED SUCCESSFULLY")
print("=" * 60)

print("\nTrain:")
print(f"  Cat:     842")
print(f"  Non-Cat: {train_count}")

print("\nValidation:")
print(f"  Cat:     105")
print(f"  Non-Cat: {validation_count}")

print("\nTest:")
print(f"  Cat:     106")
print(f"  Non-Cat: {test_count}")

print("\nTotal Non-Cat images copied:")
print(
    train_count +
    validation_count +
    test_count
)

print("\nDestination:")
print(DEST_DIR)

print("\nOriginal dataset was NOT modified.")
print("Only copies were created.")

print("\n" + "=" * 60)
print("NEXT STEP: DATASET VERIFICATION")
print("=" * 60)