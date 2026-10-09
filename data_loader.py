import os
import tensorflow as tf

# ============================================================
# PAWCARE - CAT BREED DATA LOADER
# ============================================================

print("\n" + "=" * 60)
print("PAWCARE - CAT BREED DATA LOADER")
print("=" * 60)

# ============================================================
# CONFIGURATION
# ============================================================

# Dataset is stored outside OneDrive
DATA_DIR = r"C:\PawCareData\cat_breed_split"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation")
TEST_DIR = os.path.join(DATA_DIR, "test")

IMG_SIZE = 224

# Using a smaller batch size to reduce Windows memory/file-resource pressure
BATCH_SIZE = 8

SEED = 42

# ============================================================
# CHECK DATASET PATHS
# ============================================================

print("\nChecking dataset folders...")

print("\nDataset location:")
print(DATA_DIR)

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(
        f"\nDataset folder was not found:\n{DATA_DIR}\n\n"
        "Please make sure the dataset was copied to:\n"
        "C:\\PawCareData\\cat_breed_split"
    )

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"\nTraining folder was not found:\n{TRAIN_DIR}"
    )

if not os.path.exists(VALIDATION_DIR):
    raise FileNotFoundError(
        f"\nValidation folder was not found:\n{VALIDATION_DIR}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"\nTest folder was not found:\n{TEST_DIR}"
    )

print("Dataset folders found successfully!")

# ============================================================
# LOAD TRAINING DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING TRAINING DATA")
print("=" * 60)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

# ============================================================
# GET CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nClasses detected:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")

print(f"\nNumber of classes: {len(class_names)}")

# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING VALIDATION DATA")
print("=" * 60)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels="inferred",
    label_mode="int",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# VERIFY CLASS ORDER
# ============================================================

print("\n" + "=" * 60)
print("VERIFYING CLASS NAMES")
print("=" * 60)

if validation_dataset.class_names != class_names:
    raise ValueError(
        "\nTraining and validation class names do not match!"
    )

if test_dataset.class_names != class_names:
    raise ValueError(
        "\nTraining and test class names do not match!"
    )

print("Training classes   : OK")
print("Validation classes : OK")
print("Test classes       : OK")

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.05
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        )
    ],
    name="data_augmentation"
)

print("\nData augmentation:")
print("- Random horizontal flip")
print("- Small rotation")
print("- Small zoom")
print("- Small translation")

# ============================================================
# APPLY AUGMENTATION TO TRAINING DATA
# ============================================================

def augment_images(images, labels):
    images = data_augmentation(
        images,
        training=True
    )

    return images, labels


train_dataset = train_dataset.map(
    augment_images,
    num_parallel_calls=1
)

# ============================================================
# IMPORTANT:
# DO NOT USE Rescaling(1./255) HERE
# ============================================================
#
# EfficientNetB0 already contains its own input
# preprocessing and expects pixel values in the
# 0-255 range.
#
# Therefore, we leave the images in their original
# pixel-value range.
# ============================================================

print("\nEfficientNetB0 preprocessing:")
print("- No manual Rescaling layer")
print("- Images remain in 0-255 range")

# ============================================================
# PREFETCH
# ============================================================
#
# Using a small/simple pipeline instead of AUTOTUNE
# to reduce Windows file-resource pressure.
# ============================================================

train_dataset = train_dataset.prefetch(1)

validation_dataset = validation_dataset.prefetch(1)

test_dataset = test_dataset.prefetch(1)

# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nImage size:")
print(f"{IMG_SIZE} x {IMG_SIZE}")

print("\nBatch size:")
print(BATCH_SIZE)

print("\nNumber of classes:")
print(len(class_names))

# ============================================================
# COUNT DATASET IMAGES
# ============================================================

def count_images(folder):
    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif"
    )

    count = 0

    for root, directories, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                count += 1

    return count


train_count = count_images(TRAIN_DIR)

validation_count = count_images(VALIDATION_DIR)

test_count = count_images(TEST_DIR)

print("\nImage counts:")

print(f"Training   : {train_count}")

print(f"Validation : {validation_count}")

print(f"Test       : {test_count}")

print(f"Total      : {train_count + validation_count + test_count}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DATA LOADER READY")
print("=" * 60)

print("\nDataset path:")
print(DATA_DIR)

print("\nClasses:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")

print("\nTraining images   :", train_count)

print("Validation images :", validation_count)

print("Test images       :", test_count)

print("\nBatch size        :", BATCH_SIZE)

print("Image size        :", f"{IMG_SIZE} x {IMG_SIZE}")

print("\nData augmentation : ENABLED")

print("EfficientNet preprocessing : ENABLED")

print("\n" + "=" * 60)
print("PAWCARE DATA LOADER COMPLETED")
print("=" * 60)