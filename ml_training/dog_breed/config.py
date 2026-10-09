from pathlib import Path


# ============================================================
# PawCare - Dog Breed Classifier
# Training Configuration
# ============================================================


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\saisa\OneDrive\Documents\PawCare"
)


DATASET_DIR = (
    PROJECT_ROOT
    / "dog_breed"
    / "dataset"
)


TRAIN_DIR = (
    DATASET_DIR
    / "train"
)


VALIDATION_DIR = (
    DATASET_DIR
    / "validation"
)


TEST_DIR = (
    DATASET_DIR
    / "test"
)


ML_TRAINING_DIR = (
    PROJECT_ROOT
    / "ml_training"
    / "dog_breed"
)


RESULTS_DIR = (
    ML_TRAINING_DIR
    / "results"
)


TRAINED_MODEL_DIR = (
    ML_TRAINING_DIR
    / "trained_model"
)


# ============================================================
# Dataset Configuration
# ============================================================

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


NUM_CLASSES = 10


# ============================================================
# Image Configuration
# ============================================================

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 3

IMAGE_SIZE = (
    IMAGE_HEIGHT,
    IMAGE_WIDTH
)


# ============================================================
# Training Configuration
# ============================================================

BATCH_SIZE = 8

EPOCHS = 20

LEARNING_RATE = 0.001

DROPOUT_RATE = 0.30


# ============================================================
# Model Configuration
# ============================================================

MODEL_NAME = (
    "PawCare_Dog_Breed_Classifier"
)


BASE_MODEL_NAME = (
    "EfficientNetB0"
)


BASE_MODEL_WEIGHTS = (
    PROJECT_ROOT
    / "models"
    / "efficientnetb0_notop.h5"
)


FREEZE_BASE_MODEL = True


POOLING = (
    "global_average"
)


# ============================================================
# Data Augmentation
# ============================================================

USE_AUGMENTATION = True

HORIZONTAL_FLIP = True

ROTATION_FACTOR = 0.05

ZOOM_FACTOR = 0.10

TRANSLATION_HEIGHT = 0.05

TRANSLATION_WIDTH = 0.05


# ============================================================
# Optimizer
# ============================================================

OPTIMIZER_NAME = "Adam"


# ============================================================
# Callbacks
# ============================================================

MONITOR = "val_accuracy"

MONITOR_MODE = "max"

PATIENCE = 5

RESTORE_BEST_WEIGHTS = True


# ============================================================
# Reproducibility
# ============================================================

RANDOM_SEED = 42


# ============================================================
# Dataset Loading
# ============================================================

SHUFFLE_TRAIN = True

SHUFFLE_VALIDATION = False

SHUFFLE_TEST = False


# ============================================================
# Performance
# ============================================================

PREFETCH_BUFFER = 1


# ============================================================
# Configuration Test
# ============================================================

def test_configuration():

    print("=" * 70)
    print("PAWCARE - DOG BREED CLASSIFIER")
    print("CONFIGURATION TEST")
    print("=" * 70)

    print(
        f"\nProject root:\n"
        f"{PROJECT_ROOT}"
    )

    print(
        f"\nDataset directory:\n"
        f"{DATASET_DIR}"
    )

    print(
        f"\nTraining directory:\n"
        f"{TRAIN_DIR}"
    )

    print(
        f"\nValidation directory:\n"
        f"{VALIDATION_DIR}"
    )

    print(
        f"\nTest directory:\n"
        f"{TEST_DIR}"
    )

    print(
        f"\nResults directory:\n"
        f"{RESULTS_DIR}"
    )

    print(
        f"\nTrained model directory:\n"
        f"{TRAINED_MODEL_DIR}"
    )

    print(
        f"\nClasses:\n"
        f"{EXPECTED_CLASSES}"
    )

    print(
        f"\nNumber of classes: "
        f"{NUM_CLASSES}"
    )

    print(
        f"\nImage size: "
        f"{IMAGE_SIZE}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs: "
        f"{EPOCHS}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Dropout rate: "
        f"{DROPOUT_RATE}"
    )

    print(
        f"\nBase model: "
        f"{BASE_MODEL_NAME}"
    )

    print(
        f"Base model weights:\n"
        f"{BASE_MODEL_WEIGHTS}"
    )

    # --------------------------------------------------------
    # Verify paths
    # --------------------------------------------------------

    required_directories = [
        DATASET_DIR,
        TRAIN_DIR,
        VALIDATION_DIR,
        TEST_DIR,
        ML_TRAINING_DIR,
        RESULTS_DIR,
        TRAINED_MODEL_DIR
    ]

    for directory in required_directories:

        if not directory.exists():

            raise FileNotFoundError(
                f"\nRequired directory not found:\n"
                f"{directory}"
            )

    print(
        "\n✓ All required directories exist"
    )

    # --------------------------------------------------------
    # Verify local EfficientNet weights
    # --------------------------------------------------------

    if not BASE_MODEL_WEIGHTS.exists():

        raise FileNotFoundError(
            "\nEfficientNetB0 local weights "
            "not found:\n"
            f"{BASE_MODEL_WEIGHTS}"
        )

    weight_size = (
        BASE_MODEL_WEIGHTS.stat().st_size
    )

    print(
        f"\n✓ EfficientNetB0 local weights found"
    )

    print(
        f"  File size: "
        f"{weight_size:,} bytes"
    )

    # --------------------------------------------------------
    # Verify class count
    # --------------------------------------------------------

    if len(EXPECTED_CLASSES) != NUM_CLASSES:

        raise ValueError(
            "\nNUM_CLASSES does not match "
            "EXPECTED_CLASSES."
        )

    print(
        "\n✓ Class configuration is valid"
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STATUS: CONFIGURATION TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":

    test_configuration()