from pathlib import Path


# ============================================================
# PawCare - Dog vs Non-Dog Detector
# Training Configuration
# ============================================================


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(
    r"C:\Users\saisa\OneDrive\Documents\PawCare"
)


DATASET_DIR = (
    PROJECT_ROOT
    / "dog_breed"
    / "dog_detector"
    / "dataset"
)


TRAIN_DIR = DATASET_DIR / "train"

VALIDATION_DIR = DATASET_DIR / "validation"

TEST_DIR = DATASET_DIR / "test"


# ------------------------------------------------------------
# ML training directories
# ------------------------------------------------------------

ML_TRAINING_DIR = (
    PROJECT_ROOT
    / "ml_training"
    / "dog_detector"
)


RESULTS_DIR = (
    ML_TRAINING_DIR
    / "results"
)


TRAINED_MODEL_DIR = (
    ML_TRAINING_DIR
    / "trained_model"
)


# ------------------------------------------------------------
# Classes
# ------------------------------------------------------------

EXPECTED_CLASSES = [
    "dog",
    "non_dog"
]


NUM_CLASSES = 2


# ------------------------------------------------------------
# Image configuration
# ------------------------------------------------------------

IMAGE_HEIGHT = 224

IMAGE_WIDTH = 224

CHANNELS = 3

IMAGE_SIZE = (
    IMAGE_HEIGHT,
    IMAGE_WIDTH
)


# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

BATCH_SIZE = 8

EPOCHS = 20

LEARNING_RATE = 0.001

DROPOUT_RATE = 0.30


# ------------------------------------------------------------
# EfficientNetB0 configuration
# ------------------------------------------------------------

MODEL_NAME = "PawCare_Dog_Detector"

BASE_MODEL_NAME = "EfficientNetB0"


# Use the local EfficientNetB0 weights already available
# in the PawCare project.

BASE_MODEL_WEIGHTS = (
    PROJECT_ROOT
    / "models"
    / "efficientnetb0_notop.h5"
)


FREEZE_BASE_MODEL = True

POOLING = "global_average"


# ------------------------------------------------------------
# Data augmentation
# ------------------------------------------------------------

USE_AUGMENTATION = True

HORIZONTAL_FLIP = True

ROTATION_FACTOR = 0.05

ZOOM_FACTOR = 0.10

TRANSLATION_HEIGHT = 0.05

TRANSLATION_WIDTH = 0.05


# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

OPTIMIZER_NAME = "Adam"


# ------------------------------------------------------------
# Callback configuration
# ------------------------------------------------------------

MONITOR = "val_accuracy"

MONITOR_MODE = "max"

PATIENCE = 5

RESTORE_BEST_WEIGHTS = True


# ------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------

RANDOM_SEED = 42

SHUFFLE_TRAIN = True

SHUFFLE_VALIDATION = False

SHUFFLE_TEST = False

PREFETCH_BUFFER = 1


# ------------------------------------------------------------
# Configuration test
# ------------------------------------------------------------

def test_configuration():

    print("=" * 70)
    print("PAWCARE - DOG DETECTOR CONFIGURATION TEST")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    if PROJECT_ROOT.exists():
        print("✓ Project root found")
    else:
        print("ERROR: Project root not found")
        return False

    print("\nDataset:")
    print(DATASET_DIR)

    if DATASET_DIR.exists():
        print("✓ Dataset found")
    else:
        print("ERROR: Dataset not found")
        return False

    print("\nChecking dataset splits...")

    for split_name, split_dir in [
        ("train", TRAIN_DIR),
        ("validation", VALIDATION_DIR),
        ("test", TEST_DIR),
    ]:

        if split_dir.exists():
            print(f"✓ {split_name} directory found")
        else:
            print(
                f"ERROR: {split_name} directory not found"
            )
            return False

    print("\nChecking EfficientNetB0 weights...")

    if BASE_MODEL_WEIGHTS.exists():

        file_size = BASE_MODEL_WEIGHTS.stat().st_size

        print(
            f"✓ Local weights found"
        )

        print(
            f"  Path: {BASE_MODEL_WEIGHTS}"
        )

        print(
            f"  Size: {file_size:,} bytes"
        )

    else:

        print(
            "ERROR: EfficientNetB0 local weights not found"
        )

        print(
            f"Expected path: {BASE_MODEL_WEIGHTS}"
        )

        return False

    print("\nModel configuration:")

    print(
        f"  Model: {MODEL_NAME}"
    )

    print(
        f"  Base model: {BASE_MODEL_NAME}"
    )

    print(
        f"  Image size: {IMAGE_HEIGHT} x {IMAGE_WIDTH}"
    )

    print(
        f"  Channels: {CHANNELS}"
    )

    print(
        f"  Classes: {EXPECTED_CLASSES}"
    )

    print(
        f"  Number of classes: {NUM_CLASSES}"
    )

    print(
        f"  Batch size: {BATCH_SIZE}"
    )

    print(
        f"  Epochs: {EPOCHS}"
    )

    print(
        f"  Learning rate: {LEARNING_RATE}"
    )

    print(
        f"  Dropout: {DROPOUT_RATE}"
    )

    print(
        f"  Optimizer: {OPTIMIZER_NAME}"
    )

    print(
        f"  Frozen base model: {FREEZE_BASE_MODEL}"
    )

    print("\nAugmentation:")

    print(
        f"  Enabled: {USE_AUGMENTATION}"
    )

    print(
        f"  Horizontal flip: {HORIZONTAL_FLIP}"
    )

    print(
        f"  Rotation: {ROTATION_FACTOR}"
    )

    print(
        f"  Zoom: {ZOOM_FACTOR}"
    )

    print(
        f"  Translation height: {TRANSLATION_HEIGHT}"
    )

    print(
        f"  Translation width: {TRANSLATION_WIDTH}"
    )

    print("\nCallbacks:")

    print(
        f"  Monitor: {MONITOR}"
    )

    print(
        f"  Mode: {MONITOR_MODE}"
    )

    print(
        f"  Patience: {PATIENCE}"
    )

    print(
        f"  Restore best weights: {RESTORE_BEST_WEIGHTS}"
    )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TRAINED_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nOutput directories:")

    print(
        f"✓ Results: {RESULTS_DIR}"
    )

    print(
        f"✓ Trained model: {TRAINED_MODEL_DIR}"
    )

    print("\n" + "=" * 70)
    print("STATUS: CONFIGURATION TEST PASSED")
    print("=" * 70)

    return True


if __name__ == "__main__":

    success = test_configuration()

    if not success:
        raise SystemExit(1)