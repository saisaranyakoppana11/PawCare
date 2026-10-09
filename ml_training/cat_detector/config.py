from pathlib import Path


# ============================================================
# PAWCARE - CAT VS NON-CAT DETECTOR CONFIGURATION
# ============================================================


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(
    r"C:\Users\saisa\OneDrive\Documents\PawCare"
)


# ------------------------------------------------------------
# DATASET PATHS
# ------------------------------------------------------------

DATASET_DIR = Path(
    r"C:\PawCareData\cat_detector"
)

TRAIN_DIR = DATASET_DIR / "train"

VALIDATION_DIR = DATASET_DIR / "validation"

TEST_DIR = DATASET_DIR / "test"


# ------------------------------------------------------------
# OUTPUT PATHS
# ------------------------------------------------------------

ML_TRAINING_DIR = (
    PROJECT_ROOT /
    "ml_training" /
    "cat_detector"
)

RESULTS_DIR = (
    ML_TRAINING_DIR /
    "results"
)

TRAINED_MODEL_DIR = (
    ML_TRAINING_DIR /
    "trained_model"
)


# ------------------------------------------------------------
# CLASS CONFIGURATION
# ------------------------------------------------------------

EXPECTED_CLASSES = [
    "cat",
    "non_cat"
]

NUM_CLASSES = 2


# ------------------------------------------------------------
# IMAGE CONFIGURATION
# ------------------------------------------------------------

IMAGE_HEIGHT = 224

IMAGE_WIDTH = 224

CHANNELS = 3

IMAGE_SIZE = (
    IMAGE_HEIGHT,
    IMAGE_WIDTH
)


# ------------------------------------------------------------
# BATCH / TRAINING CONFIGURATION
# ------------------------------------------------------------

BATCH_SIZE = 8

EPOCHS = 20

LEARNING_RATE = 0.001

DROPOUT_RATE = 0.30


# ------------------------------------------------------------
# MODEL CONFIGURATION
# ------------------------------------------------------------

MODEL_NAME = "PawCare_Cat_Detector"

BASE_MODEL_NAME = "EfficientNetB0"

BASE_MODEL_WEIGHTS = (
    PROJECT_ROOT /
    "models" /
    "efficientnetb0_notop.h5"
)

FREEZE_BASE_MODEL = True

POOLING = "global_average"


# ------------------------------------------------------------
# DATA AUGMENTATION CONFIGURATION
#
# These names match the variables used by the existing
# cat_detector/data_loader.py
# ------------------------------------------------------------

USE_AUGMENTATION = True

HORIZONTAL_FLIP = True

ROTATION_FACTOR = 0.05

ZOOM_FACTOR = 0.10

TRANSLATION_HEIGHT = 0.05

TRANSLATION_WIDTH = 0.05


# ------------------------------------------------------------
# OPTIMIZER CONFIGURATION
# ------------------------------------------------------------

OPTIMIZER_NAME = "Adam"


# ------------------------------------------------------------
# CALLBACK CONFIGURATION
# ------------------------------------------------------------

MONITOR = "val_accuracy"

MONITOR_MODE = "max"

PATIENCE = 5

RESTORE_BEST_WEIGHTS = True


# ------------------------------------------------------------
# DATA LOADING CONFIGURATION
# ------------------------------------------------------------

RANDOM_SEED = 42

SHUFFLE_TRAIN = True

SHUFFLE_VALIDATION = False

SHUFFLE_TEST = False

PREFETCH_BUFFER = 1


# ============================================================
# CONFIGURATION TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("PAWCARE - CAT VS NON-CAT DETECTOR CONFIGURATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Project
    # --------------------------------------------------------

    print("\nPROJECT")
    print("-" * 70)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    if PROJECT_ROOT.exists():
        print("✓ Project directory found")
    else:
        print("✗ Project directory not found")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nDATASET")
    print("-" * 70)

    print(
        f"Dataset directory : {DATASET_DIR}"
    )

    if DATASET_DIR.exists():
        print("✓ Dataset directory found")
    else:
        print("✗ Dataset directory not found")

    print(
        f"\nTrain directory      : {TRAIN_DIR}"
    )

    print(
        f"Validation directory : {VALIDATION_DIR}"
    )

    print(
        f"Test directory       : {TEST_DIR}"
    )

    if TRAIN_DIR.exists():
        print("✓ Train directory found")
    else:
        print("✗ Train directory not found")

    if VALIDATION_DIR.exists():
        print("✓ Validation directory found")
    else:
        print("✗ Validation directory not found")

    if TEST_DIR.exists():
        print("✓ Test directory found")
    else:
        print("✗ Test directory not found")

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    print("\nCLASSES")
    print("-" * 70)

    print(
        f"Number of classes : {NUM_CLASSES}"
    )

    print(
        f"Classes           : {EXPECTED_CLASSES}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nMODEL")
    print("-" * 70)

    print(
        f"Base model : {BASE_MODEL_NAME}"
    )

    print(
        f"Model name : {MODEL_NAME}"
    )

    print(
        f"Image size : "
        f"{IMAGE_HEIGHT} x {IMAGE_WIDTH}"
    )

    print(
        f"Channels   : {CHANNELS}"
    )

    print(
        f"\nLocal weights:\n"
        f"{BASE_MODEL_WEIGHTS}"
    )

    if BASE_MODEL_WEIGHTS.exists():

        print(
            "✓ Local EfficientNetB0 weights found"
        )

        print(
            f"  File size: "
            f"{BASE_MODEL_WEIGHTS.stat().st_size:,} bytes"
        )

    else:

        print(
            "✗ Local EfficientNetB0 weights not found"
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nTRAINING")
    print("-" * 70)

    print(
        f"Batch size    : {BATCH_SIZE}"
    )

    print(
        f"Epochs        : {EPOCHS}"
    )

    print(
        f"Learning rate : {LEARNING_RATE}"
    )

    print(
        f"Dropout       : {DROPOUT_RATE}"
    )

    print(
        f"Optimizer     : {OPTIMIZER_NAME}"
    )

    print(
        f"Freeze base   : {FREEZE_BASE_MODEL}"
    )

    print(
        f"Monitor       : {MONITOR}"
    )

    print(
        f"Patience      : {PATIENCE}"
    )

    # --------------------------------------------------------
    # Augmentation
    # --------------------------------------------------------

    print("\nDATA AUGMENTATION")
    print("-" * 70)

    print(
        f"Use augmentation    : {USE_AUGMENTATION}"
    )

    print(
        f"Horizontal flip     : {HORIZONTAL_FLIP}"
    )

    print(
        f"Rotation factor     : {ROTATION_FACTOR}"
    )

    print(
        f"Zoom factor         : {ZOOM_FACTOR}"
    )

    print(
        f"Translation height  : {TRANSLATION_HEIGHT}"
    )

    print(
        f"Translation width   : {TRANSLATION_WIDTH}"
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    print("\nOUTPUT DIRECTORIES")
    print("-" * 70)

    print(
        f"Results        : {RESULTS_DIR}"
    )

    print(
        f"Trained models : {TRAINED_MODEL_DIR}"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TRAINED_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\n✓ Results directory ready"
    )

    print(
        "✓ Trained model directory ready"
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "STATUS: CONFIGURATION TEST COMPLETED"
    )
    print("=" * 70)