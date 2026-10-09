from pathlib import Path


# ============================================================
# PAWCARE - CAT VS NON-CAT DETECTOR CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\saisa\OneDrive\Documents\PawCare"
)

# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------
# IMPORTANT:
# This should point to the already prepared Cat Detector
# dataset containing:
#   cat
#   non_cat
#
# Expected structure:
#
# cat_detector_dataset/
# ├── train/
# │   ├── cat/
# │   └── non_cat/
# ├── validation/
# │   ├── cat/
# │   └── non_cat/
# └── test/
#     ├── cat/
#     └── non_cat/
#
# Update this path only if your actual dataset is located
# somewhere else.

DATASET_DIR = Path(
    r"C:\PawCareData\cat_detector_split"
)

TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"


# ------------------------------------------------------------
# OUTPUT DIRECTORIES
# ------------------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml_training"
    / "cat_detector"
)

RESULTS_DIR = OUTPUT_DIR / "results"

MODEL_DIR = OUTPUT_DIR / "trained_model"


# ------------------------------------------------------------
# OUTPUT FILES
# ------------------------------------------------------------

MODEL_PATH = (
    MODEL_DIR
    / "cat_detector.keras"
)

CLASS_NAMES_PATH = (
    MODEL_DIR
    / "class_names.json"
)

HISTORY_PATH = (
    RESULTS_DIR
    / "training_history.json"
)

CONFUSION_MATRIX_PATH = (
    RESULTS_DIR
    / "confusion_matrix.png"
)

CLASSIFICATION_REPORT_PATH = (
    RESULTS_DIR
    / "classification_report.txt"
)

TRAINING_CURVES_PATH = (
    RESULTS_DIR
    / "training_curves.png"
)

LOSS_CURVE_PATH = (
    RESULTS_DIR
    / "loss_curve.png"
)

MISCLASSIFICATIONS_PATH = (
    RESULTS_DIR
    / "misclassifications.txt"
)


# ------------------------------------------------------------
# IMAGE SETTINGS
# ------------------------------------------------------------

IMAGE_SIZE = 224
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 3


# ------------------------------------------------------------
# CLASS SETTINGS
# ------------------------------------------------------------

NUM_CLASSES = 2

EXPECTED_CLASSES = [
    "cat",
    "non_cat"
]


# ------------------------------------------------------------
# DATA SPLIT
# ------------------------------------------------------------

TRAIN_SPLIT = 0.80
VALIDATION_SPLIT = 0.10
TEST_SPLIT = 0.10

RANDOM_SEED = 42


# ------------------------------------------------------------
# TRAINING SETTINGS
# ------------------------------------------------------------

BATCH_SIZE = 8

EPOCHS = 20

LEARNING_RATE = 0.001

DROPOUT_RATE = 0.30

OPTIMIZER_NAME = "Adam"


# ------------------------------------------------------------
# BASE MODEL
# ------------------------------------------------------------

BASE_MODEL_NAME = "EfficientNetB0"

BASE_MODEL_WEIGHTS = (
    PROJECT_ROOT
    / "models"
    / "efficientnetb0_notop.h5"
)

FREEZE_BASE_MODEL = True

POOLING = "global_average"


# ------------------------------------------------------------
# DATA AUGMENTATION
# ------------------------------------------------------------

USE_AUGMENTATION = True

HORIZONTAL_FLIP = True

ROTATION_FACTOR = 0.05

ZOOM_FACTOR = 0.10

TRANSLATION_HEIGHT = 0.05

TRANSLATION_WIDTH = 0.05


# ------------------------------------------------------------
# DATASET PERFORMANCE SETTINGS
# ------------------------------------------------------------

SHUFFLE_TRAIN = True

SHUFFLE_VALIDATION = False

SHUFFLE_TEST = False

PREFETCH_SIZE = 1

NUM_PARALLEL_CALLS = 1


# ------------------------------------------------------------
# CALLBACK SETTINGS
# ------------------------------------------------------------

USE_EARLY_STOPPING = True

EARLY_STOPPING_PATIENCE = 5

RESTORE_BEST_WEIGHTS = True

USE_MODEL_CHECKPOINT = True

MONITOR_METRIC = "val_accuracy"

MONITOR_MODE = "max"


# ------------------------------------------------------------
# RANDOM SEED
# ------------------------------------------------------------

SEED = 42


# ------------------------------------------------------------
# VERBOSE OUTPUT
# ------------------------------------------------------------

VERBOSE = 1


# ============================================================
# CONFIGURATION DISPLAY
# ============================================================

def print_config():

    print("=" * 70)
    print("PAWCARE - CAT VS NON-CAT DETECTOR CONFIGURATION")
    print("=" * 70)

    print("\nPROJECT:")
    print("-" * 70)
    print(f"Project root       : {PROJECT_ROOT}")

    print("\nDATASET:")
    print("-" * 70)
    print(f"Dataset directory  : {DATASET_DIR}")
    print(f"Training directory : {TRAIN_DIR}")
    print(f"Validation dir     : {VALIDATION_DIR}")
    print(f"Test directory     : {TEST_DIR}")

    print("\nCLASSES:")
    print("-" * 70)
    print(f"Number of classes  : {NUM_CLASSES}")

    for index, class_name in enumerate(EXPECTED_CLASSES):
        print(f" {index}: {class_name}")

    print("\nIMAGE SETTINGS:")
    print("-" * 70)
    print(f"Image size         : {IMAGE_HEIGHT} x {IMAGE_WIDTH}")
    print(f"Channels           : {CHANNELS}")

    print("\nTRAINING SETTINGS:")
    print("-" * 70)
    print(f"Batch size         : {BATCH_SIZE}")
    print(f"Epochs             : {EPOCHS}")
    print(f"Learning rate      : {LEARNING_RATE}")
    print(f"Optimizer          : {OPTIMIZER_NAME}")
    print(f"Dropout            : {DROPOUT_RATE}")

    print("\nBASE MODEL:")
    print("-" * 70)
    print(f"Architecture       : {BASE_MODEL_NAME}")
    print(f"Local weights      : {BASE_MODEL_WEIGHTS}")
    print(f"Freeze base model  : {FREEZE_BASE_MODEL}")
    print(f"Pooling            : {POOLING}")

    print("\nDATA AUGMENTATION:")
    print("-" * 70)
    print(f"Enabled            : {USE_AUGMENTATION}")
    print(f"Horizontal flip    : {HORIZONTAL_FLIP}")
    print(f"Rotation           : {ROTATION_FACTOR}")
    print(f"Zoom               : {ZOOM_FACTOR}")
    print(f"Translation height : {TRANSLATION_HEIGHT}")
    print(f"Translation width  : {TRANSLATION_WIDTH}")

    print("\nCALLBACKS:")
    print("-" * 70)
    print(f"Early stopping     : {USE_EARLY_STOPPING}")
    print(f"Patience           : {EARLY_STOPPING_PATIENCE}")
    print(f"Restore best       : {RESTORE_BEST_WEIGHTS}")
    print(f"Checkpoint         : {USE_MODEL_CHECKPOINT}")
    print(f"Monitor            : {MONITOR_METRIC}")
    print(f"Monitor mode       : {MONITOR_MODE}")

    print("\nOUTPUT:")
    print("-" * 70)
    print(f"Model              : {MODEL_PATH}")
    print(f"Class names        : {CLASS_NAMES_PATH}")
    print(f"Results directory  : {RESULTS_DIR}")

    print("=" * 70)


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print_config()

    print("\n" + "-" * 70)
    print("CHECKING REQUIRED PATHS")
    print("-" * 70)

    if not DATASET_DIR.exists():
        print("WARNING: DATASET DIRECTORY NOT FOUND")
        print(f"Expected path: {DATASET_DIR}")
    else:
        print("✓ Dataset directory found")
        print(f"  {DATASET_DIR}")

    if not TRAIN_DIR.exists():
        print("WARNING: TRAINING DIRECTORY NOT FOUND")
        print(f"Expected path: {TRAIN_DIR}")
    else:
        print("✓ Training directory found")

    if not VALIDATION_DIR.exists():
        print("WARNING: VALIDATION DIRECTORY NOT FOUND")
        print(f"Expected path: {VALIDATION_DIR}")
    else:
        print("✓ Validation directory found")

    if not TEST_DIR.exists():
        print("WARNING: TEST DIRECTORY NOT FOUND")
        print(f"Expected path: {TEST_DIR}")
    else:
        print("✓ Test directory found")

    if not BASE_MODEL_WEIGHTS.exists():
        print("\nWARNING: LOCAL EFFICIENTNETB0 WEIGHTS NOT FOUND")
        print(f"Expected path: {BASE_MODEL_WEIGHTS}")
    else:
        print("\n✓ Local EfficientNetB0 weights found")
        print(
            f"  File size: "
            f"{BASE_MODEL_WEIGHTS.stat().st_size:,} bytes"
        )

    print("\nSTATUS: CONFIGURATION TEST COMPLETED")