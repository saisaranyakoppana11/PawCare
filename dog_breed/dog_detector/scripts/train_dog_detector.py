import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "dog_breed",
    "dog_detector",
    "dataset"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "dog_breed",
    "dog_detector",
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "dog_detector.keras"
)

CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "dog_detector_class_names.json"
)

# Existing local EfficientNet weights
LOCAL_WEIGHTS = os.path.join(
    PROJECT_ROOT,
    "models",
    "efficientnetb0_notop.h5"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 8
SEED = 42

NUM_CLASSES = 2

LEARNING_RATE = 0.001
MAX_EPOCHS = 20


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 70)
print("PAWCARE DOG VS NON-DOG DETECTOR")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nDataset:")
print(DATA_DIR)

print("\nModel output:")
print(MODEL_PATH)

print("\nClass names:")
print(CLASS_NAMES_PATH)

print("\nImage size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Maximum epochs:", MAX_EPOCHS)


# ============================================================
# CHECK DATASET
# ============================================================

print("\nChecking dataset folders...")

required_folders = [
    os.path.join(DATA_DIR, "train", "dog"),
    os.path.join(DATA_DIR, "train", "non_dog"),

    os.path.join(DATA_DIR, "validation", "dog"),
    os.path.join(DATA_DIR, "validation", "non_dog"),

    os.path.join(DATA_DIR, "test", "dog"),
    os.path.join(DATA_DIR, "test", "non_dog")
]

for folder in required_folders:

    if not os.path.exists(folder):

        raise FileNotFoundError(
            f"Required folder not found:\n{folder}"
        )

print("All dataset folders found.")


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    label_mode="int"
)


print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "validation"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)


print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nDetected classes:")
print(class_names)

if len(class_names) != NUM_CLASSES:

    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(class_names)}."
    )


# ============================================================
# VERIFY CLASS CONSISTENCY
# ============================================================

if validation_dataset.class_names != class_names:

    raise ValueError(
        "Validation classes do not match training classes."
    )

if test_dataset.class_names != class_names:

    raise ValueError(
        "Test classes do not match training classes."
    )

print("\nClass consistency check passed.")


# ============================================================
# DATASET INFORMATION
# ============================================================

train_count = len(
    train_dataset.file_paths
)

validation_count = len(
    validation_dataset.file_paths
)

test_count = len(
    test_dataset.file_paths
)

print("\nDataset sizes:")
print("Training   :", train_count)
print("Validation :", validation_count)
print("Test       :", test_count)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.05
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        )
    ],
    name="data_augmentation"
)


# ============================================================
# LOAD EFFICIENTNETB0
# ============================================================

print("\nLoading EfficientNetB0...")

if os.path.isfile(LOCAL_WEIGHTS):

    print(
        "Using local EfficientNetB0 weights:"
    )

    print(LOCAL_WEIGHTS)

    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        )
    )

    base_model.load_weights(
        LOCAL_WEIGHTS
    )

else:

    print(
        "Local weights not found."
    )

    print(
        "Using ImageNet weights..."
    )

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        )
    )


# ============================================================
# FREEZE BASE MODEL
# ============================================================

base_model.trainable = False


# ============================================================
# BUILD DETECTOR MODEL
# ============================================================

inputs = layers.Input(
    shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)

x = data_augmentation(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(
    0.30
)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs,
    name="PawCare_Dog_Detector"
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nModel summary:\n")

model.summary()


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "w"
) as f:

    json.dump(
        class_names,
        f,
        indent=4
    )

print(
    "\nClass names saved to:"
)

print(
    CLASS_NAMES_PATH
)


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


model_checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=MAX_EPOCHS,
    callbacks=[
        early_stopping,
        model_checkpoint,
        reduce_lr
    ]
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best saved detector...")

best_model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("INDEPENDENT TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = best_model.evaluate(
    test_dataset,
    verbose=1
)


print("\nTest Loss     :", f"{test_loss:.4f}")
print(
    "Test Accuracy :",
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DOG VS NON-DOG DETECTOR TRAINING COMPLETED")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nClass names:")
print(CLASS_NAMES_PATH)

print("\nClasses:")
print(class_names)

print(
    "\nFinal Test Accuracy:",
    f"{test_accuracy * 100:.2f}%"
)

print("\nModel is ready!")