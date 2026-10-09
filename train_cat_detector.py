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
# SETTINGS
# ============================================================

DATA_DIR = r"C:\PawCareData\cat_detector"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "validation")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cat_detector.keras"
)

CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "cat_detector_class_names.json"
)

IMG_SIZE = 224
BATCH_SIZE = 8
SEED = 42

EPOCHS = 20

# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

# ============================================================
# CHECK DATASET
# ============================================================

print("=" * 60)
print("PAWCARE CAT VS NON-CAT MODEL TRAINING")
print("=" * 60)

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training directory not found: {TRAIN_DIR}"
    )

if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(
        f"Validation directory not found: {VAL_DIR}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Test directory not found: {TEST_DIR}"
    )

print("\nDataset:")
print(DATA_DIR)

# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=SEED
)

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_ds.class_names

print("\nClass names:")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")

if len(class_names) != 2:
    raise ValueError(
        f"Expected 2 classes, found {len(class_names)}"
    )

# Save class names
with open(CLASS_NAMES_PATH, "w") as file:

    json.dump(
        class_names,
        file,
        indent=4
    )

print(
    f"\nClass names saved to: "
    f"{CLASS_NAMES_PATH}"
)

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
            0.05,
            0.05
        )
    ],
    name="data_augmentation"
)

# ============================================================
# DATA PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(1)
val_ds = val_ds.prefetch(1)
test_ds = test_ds.prefetch(1)

# ============================================================
# EFFICIENTNETB0 BASE MODEL
# ============================================================

print("\nLoading EfficientNetB0...")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)

# Freeze base model
base_model.trainable = False

print("EfficientNetB0 loaded.")
print("Base model frozen.")

# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(
    0.30
)(x)

outputs = layers.Dense(
    2,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="categorical_crossentropy",

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
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
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

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        checkpoint,
        reduce_lr
    ]
)

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best saved model...")

best_model = tf.keras.models.load_model(
    MODEL_PATH
)

# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("TEST EVALUATION")
print("=" * 60)

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)

print("\nTest Loss:")
print(f"{test_loss:.4f}")

print("\nTest Accuracy:")
print(f"{test_accuracy * 100:.2f}%")

# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print("\nModel:")
print(MODEL_PATH)

print("\nClass names:")
print(CLASS_NAMES_PATH)

print("\nTest Accuracy:")
print(f"{test_accuracy * 100:.2f}%")

print("\nClasses:")
print(class_names)

print("\nNext step:")
print("Test the detector using cat, Pug, dog, and non-animal images.")