import os
import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# PAWCARE - DOG BREED CLASSIFICATION
# EfficientNetB0 Transfer Learning
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Current file:
# PawCare/dog_breed/scripts/train_dog_breed.py

DOG_BREED_DIR = Path(__file__).resolve().parents[1]

PAWCARE_DIR = DOG_BREED_DIR.parent

DATASET_DIR = DOG_BREED_DIR / "dataset"

MODEL_DIR = DOG_BREED_DIR / "models"

ROOT_MODEL_DIR = PAWCARE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. DATASET PATHS
# ============================================================

TRAIN_DIR = DATASET_DIR / "train"

VALIDATION_DIR = DATASET_DIR / "validation"

TEST_DIR = DATASET_DIR / "test"


# ============================================================
# 3. MODEL FILES
# ============================================================

MODEL_PATH = (
    MODEL_DIR /
    "dog_breed_classifier.keras"
)

CLASS_NAMES_PATH = (
    MODEL_DIR /
    "dog_class_names.json"
)

# Existing EfficientNetB0 weights from PawCare
EFFICIENTNET_WEIGHTS = (
    ROOT_MODEL_DIR /
    "efficientnetb0_notop.h5"
)


# ============================================================
# 4. SETTINGS
# ============================================================

IMG_SIZE = 224

BATCH_SIZE = 8

SEED = 42

NUM_CLASSES = 10

INITIAL_LEARNING_RATE = 0.001

MAX_EPOCHS = 20


# ============================================================
# 5. DISPLAY INFORMATION
# ============================================================

print()
print("=" * 70)
print("             PAWCARE DOG BREED CLASSIFIER")
print("=" * 70)

print()
print("TensorFlow version:")
print(tf.__version__)

print()
print("Python environment:")
print(os.environ.get("VIRTUAL_ENV", "Not detected"))

print()
print("Dog breed project:")
print(DOG_BREED_DIR)

print()
print("Dataset:")
print(DATASET_DIR)

print()
print("Model output:")
print(MODEL_PATH)


# ============================================================
# 6. CHECK DATASET
# ============================================================

print()
print("=" * 70)
print("CHECKING DATASET")
print("=" * 70)


if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR}"
    )


if not VALIDATION_DIR.exists():
    raise FileNotFoundError(
        f"Validation directory not found:\n{VALIDATION_DIR}"
    )


if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )


# ============================================================
# 7. GET CLASS NAMES
# ============================================================

class_names = sorted(
    [
        folder.name
        for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
    ]
)


print()
print("Detected dog breeds:")

for index, breed in enumerate(class_names):
    print(f"{index}: {breed}")


print()
print(f"Number of classes: {len(class_names)}")


if len(class_names) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(class_names)}."
    )


# ============================================================
# 8. CHECK SAME CLASSES IN ALL SPLITS
# ============================================================

validation_classes = sorted(
    [
        folder.name
        for folder in VALIDATION_DIR.iterdir()
        if folder.is_dir()
    ]
)

test_classes = sorted(
    [
        folder.name
        for folder in TEST_DIR.iterdir()
        if folder.is_dir()
    ]
)


if class_names != validation_classes:
    raise ValueError(
        "Training and validation class names do not match."
    )


if class_names != test_classes:
    raise ValueError(
        "Training and test class names do not match."
    )


print()
print("Class consistency check: PASSED")


# ============================================================
# 9. LOAD TRAINING DATASET
# ============================================================

print()
print("=" * 70)
print("LOADING TRAINING DATASET")
print("=" * 70)


train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


# ============================================================
# 10. LOAD VALIDATION DATASET
# ============================================================

print()
print("=" * 70)
print("LOADING VALIDATION DATASET")
print("=" * 70)


validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 11. LOAD TEST DATASET
# ============================================================

print()
print("=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)


test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 12. DATASET INFORMATION
# ============================================================

print()
print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)


train_count = sum(
    1 for _ in TRAIN_DIR.rglob("*")
    if _.is_file()
)

validation_count = sum(
    1 for _ in VALIDATION_DIR.rglob("*")
    if _.is_file()
)

test_count = sum(
    1 for _ in TEST_DIR.rglob("*")
    if _.is_file()
)


print(f"Training images   : {train_count}")
print(f"Validation images : {validation_count}")
print(f"Test images       : {test_count}")
print(
    f"Total images      : "
    f"{train_count + validation_count + test_count}"
)


# ============================================================
# 13. DATA AUGMENTATION
# ============================================================

print()
print("=" * 70)
print("CREATING DATA AUGMENTATION")
print("=" * 70)


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
        ),
    ],
    name="data_augmentation"
)


# ============================================================
# 14. LOAD EFFICIENTNETB0
# ============================================================

print()
print("=" * 70)
print("LOADING EFFICIENTNETB0")
print("=" * 70)


if EFFICIENTNET_WEIGHTS.exists():

    print()
    print("Using local EfficientNetB0 weights:")
    print(EFFICIENTNET_WEIGHTS)

    base_model = EfficientNetB0(
        include_top=False,
        weights=str(EFFICIENTNET_WEIGHTS),
        input_shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        )
    )

else:

    print()
    print(
        "Local EfficientNetB0 weights were not found."
    )

    print(
        "Attempting to download ImageNet weights..."
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
# 15. FREEZE BASE MODEL
# ============================================================

base_model.trainable = False


print()
print("EfficientNetB0 base model:")
print("Trainable: False")


# ============================================================
# 16. BUILD CLASSIFIER
# ============================================================

print()
print("=" * 70)
print("BUILDING DOG BREED CLASSIFIER")
print("=" * 70)


# EfficientNetB0 already contains its preprocessing.
# Therefore, we do NOT add Rescaling(1./255).


inputs = layers.Input(
    shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    ),
    name="input_image"
)


x = data_augmentation(inputs)


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)


x = layers.Dropout(
    0.30,
    name="dropout"
)(x)


outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="dog_breed_output"
)(x)


model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="PawCare_Dog_Breed_Classifier"
)


# ============================================================
# 17. COMPILE MODEL
# ============================================================

print()
print("=" * 70)
print("COMPILING MODEL")
print("=" * 70)


optimizer = tf.keras.optimizers.Adam(
    learning_rate=INITIAL_LEARNING_RATE
)


model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 18. MODEL SUMMARY
# ============================================================

print()
print("=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary()


# ============================================================
# 19. CALLBACKS
# ============================================================

print()
print("=" * 70)
print("CREATING TRAINING CALLBACKS")
print("=" * 70)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)


reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


callbacks = [
    early_stopping,
    model_checkpoint,
    reduce_learning_rate
]


# ============================================================
# 20. SAVE CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_names,
        file,
        indent=4
    )


print()
print("Class names saved to:")
print(CLASS_NAMES_PATH)


# ============================================================
# 21. TRAIN MODEL
# ============================================================

print()
print("=" * 70)
print("STARTING DOG BREED TRAINING")
print("=" * 70)

print()
print(f"Maximum epochs: {MAX_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Image size: {IMG_SIZE} x {IMG_SIZE}")
print(f"Learning rate: {INITIAL_LEARNING_RATE}")


history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=MAX_EPOCHS,
    callbacks=callbacks
)


# ============================================================
# 22. TRAINING COMPLETED
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)


# ============================================================
# 23. CHECK BEST MODEL
# ============================================================

if MODEL_PATH.exists():

    print()
    print("Best model saved successfully:")
    print(MODEL_PATH)

else:

    print()
    print(
        "WARNING: Model checkpoint file "
        "was not found."
    )


# ============================================================
# 24. EVALUATE ON TEST DATASET
# ============================================================

print()
print("=" * 70)
print("EVALUATING ON INDEPENDENT TEST DATASET")
print("=" * 70)


# Load the best saved model again
# so the final evaluation uses the checkpoint.

best_model = tf.keras.models.load_model(
    MODEL_PATH
)


test_loss, test_accuracy = (
    best_model.evaluate(
        test_dataset,
        verbose=1
    )
)


print()
print("=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ============================================================
# 25. GET TEST PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)


all_true_labels = []

all_predictions = []


for images, labels in test_dataset:

    predictions = best_model.predict(
        images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    all_true_labels.extend(
        labels.numpy()
    )

    all_predictions.extend(
        predicted_labels
    )


all_true_labels = np.array(
    all_true_labels
)

all_predictions = np.array(
    all_predictions
)


# ============================================================
# 26. CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


report = classification_report(
    all_true_labels,
    all_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)


print(report)


# ============================================================
# 27. CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


cm = confusion_matrix(
    all_true_labels,
    all_predictions
)


print()

print("Rows = Actual")
print("Columns = Predicted")

print()

print("              ", end="")

for breed in class_names:

    print(
        f"{breed[:12]:>13}",
        end=""
    )

print()


for index, row in enumerate(cm):

    print(
        f"{class_names[index][:12]:>12} ",
        end=""
    )

    for value in row:

        print(
            f"{value:>13}",
            end=""
        )

    print()


# ============================================================
# 28. FINAL MODEL INFORMATION
# ============================================================

print()
print("=" * 70)
print("FINAL MODEL INFORMATION")
print("=" * 70)

print()
print("Model:")
print(MODEL_PATH)

print()
print("Class names:")
print(CLASS_NAMES_PATH)

print()
print("Classes:")

for index, breed in enumerate(class_names):

    print(
        f"{index}: {breed}"
    )


# ============================================================
# 29. FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("PAWCARE DOG BREED MODEL READY")
print("=" * 70)

print()
print("The dog breed classifier has been:")
print("1. Trained using EfficientNetB0")
print("2. Evaluated using validation data")
print("3. Evaluated using an independent test set")
print("4. Saved as a Keras model")
print("5. Saved with its class names")

print()
print("Next stage:")
print("Create the dog breed prediction system.")

print()
print("=" * 70)