import os
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# PAWCARE - CAT BREED MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("PAWCARE - CAT BREED MODEL EVALUATION")
print("=" * 60)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/cat_breed_classifier.keras"
CLASS_NAMES_PATH = "models/class_names.json"

TEST_DIR = r"C:\PawCareData\cat_breed_split\test"

IMG_SIZE = 224
BATCH_SIZE = 8


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking required files...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"\nClass names file not found:\n{CLASS_NAMES_PATH}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"\nTest dataset not found:\n{TEST_DIR}"
    )

print("Model file       : FOUND")
print("Class names file : FOUND")
print("Test dataset     : FOUND")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

print("\nLoading class names...")

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

print("\nClasses:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")

print(f"\nNumber of classes: {len(class_names)}")


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING TEST DATASET")
print("=" * 60)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nTest dataset loaded successfully.")

print("Test images:", len(test_dataset.file_paths))


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING TRAINED MODEL")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully!")


# ============================================================
# BASIC MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print("\nTest Loss:")
print(round(test_loss, 4))

print("\nTest Accuracy:")
print(round(test_accuracy * 100, 2), "%")


# ============================================================
# MAKE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

predictions = model.predict(
    test_dataset,
    verbose=1
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)


# ============================================================
# GET TRUE LABELS
# ============================================================

true_classes = np.concatenate(
    [
        labels.numpy()
        for images, labels in test_dataset
    ]
)


# ============================================================
# VERIFY LABEL COUNT
# ============================================================

print("\nTrue labels:", len(true_classes))

print("Predictions :", len(predicted_classes))


if len(true_classes) != len(predicted_classes):
    raise ValueError(
        "Number of true labels and predictions do not match."
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_names,
    digits=4
)

print("\n")
print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print("\nRows = Actual class")
print("Columns = Predicted class\n")

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

print("\nCreating confusion matrix image...")

plt.figure(figsize=(12, 10))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "PawCare Cat Breed Classification - Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(len(class_names))

plt.xticks(
    tick_marks,
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    tick_marks,
    class_names
)

plt.xlabel("Predicted Breed")

plt.ylabel("Actual Breed")


# Add numbers inside the matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            verticalalignment="center"
        )


plt.tight_layout()

CONFUSION_MATRIX_PATH = (
    "models/cat_breed_confusion_matrix.png"
)

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion matrix saved to:")
print(CONFUSION_MATRIX_PATH)


# ============================================================
# PER-BREED ACCURACY
# ============================================================

print("\n" + "=" * 60)
print("PER-BREED RESULTS")
print("=" * 60)

print(
    f"\n{'Breed':25s} "
    f"{'Correct':>8s} "
    f"{'Total':>8s} "
    f"{'Accuracy':>10s}"
)

print("-" * 58)


for index, breed in enumerate(class_names):

    total = np.sum(true_classes == index)

    correct = np.sum(
        (true_classes == index)
        &
        (predicted_classes == index)
    )

    if total > 0:
        accuracy = correct / total * 100
    else:
        accuracy = 0

    print(
        f"{breed:25s} "
        f"{correct:8d} "
        f"{total:8d} "
        f"{accuracy:9.2f}%"
    )


# ============================================================
# SAVE PREDICTION RESULTS
# ============================================================

print("\n" + "=" * 60)
print("SAVING PREDICTION RESULTS")
print("=" * 60)

prediction_file = "models/test_predictions.txt"

with open(prediction_file, "w") as file:

    for index in range(len(true_classes)):

        actual = class_names[
            true_classes[index]
        ]

        predicted = class_names[
            predicted_classes[index]
        ]

        confidence = float(
            np.max(predictions[index])
        ) * 100

        file.write(
            f"Image {index + 1}: "
            f"Actual = {actual}, "
            f"Predicted = {predicted}, "
            f"Confidence = {confidence:.2f}%\n"
        )

print("\nPrediction results saved to:")
print(prediction_file)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

print("\nTest Accuracy:")
print(round(test_accuracy * 100, 2), "%")

print("\nTest Loss:")
print(round(test_loss, 4))

print("\nConfusion Matrix:")
print(CONFUSION_MATRIX_PATH)

print("\nPrediction Results:")
print(prediction_file)

print("\n" + "=" * 60)
print("PAWCARE MODEL EVALUATION READY")
print("=" * 60)
