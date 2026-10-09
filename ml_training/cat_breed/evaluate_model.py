"""
PawCare - Cat Breed Classifier Evaluation

This script evaluates the best trained Cat Breed model
on the completely separate test dataset.

Generated results:
1. Test loss
2. Test accuracy
3. Classification report
4. Confusion matrix
5. Training/validation accuracy curve
6. Training/validation loss curve
7. Misclassified image list
"""

from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

import config
from data_loader import load_datasets


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = config.MODEL_PATH
CLASS_NAMES_PATH = config.CLASS_NAMES_PATH
HISTORY_PATH = config.HISTORY_PATH

RESULTS_DIR = config.RESULTS_DIR

CLASSIFICATION_REPORT_PATH = config.CLASSIFICATION_REPORT_PATH
CONFUSION_MATRIX_PATH = config.CONFUSION_MATRIX_PATH
TRAINING_CURVES_PATH = config.TRAINING_CURVES_PATH

MISCLASSIFICATIONS_PATH = RESULTS_DIR / "misclassifications.txt"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_class_names():
    """Load class names saved during training."""

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found:\n{CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    return class_names


def load_training_history():
    """Load training history saved during training."""

    if not HISTORY_PATH.exists():
        print()
        print("WARNING: Training history file not found.")
        print(f"Expected location: {HISTORY_PATH}")
        return None

    with open(HISTORY_PATH, "r", encoding="utf-8") as file:
        history = json.load(file)

    return history


def collect_predictions(model, test_dataset):
    """
    Generate predictions for every image in the test dataset.
    """

    print()
    print("-" * 70)
    print("Generating predictions on test dataset...")
    print("-" * 70)

    true_labels = []
    predicted_labels = []
    probabilities = []

    total_batches = 0

    for images, labels in test_dataset:

        predictions = model.predict(
            images,
            verbose=0
        )

        predicted = np.argmax(
            predictions,
            axis=1
        )

        true_labels.extend(
            labels.numpy().astype(int).tolist()
        )

        predicted_labels.extend(
            predicted.astype(int).tolist()
        )

        probabilities.extend(
            np.max(predictions, axis=1).tolist()
        )

        total_batches += 1

    print(f"Prediction batches processed: {total_batches}")

    return (
        np.array(true_labels),
        np.array(predicted_labels),
        np.array(probabilities),
    )


def save_classification_report(
    true_labels,
    predicted_labels,
    class_names,
):
    """Create and save classification report."""

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    with open(
        CLASSIFICATION_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "PAWCARE - CAT BREED CLASSIFIER\n"
        )

        file.write(
            "TEST DATASET CLASSIFICATION REPORT\n"
        )

        file.write("=" * 70 + "\n\n")

        file.write(
            f"Number of test images: {len(true_labels)}\n"
        )

        file.write(
            f"Number of classes: {len(class_names)}\n\n"
        )

        file.write(report)

    print()
    print("Classification report:")
    print("-" * 70)
    print(report)

    print()
    print("Classification report saved to:")
    print(CLASSIFICATION_REPORT_PATH)


def save_confusion_matrix(
    true_labels,
    predicted_labels,
    class_names,
):
    """Create and save confusion matrix."""

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
    )

    plt.figure(figsize=(12, 10))

    plt.imshow(matrix)

    plt.title(
        "PawCare Cat Breed Classifier - Confusion Matrix"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(class_names)),
        class_names,
    )

    # Display values inside the matrix.
    for row in range(matrix.shape[0]):

        for column in range(matrix.shape[1]):

            plt.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("Confusion matrix saved to:")
    print(CONFUSION_MATRIX_PATH)


def save_training_curves(history):
    """Create and save training accuracy/loss curves."""

    if history is None:
        print()
        print(
            "Training history unavailable. "
            "Skipping training curves."
        )
        return

    required_keys = [
        "accuracy",
        "val_accuracy",
        "loss",
        "val_loss",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in history
    ]

    if missing_keys:

        print()
        print(
            "WARNING: Training history is missing:"
        )

        for key in missing_keys:
            print(f" - {key}")

        return

    epochs = range(
        1,
        len(history["accuracy"]) + 1
    )

    # --------------------------------------------------------
    # Accuracy curve
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        history["accuracy"],
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.title(
        "PawCare Cat Breed Classifier - Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        TRAINING_CURVES_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("Training curves saved to:")
    print(TRAINING_CURVES_PATH)

    # --------------------------------------------------------
    # Loss curve
    # --------------------------------------------------------

    loss_curve_path = RESULTS_DIR / "loss_curve.png"

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        history["loss"],
        label="Training Loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
    )

    plt.title(
        "PawCare Cat Breed Classifier - Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        loss_curve_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("Loss curve saved to:")
    print(loss_curve_path)


def save_misclassifications(
    test_dataset,
    true_labels,
    predicted_labels,
    probabilities,
    class_names,
):
    """
    Save details of misclassified test images.

    image_dataset_from_directory provides file paths
    through the dataset's file_paths attribute.
    """

    print()
    print("-" * 70)
    print("Analyzing misclassified images...")
    print("-" * 70)

    if not hasattr(test_dataset, "file_paths"):

        print(
            "WARNING: Test dataset does not expose file paths."
        )

        return

    file_paths = test_dataset.file_paths

    misclassified_count = 0

    lines = []

    lines.append(
        "PAWCARE - CAT BREED MISCLASSIFICATION ANALYSIS"
    )

    lines.append("=" * 70)
    lines.append("")

    for index in range(len(true_labels)):

        true_index = int(true_labels[index])
        predicted_index = int(predicted_labels[index])

        if true_index == predicted_index:
            continue

        misclassified_count += 1

        confidence = float(probabilities[index])

        image_path = file_paths[index]

        true_class = class_names[true_index]
        predicted_class = class_names[predicted_index]

        lines.append(
            f"Image {misclassified_count}"
        )

        lines.append(
            f"Path             : {image_path}"
        )

        lines.append(
            f"Actual breed     : {true_class}"
        )

        lines.append(
            f"Predicted breed  : {predicted_class}"
        )

        lines.append(
            f"Confidence       : {confidence:.4f}"
        )

        lines.append("")

    lines.append("=" * 70)

    lines.append(
        f"Total test images      : {len(true_labels)}"
    )

    lines.append(
        f"Correct predictions    : "
        f"{len(true_labels) - misclassified_count}"
    )

    lines.append(
        f"Incorrect predictions  : {misclassified_count}"
    )

    with open(
        MISCLASSIFICATIONS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(lines)
        )

    print(
        f"Misclassified images: {misclassified_count}"
    )

    print()
    print("Misclassification analysis saved to:")
    print(MISCLASSIFICATIONS_PATH)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print_header(
        "PAWCARE - CAT BREED CLASSIFIER EVALUATION"
    )

    # --------------------------------------------------------
    # STEP 1: Check files
    # --------------------------------------------------------

    print()
    print("STEP 1: Checking required files")
    print("-" * 70)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}"
        )

    print("✓ Trained model found")
    print(f"  {MODEL_PATH}")

    if CLASS_NAMES_PATH.exists():

        print("✓ Class names file found")

    else:

        raise FileNotFoundError(
            f"Class names file not found:\n"
            f"{CLASS_NAMES_PATH}"
        )

    # --------------------------------------------------------
    # STEP 2: Load class names
    # --------------------------------------------------------

    print()
    print("STEP 2: Loading class names")
    print("-" * 70)

    class_names = load_class_names()

    print(f"Number of classes: {len(class_names)}")

    for index, class_name in enumerate(class_names):

        print(
            f"{index:2d}: {class_name}"
        )

    # --------------------------------------------------------
    # STEP 3: Load test dataset
    # --------------------------------------------------------

    print()
    print("STEP 3: Loading test dataset")
    print("-" * 70)

    _, _, test_dataset, _ = load_datasets()

    print()
    print(
        f"Test batches: "
        f"{tf.data.experimental.cardinality(test_dataset).numpy()}"
    )

    # --------------------------------------------------------
    # STEP 4: Load trained model
    # --------------------------------------------------------

    print()
    print("STEP 4: Loading trained model")
    print("-" * 70)

    print(
        "Loading best model saved during training..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("✓ Model loaded successfully")

    # --------------------------------------------------------
    # STEP 5: Evaluate test dataset
    # --------------------------------------------------------

    print()
    print("STEP 5: Evaluating on TEST dataset")
    print("-" * 70)

    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=1,
    )

    print()
    print(
        f"Test loss     : {test_loss:.4f}"
    )

    print(
        f"Test accuracy : {test_accuracy:.4f}"
    )

    print(
        f"Test accuracy : {test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # STEP 6: Generate predictions
    # --------------------------------------------------------

    print()
    print("STEP 6: Generating predictions")

    (
        true_labels,
        predicted_labels,
        probabilities,
    ) = collect_predictions(
        model,
        test_dataset,
    )

    calculated_accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    print()
    print(
        f"Calculated prediction accuracy: "
        f"{calculated_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # STEP 7: Classification report
    # --------------------------------------------------------

    print()
    print("STEP 7: Creating classification report")

    save_classification_report(
        true_labels,
        predicted_labels,
        class_names,
    )

    # --------------------------------------------------------
    # STEP 8: Confusion matrix
    # --------------------------------------------------------

    print()
    print("STEP 8: Creating confusion matrix")

    save_confusion_matrix(
        true_labels,
        predicted_labels,
        class_names,
    )

    # --------------------------------------------------------
    # STEP 9: Training curves
    # --------------------------------------------------------

    print()
    print("STEP 9: Creating training curves")

    history = load_training_history()

    save_training_curves(history)

    # --------------------------------------------------------
    # STEP 10: Misclassification analysis
    # --------------------------------------------------------

    print()
    print("STEP 10: Misclassification analysis")

    save_misclassifications(
        test_dataset,
        true_labels,
        predicted_labels,
        probabilities,
        class_names,
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_header(
        "CAT BREED MODEL EVALUATION COMPLETED"
    )

    print()
    print("FINAL TEST RESULTS")
    print("-" * 70)

    print(
        f"Test images       : {len(true_labels)}"
    )

    print(
        f"Test loss         : {test_loss:.4f}"
    )

    print(
        f"Test accuracy     : {test_accuracy * 100:.2f}%"
    )

    print(
        f"Correct           : "
        f"{np.sum(true_labels == predicted_labels)}"
    )

    print(
        f"Incorrect         : "
        f"{np.sum(true_labels != predicted_labels)}"
    )

    print()
    print("Generated evaluation files:")
    print()

    print(
        f"✓ Classification report:"
    )
    print(
        f"  {CLASSIFICATION_REPORT_PATH}"
    )

    print(
        f"✓ Confusion matrix:"
    )
    print(
        f"  {CONFUSION_MATRIX_PATH}"
    )

    print(
        f"✓ Accuracy curves:"
    )
    print(
        f"  {TRAINING_CURVES_PATH}"
    )

    print(
        f"✓ Misclassification analysis:"
    )
    print(
        f"  {MISCLASSIFICATIONS_PATH}"
    )

    print()
    print(
        "STATUS: CAT BREED MODEL EVALUATION COMPLETED"
    )


if __name__ == "__main__":
    main()