import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

import config
from data_loader import load_datasets


# ============================================================
# PawCare - Dog vs Non-Dog Detector
# Model Evaluation
# ============================================================


def load_class_names():

    class_names_path = (
        config.TRAINED_MODEL_DIR
        / "class_names.json"
    )

    with open(
        class_names_path,
        "r",
        encoding="utf-8"
    ) as file:

        class_names = json.load(file)

    return class_names


def evaluate_model():

    print("=" * 70)
    print("PAWCARE - DOG DETECTOR MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Model path
    # --------------------------------------------------------

    model_path = (
        config.TRAINED_MODEL_DIR
        / "dog_detector.keras"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    print(
        f"\nLoading model:\n{model_path}"
    )

    model = tf.keras.models.load_model(
        model_path
    )

    print("✓ Model loaded successfully")

    # --------------------------------------------------------
    # Load class names
    # --------------------------------------------------------

    class_names = load_class_names()

    print(
        f"\nClasses: {class_names}"
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    (
        _,
        _,
        test_dataset,
        _
    ) = load_datasets()

    print("✓ Test dataset loaded")

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    all_true_labels = []

    all_predicted_labels = []

    all_probabilities = []

    for images, labels in test_dataset:

        predictions = model.predict(
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

        all_predicted_labels.extend(
            predicted_labels
        )

        all_probabilities.extend(
            predictions
        )

    y_true = np.array(
        all_true_labels
    )

    y_pred = np.array(
        all_predicted_labels
    )

    probabilities = np.array(
        all_probabilities
    )

    print("✓ Predictions generated")

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Correct / incorrect predictions
    # --------------------------------------------------------

    correct_predictions = int(
        np.sum(y_true == y_pred)
    )

    incorrect_predictions = int(
        np.sum(y_true != y_pred)
    )

    total_predictions = len(
        y_true
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)

    print(
        f"\nTotal test images: "
        f"{total_predictions}"
    )

    print(
        f"Correct predictions: "
        f"{correct_predictions}"
    )

    print(
        f"Incorrect predictions: "
        f"{incorrect_predictions}"
    )

    print(
        f"\nTest accuracy: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Weighted precision: "
        f"{precision:.4f}"
    )

    print(
        f"Weighted recall: "
        f"{recall:.4f}"
    )

    print(
        f"Weighted F1-score: "
        f"{f1:.4f}"
    )

    print(
        f"\nMacro precision: "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro recall: "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1-score: "
        f"{macro_f1:.4f}"
    )

    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)

    print(report)

    print("-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)

    print(
        f"\nClasses: {class_names}\n"
    )

    print(cm)

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    report_path = (
        config.RESULTS_DIR
        / "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "PawCare Dog Detector "
            "Classification Report\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Test images: {total_predictions}\n"
        )

        file.write(
            f"Correct: {correct_predictions}\n"
        )

        file.write(
            f"Incorrect: {incorrect_predictions}\n\n"
        )

        file.write(
            f"Accuracy: {accuracy:.4f}\n"
        )

        file.write(
            f"Weighted Precision: "
            f"{precision:.4f}\n"
        )

        file.write(
            f"Weighted Recall: "
            f"{recall:.4f}\n"
        )

        file.write(
            f"Weighted F1: "
            f"{f1:.4f}\n\n"
        )

        file.write(report)

        file.write(
            "\n\nConfusion Matrix:\n"
        )

        file.write(
            str(cm)
        )

    print(
        f"\n✓ Classification report saved:\n"
        f"{report_path}"
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    confusion_matrix_path = (
        config.RESULTS_DIR
        / "confusion_matrix.txt"
    )

    with open(
        confusion_matrix_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "PawCare Dog Detector "
            "Confusion Matrix\n"
        )

        file.write(
            "=" * 50 + "\n\n"
        )

        file.write(
            f"Class order: {class_names}\n\n"
        )

        file.write(
            str(cm)
        )

    print(
        f"✓ Confusion matrix saved:\n"
        f"{confusion_matrix_path}"
    )

    # --------------------------------------------------------
    # Save metrics JSON
    # --------------------------------------------------------

    metrics = {

        "model": config.MODEL_NAME,

        "test_images": total_predictions,

        "correct_predictions":
            correct_predictions,

        "incorrect_predictions":
            incorrect_predictions,

        "accuracy":
            float(accuracy),

        "weighted_precision":
            float(precision),

        "weighted_recall":
            float(recall),

        "weighted_f1":
            float(f1),

        "macro_precision":
            float(macro_precision),

        "macro_recall":
            float(macro_recall),

        "macro_f1":
            float(macro_f1),

        "classes":
            class_names,

        "confusion_matrix":
            cm.tolist()
    }

    metrics_path = (
        config.RESULTS_DIR
        / "evaluation_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        f"✓ Evaluation metrics saved:\n"
        f"{metrics_path}"
    )

    # --------------------------------------------------------
    # Misclassification analysis
    # --------------------------------------------------------

    misclassified_indices = np.where(
        y_true != y_pred
    )[0]

    misclassification_path = (
        config.RESULTS_DIR
        / "misclassifications.txt"
    )

    with open(
        misclassification_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "PawCare Dog Detector "
            "Misclassification Analysis\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Total test images: "
            f"{total_predictions}\n"
        )

        file.write(
            f"Misclassified images: "
            f"{len(misclassified_indices)}\n\n"
        )

        if len(misclassified_indices) == 0:

            file.write(
                "No misclassifications found.\n"
            )

        else:

            file.write(
                "Misclassified batch indices:\n\n"
            )

            for index in misclassified_indices:

                true_class = class_names[
                    y_true[index]
                ]

                predicted_class = class_names[
                    y_pred[index]
                ]

                confidence = float(
                    probabilities[
                        index,
                        y_pred[index]
                    ]
                )

                file.write(
                    f"Index {index}: "
                    f"Actual = {true_class}, "
                    f"Predicted = {predicted_class}, "
                    f"Confidence = "
                    f"{confidence:.4f}\n"
                )

    print(
        f"✓ Misclassification analysis saved:\n"
        f"{misclassification_path}"
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STATUS: DOG DETECTOR EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    evaluate_model()