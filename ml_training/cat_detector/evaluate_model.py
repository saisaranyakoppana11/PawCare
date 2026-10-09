import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support
)

import config
from data_loader import load_datasets


# ============================================================
# PAWCARE - CAT VS NON-CAT MODEL EVALUATION
# ============================================================


MODEL_PATH = (
    config.TRAINED_MODEL_DIR /
    "cat_detector.keras"
)

CLASS_NAMES_PATH = (
    config.TRAINED_MODEL_DIR /
    "class_names.json"
)

RESULTS_DIR = config.RESULTS_DIR


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

def load_trained_model():

    print("\n" + "=" * 70)
    print("STEP 1: LOADING TRAINED CAT DETECTOR")
    print("=" * 70)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "\nTrained model not found.\n"
            f"Expected:\n{MODEL_PATH}"
        )

    print(f"\nModel path:\n{MODEL_PATH}")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("\n✓ Trained model loaded successfully")

    print(f"\nModel name: {model.name}")

    print(
        f"Input shape: "
        f"{model.input_shape}"
    )

    print(
        f"Output shape: "
        f"{model.output_shape}"
    )

    return model


# ------------------------------------------------------------
# LOAD CLASS NAMES
# ------------------------------------------------------------

def load_class_names():

    print("\n" + "=" * 70)
    print("STEP 2: LOADING CLASS NAMES")
    print("=" * 70)

    if not CLASS_NAMES_PATH.exists():

        raise FileNotFoundError(
            "\nClass names file not found.\n"
            f"Expected:\n{CLASS_NAMES_PATH}"
        )

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        class_names = json.load(file)

    print("\nClass names:")

    for index, class_name in enumerate(class_names):

        print(
            f" {index}: {class_name}"
        )

    if class_names != config.EXPECTED_CLASSES:

        raise ValueError(
            "\nClass name mismatch!\n"
            f"Expected: {config.EXPECTED_CLASSES}\n"
            f"Found:    {class_names}"
        )

    print("\n✓ Class names verified")

    return class_names


# ------------------------------------------------------------
# COLLECT TRUE LABELS AND PREDICTIONS
# ------------------------------------------------------------

def collect_predictions(
    model,
    test_dataset
):

    print("\n" + "=" * 70)
    print("STEP 3: GENERATING TEST PREDICTIONS")
    print("=" * 70)

    true_labels = []
    predicted_labels = []
    prediction_confidences = []

    total_images = 0

    for images, labels in test_dataset:

        predictions = model.predict(
            images,
            verbose=0
        )

        predicted_class_indices = np.argmax(
            predictions,
            axis=1
        )

        confidence_values = np.max(
            predictions,
            axis=1
        )

        true_labels.extend(
            labels.numpy().tolist()
        )

        predicted_labels.extend(
            predicted_class_indices.tolist()
        )

        prediction_confidences.extend(
            confidence_values.tolist()
        )

        total_images += images.shape[0]

    true_labels = np.array(
        true_labels,
        dtype=np.int32
    )

    predicted_labels = np.array(
        predicted_labels,
        dtype=np.int32
    )

    prediction_confidences = np.array(
        prediction_confidences,
        dtype=np.float32
    )

    print(
        f"\nTotal test images evaluated: "
        f"{total_images}"
    )

    print("\n✓ Predictions generated successfully")

    return (
        true_labels,
        predicted_labels,
        prediction_confidences
    )


# ------------------------------------------------------------
# CALCULATE METRICS
# ------------------------------------------------------------

def calculate_metrics(
    true_labels,
    predicted_labels,
    class_names
):

    print("\n" + "=" * 70)
    print("STEP 4: CALCULATING TEST METRICS")
    print("=" * 70)

    total_images = len(true_labels)

    correct_predictions = int(
        np.sum(
            true_labels == predicted_labels
        )
    )

    incorrect_predictions = (
        total_images -
        correct_predictions
    )

    test_accuracy = (
        correct_predictions /
        total_images
    )

    precision, recall, f1_score, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=list(
                range(len(class_names))
            ),
            average="weighted",
            zero_division=0
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=list(
                range(len(class_names))
            ),
            average="macro",
            zero_division=0
        )
    )

    print(
        f"\nTotal test images : "
        f"{total_images}"
    )

    print(
        f"Correct predictions : "
        f"{correct_predictions}"
    )

    print(
        f"Incorrect predictions : "
        f"{incorrect_predictions}"
    )

    print(
        f"\nTest accuracy : "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Weighted precision : "
        f"{precision:.4f}"
    )

    print(
        f"Weighted recall : "
        f"{recall:.4f}"
    )

    print(
        f"Weighted F1-score : "
        f"{f1_score:.4f}"
    )

    print(
        f"\nMacro precision : "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro recall : "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1-score : "
        f"{macro_f1:.4f}"
    )

    metrics = {
        "total_test_images": total_images,
        "correct_predictions": correct_predictions,
        "incorrect_predictions": incorrect_predictions,
        "test_accuracy": float(test_accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1_score": float(f1_score),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1_score": float(macro_f1)
    }

    return metrics


# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

def save_classification_report(
    true_labels,
    predicted_labels,
    class_names
):

    print("\n" + "=" * 70)
    print("STEP 5: CLASSIFICATION REPORT")
    print("=" * 70)

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(
            range(len(class_names))
        ),
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print("\n")
    print(report)

    report_path = (
        RESULTS_DIR /
        "classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "PAWCARE - CAT VS NON-CAT DETECTOR\n"
        )

        file.write(
            "CLASSIFICATION REPORT\n"
        )

        file.write(
            "=" * 70 +
            "\n\n"
        )

        file.write(report)

    print(
        f"✓ Classification report saved to:\n"
        f"  {report_path}"
    )

    return report


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

def create_confusion_matrix(
    true_labels,
    predicted_labels,
    class_names
):

    print("\n" + "=" * 70)
    print("STEP 6: CONFUSION MATRIX")
    print("=" * 70)

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(
            range(len(class_names))
        )
    )

    print("\nConfusion matrix:")

    print(matrix)

    matrix_path = (
        RESULTS_DIR /
        "confusion_matrix.png"
    )

    plt.figure(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names
    )

    display.plot(
        values_format="d",
        cmap="Blues"
    )

    plt.title(
        "PawCare Cat vs Non-Cat Detector"
    )

    plt.tight_layout()

    plt.savefig(
        matrix_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\n✓ Confusion matrix saved to:\n"
        f"  {matrix_path}"
    )

    return matrix


# ------------------------------------------------------------
# MISCLASSIFICATIONS
# ------------------------------------------------------------

def save_misclassifications(
    true_labels,
    predicted_labels,
    prediction_confidences,
    class_names
):

    print("\n" + "=" * 70)
    print("STEP 7: ANALYZING MISCLASSIFICATIONS")
    print("=" * 70)

    incorrect_indices = np.where(
        true_labels != predicted_labels
    )[0]

    output_path = (
        RESULTS_DIR /
        "misclassifications.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "PAWCARE - CAT VS NON-CAT DETECTOR\n"
        )

        file.write(
            "MISCLASSIFIED TEST SAMPLES\n"
        )

        file.write(
            "=" * 70 +
            "\n\n"
        )

        if len(incorrect_indices) == 0:

            file.write(
                "No misclassified test samples.\n"
            )

            print(
                "\n✓ No misclassified images found"
            )

        else:

            for index in incorrect_indices:

                true_class = class_names[
                    true_labels[index]
                ]

                predicted_class = class_names[
                    predicted_labels[index]
                ]

                confidence = (
                    prediction_confidences[index]
                    * 100
                )

                file.write(
                    f"Test sample index: {index}\n"
                )

                file.write(
                    f"Actual class      : {true_class}\n"
                )

                file.write(
                    f"Predicted class   : {predicted_class}\n"
                )

                file.write(
                    f"Confidence        : "
                    f"{confidence:.2f}%\n"
                )

                file.write(
                    "-" * 50 +
                    "\n"
                )

            print(
                f"\nIncorrect predictions: "
                f"{len(incorrect_indices)}"
            )

            print(
                f"✓ Misclassification details saved to:\n"
                f"  {output_path}"
            )

    return len(incorrect_indices)


# ------------------------------------------------------------
# SAVE METRICS
# ------------------------------------------------------------

def save_metrics(metrics):

    metrics_path = (
        RESULTS_DIR /
        "evaluation_metrics.json"
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
        f"\n✓ Evaluation metrics saved to:\n"
        f"  {metrics_path}"
    )


# ------------------------------------------------------------
# MAIN EVALUATION
# ------------------------------------------------------------

def evaluate_model():

    print("\n" + "=" * 70)
    print("PAWCARE - CAT VS NON-CAT MODEL EVALUATION")
    print("=" * 70)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Step 1
    model = load_trained_model()

    # Step 2
    class_names = load_class_names()

    # Step 3
    (
        _,
        _,
        test_dataset,
        _
    ) = load_datasets()

    (
        true_labels,
        predicted_labels,
        prediction_confidences
    ) = collect_predictions(
        model,
        test_dataset
    )

    # Step 4
    metrics = calculate_metrics(
        true_labels,
        predicted_labels,
        class_names
    )

    # Step 5
    save_classification_report(
        true_labels,
        predicted_labels,
        class_names
    )

    # Step 6
    create_confusion_matrix(
        true_labels,
        predicted_labels,
        class_names
    )

    # Step 7
    save_misclassifications(
        true_labels,
        predicted_labels,
        prediction_confidences,
        class_names
    )

    # Save metrics
    save_metrics(metrics)

    print("\n" + "=" * 70)
    print("CAT DETECTOR EVALUATION COMPLETED")
    print("=" * 70)

    print(
        f"\nFinal test accuracy: "
        f"{metrics['test_accuracy'] * 100:.2f}%"
    )

    print(
        f"Weighted F1-score: "
        f"{metrics['weighted_f1_score']:.4f}"
    )

    print(
        "\n✓ Model evaluation completed successfully"
    )

    print(
        "\nThe Cat Detector is now ready for "
        "final verification."
    )


if __name__ == "__main__":

    evaluate_model()