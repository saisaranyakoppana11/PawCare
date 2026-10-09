from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import config
from data_loader import load_datasets


# ============================================================
# PawCare - Dog Breed Classifier
# Stage 6: Model Evaluation
# ============================================================


def load_class_names():
    """Load class names saved during training."""

    class_names_path = (
        config.TRAINED_MODEL_DIR
        / "class_names.json"
    )

    if not class_names_path.exists():
        raise FileNotFoundError(
            f"Class names file not found:\n"
            f"{class_names_path}"
        )

    with open(
        class_names_path,
        "r",
        encoding="utf-8"
    ) as file:

        class_names = json.load(file)

    return class_names


def evaluate_model():
    """Evaluate the trained dog breed classifier."""

    print("=" * 70)
    print("PAWCARE - DOG BREED CLASSIFIER")
    print("STAGE 6: MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Create results directory
    # --------------------------------------------------------

    config.RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Model path
    # --------------------------------------------------------

    model_path = (
        config.TRAINED_MODEL_DIR
        / "dog_breed_classifier.keras"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"\nTrained model not found:\n"
            f"{model_path}"
        )

    print(
        f"\nLoading trained model:\n"
        f"{model_path}"
    )

    model = tf.keras.models.load_model(
        model_path
    )

    print(
        "\n✓ Trained model loaded successfully"
    )

    # --------------------------------------------------------
    # Load class names
    # --------------------------------------------------------

    class_names = load_class_names()

    print(
        f"\nClasses ({len(class_names)}):"
    )

    for index, class_name in enumerate(
        class_names
    ):

        print(
            f"  {index}: {class_name}"
        )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING TEST DATASET")
    print("=" * 70)

    (
        _,
        _,
        test_dataset,
        loader_class_names
    ) = load_datasets()

    # --------------------------------------------------------
    # Verify class names
    # --------------------------------------------------------

    if class_names != loader_class_names:

        raise ValueError(
            "\nClass-name mismatch detected!\n"
            f"Saved classes: {class_names}\n"
            f"Loader classes: {loader_class_names}"
        )

    print(
        "\n✓ Class names verified"
    )

    # --------------------------------------------------------
    # Collect true labels
    # --------------------------------------------------------

    print(
        "\nCollecting test labels..."
    )

    true_labels = []

    for _, labels in test_dataset:

        true_labels.extend(
            labels.numpy().tolist()
        )

    true_labels = np.array(
        true_labels,
        dtype=np.int32
    )

    print(
        f"✓ Test labels collected: "
        f"{len(true_labels)}"
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    predictions = model.predict(
        test_dataset,
        verbose=1
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    print(
        "\n✓ Predictions generated"
    )

    # --------------------------------------------------------
    # Verify prediction count
    # --------------------------------------------------------

    if len(true_labels) != len(
        predicted_labels
    ):

        raise ValueError(
            "\nPrediction count does not match "
            "test-label count."
        )

    print(
        f"✓ Prediction count verified: "
        f"{len(predicted_labels)}"
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    weighted_precision = precision_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    weighted_recall = recall_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    macro_precision = precision_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    correct_predictions = int(
        np.sum(
            true_labels == predicted_labels
        )
    )

    incorrect_predictions = int(
        np.sum(
            true_labels != predicted_labels
        )
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(
            range(len(class_names))
        ),
        target_names=class_names,
        zero_division=0,
        digits=4
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(
            range(len(class_names))
        )
    )

    # --------------------------------------------------------
    # Print evaluation results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"\nTotal test images:       "
        f"{len(true_labels)}"
    )

    print(
        f"Correct predictions:     "
        f"{correct_predictions}"
    )

    print(
        f"Incorrect predictions:   "
        f"{incorrect_predictions}"
    )

    print(
        f"\nTest accuracy:           "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"\nWeighted precision:      "
        f"{weighted_precision:.4f}"
    )

    print(
        f"Weighted recall:         "
        f"{weighted_recall:.4f}"
    )

    print(
        f"Weighted F1-score:       "
        f"{weighted_f1:.4f}"
    )

    print(
        f"\nMacro precision:         "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro recall:            "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1-score:          "
        f"{macro_f1:.4f}"
    )

    # --------------------------------------------------------
    # Print classification report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        "\n" + report
    )

    # --------------------------------------------------------
    # Print confusion matrix
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        "\nClass order:"
    )

    for index, class_name in enumerate(
        class_names
    ):

        print(
            f"  {index}: {class_name}"
        )

    print(
        "\n" + str(matrix)
    )

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
            "PAWCARE - DOG BREED CLASSIFIER\n"
        )

        file.write(
            "CLASSIFICATION REPORT\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            report
        )

    print(
        f"\n✓ Classification report saved to:\n"
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
            "PAWCARE - DOG BREED CLASSIFIER\n"
        )

        file.write(
            "CONFUSION MATRIX\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            "Class order:\n"
        )

        for index, class_name in enumerate(
            class_names
        ):

            file.write(
                f"{index}: {class_name}\n"
            )

        file.write(
            "\nConfusion Matrix:\n"
        )

        file.write(
            str(matrix)
        )

    print(
        f"✓ Confusion matrix saved to:\n"
        f"{confusion_matrix_path}"
    )

    # --------------------------------------------------------
    # Save metrics as JSON
    # --------------------------------------------------------

    metrics = {

        "model": config.MODEL_NAME,

        "total_test_images": int(
            len(true_labels)
        ),

        "correct_predictions": (
            correct_predictions
        ),

        "incorrect_predictions": (
            incorrect_predictions
        ),

        "accuracy": float(
            accuracy
        ),

        "accuracy_percentage": float(
            accuracy * 100
        ),

        "weighted_precision": float(
            weighted_precision
        ),

        "weighted_recall": float(
            weighted_recall
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),

        "macro_f1": float(
            macro_f1
        ),

        "class_names": class_names
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
        f"✓ Evaluation metrics saved to:\n"
        f"{metrics_path}"
    )

    # --------------------------------------------------------
    # Save misclassification information
    # --------------------------------------------------------

    misclassifications_path = (
        config.RESULTS_DIR
        / "misclassifications.txt"
    )

    incorrect_indices = np.where(
        true_labels != predicted_labels
    )[0]

    with open(
        misclassifications_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "PAWCARE - DOG BREED CLASSIFIER\n"
        )

        file.write(
            "MISCLASSIFICATIONS\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        if len(incorrect_indices) == 0:

            file.write(
                "No misclassifications found.\n"
            )

        else:

            for index in incorrect_indices:

                true_class = class_names[
                    true_labels[index]
                ]

                predicted_class = class_names[
                    predicted_labels[index]
                ]

                confidence = float(
                    predictions[index][
                        predicted_labels[index]
                    ]
                )

                file.write(
                    f"Sample index: {index}\n"
                )

                file.write(
                    f"Actual: {true_class}\n"
                )

                file.write(
                    f"Predicted: {predicted_class}\n"
                )

                file.write(
                    f"Confidence: "
                    f"{confidence:.4f}\n"
                )

                file.write(
                    "-" * 50 + "\n"
                )

    print(
        f"✓ Misclassification report saved to:\n"
        f"{misclassifications_path}"
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STATUS: DOG BREED MODEL EVALUATION PASSED")
    print("=" * 70)


if __name__ == "__main__":

    evaluate_model()