import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

import config
from data_loader import load_datasets
from model_builder import build_model


# ============================================================
# PawCare - Dog vs Non-Dog Detector
# Model Training
# ============================================================


def save_class_names(class_names):

    output_path = (
        config.TRAINED_MODEL_DIR
        / "class_names.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            class_names,
            file,
            indent=4
        )

    print(
        f"✓ Class names saved: {output_path}"
    )


def save_training_history(history):

    history_path = (
        config.RESULTS_DIR
        / "training_history.json"
    )

    history_data = {
        key: [
            float(value)
            for value in values
        ]
        for key, values in history.history.items()
    }

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history_data,
            file,
            indent=4
        )

    print(
        f"✓ Training history saved: {history_path}"
    )


def plot_training_curves(history):

    # --------------------------------------------------------
    # Accuracy curve
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title(
        "PawCare Dog Detector - Accuracy"
    )

    plt.legend()

    plt.grid(True)

    accuracy_path = (
        config.RESULTS_DIR
        / "accuracy_curve.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Accuracy curve saved: {accuracy_path}"
    )

    # --------------------------------------------------------
    # Loss curve
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        "PawCare Dog Detector - Loss"
    )

    plt.legend()

    plt.grid(True)

    loss_path = (
        config.RESULTS_DIR
        / "loss_curve.png"
    )

    plt.savefig(
        loss_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Loss curve saved: {loss_path}"
    )


def train_model():

    print("=" * 70)
    print("PAWCARE - DOG DETECTOR TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names
    ) = load_datasets()

    print("\n✓ Datasets loaded")

    print(
        f"Classes: {class_names}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nBuilding model...")

    model = build_model()

    print("✓ Model built")

    # --------------------------------------------------------
    # Callback paths
    # --------------------------------------------------------

    best_model_path = (
        config.TRAINED_MODEL_DIR
        / "best_dog_detector.keras"
    )

    final_model_path = (
        config.TRAINED_MODEL_DIR
        / "dog_detector.keras"
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=config.MONITOR,
        mode=config.MONITOR_MODE,
        patience=config.PATIENCE,
        restore_best_weights=config.RESTORE_BEST_WEIGHTS,
        verbose=1
    )

    # --------------------------------------------------------
    # Model checkpoint
    # --------------------------------------------------------

    model_checkpoint = (
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor=config.MONITOR,
            mode=config.MONITOR_MODE,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    print(
        f"\nMaximum epochs: {config.EPOCHS}"
    )

    print(
        f"Batch size: {config.BATCH_SIZE}"
    )

    print(
        f"Learning rate: {config.LEARNING_RATE}"
    )

    print(
        f"Early stopping patience: {config.PATIENCE}"
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=config.EPOCHS,
        callbacks=[
            early_stopping,
            model_checkpoint
        ],
        verbose=1
    )

    # --------------------------------------------------------
    # Training summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    epochs_completed = len(
        history.history["loss"]
    )

    print(
        f"\nEpochs completed: "
        f"{epochs_completed}/{config.EPOCHS}"
    )

    best_epoch = (
        int(
            max(
                range(
                    len(
                        history.history[
                            "val_accuracy"
                        ]
                    )
                ),
                key=lambda index:
                history.history[
                    "val_accuracy"
                ][index]
            )
        )
        + 1
    )

    best_train_accuracy = max(
        history.history["accuracy"]
    )

    best_validation_accuracy = max(
        history.history["val_accuracy"]
    )

    best_validation_loss = min(
        history.history["val_loss"]
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best training accuracy: "
        f"{best_train_accuracy:.4f}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.4f}"
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    print("\nSaving final model...")

    model.save(
        final_model_path
    )

    print(
        f"✓ Final model saved: "
        f"{final_model_path}"
    )

    # --------------------------------------------------------
    # Save class names
    # --------------------------------------------------------

    save_class_names(
        class_names
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    save_training_history(
        history
    )

    # --------------------------------------------------------
    # Save curves
    # --------------------------------------------------------

    plot_training_curves(
        history
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING OUTPUTS")
    print("=" * 70)

    print(
        f"\nFinal model:"
    )

    print(
        final_model_path
    )

    print(
        f"\nBest checkpoint:"
    )

    print(
        best_model_path
    )

    print(
        f"\nResults directory:"
    )

    print(
        config.RESULTS_DIR
    )

    print("\n" + "=" * 70)
    print("STATUS: DOG DETECTOR TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":

    train_model()