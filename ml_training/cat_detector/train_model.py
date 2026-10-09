import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

import config
from data_loader import load_datasets
from model_builder import build_model


# ============================================================
# PAWCARE - CAT VS NON-CAT MODEL TRAINING
# ============================================================


def save_training_history(history):

    history_path = (
        config.RESULTS_DIR /
        "training_history.json"
    )

    history_data = {
        key: [float(value) for value in values]
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
        f"\n✓ Training history saved to:\n"
        f"  {history_path}"
    )


def plot_accuracy(history):

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("Cat Detector - Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    output_path = (
        config.RESULTS_DIR /
        "accuracy_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Accuracy curve saved to:\n"
        f"  {output_path}"
    )


def plot_loss(history):

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("Cat Detector - Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    output_path = (
        config.RESULTS_DIR /
        "loss_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Loss curve saved to:\n"
        f"  {output_path}"
    )


def train_model():

    print("\n" + "=" * 70)
    print("PAWCARE - CAT VS NON-CAT MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    config.RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    config.TRAINED_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nSTEP 1: Loading datasets")
    print("-" * 70)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names
    ) = load_datasets()

    print(
        f"✓ Training dataset loaded"
    )

    print(
        f"✓ Validation dataset loaded"
    )

    print(
        f"✓ Test dataset loaded"
    )

    print(
        f"✓ Classes: {class_names}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\nSTEP 2: Building model")
    print("-" * 70)

    model = build_model()

    print(
        "\n✓ Cat detector model ready for training"
    )

    # --------------------------------------------------------
    # Checkpoint path
    # --------------------------------------------------------

    checkpoint_path = (
        config.TRAINED_MODEL_DIR /
        "best_cat_detector.keras"
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    print("\nSTEP 3: Configuring training callbacks")
    print("-" * 70)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=config.MONITOR,
        patience=config.PATIENCE,
        mode=config.MONITOR_MODE,
        restore_best_weights=config.RESTORE_BEST_WEIGHTS,
        verbose=1
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor=config.MONITOR,
        mode=config.MONITOR_MODE,
        save_best_only=True,
        verbose=1
    )

    print(
        f"✓ Early stopping configured"
    )

    print(
        f"✓ Best model checkpoint:\n"
        f"  {checkpoint_path}"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nSTEP 4: Starting model training")
    print("-" * 70)

    print(
        f"Epochs       : {config.EPOCHS}"
    )

    print(
        f"Batch size   : {config.BATCH_SIZE}"
    )

    print(
        f"Learning rate: {config.LEARNING_RATE}"
    )

    print(
        f"Monitor      : {config.MONITOR}"
    )

    print(
        "\nTraining has started...\n"
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
    # Training completed
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    epochs_completed = len(
        history.history["loss"]
    )

    print(
        f"\nEpochs completed: {epochs_completed}"
    )

    # --------------------------------------------------------
    # Best epoch
    # --------------------------------------------------------

    validation_accuracies = (
        history.history["val_accuracy"]
    )

    best_epoch_index = max(
        range(len(validation_accuracies)),
        key=lambda index: validation_accuracies[index]
    )

    best_epoch = best_epoch_index + 1

    best_val_accuracy = (
        validation_accuracies[best_epoch_index]
    )

    best_train_accuracy = (
        history.history["accuracy"][best_epoch_index]
    )

    best_val_loss = (
        history.history["val_loss"][best_epoch_index]
    )

    print(
        f"\nBest epoch           : {best_epoch}"
    )

    print(
        f"Best training acc.   : "
        f"{best_train_accuracy * 100:.2f}%"
    )

    print(
        f"Best validation acc. : "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best validation loss : "
        f"{best_val_loss:.4f}"
    )

    # --------------------------------------------------------
    # Save final best model
    # --------------------------------------------------------

    final_model_path = (
        config.TRAINED_MODEL_DIR /
        "cat_detector.keras"
    )

    model.save(
        final_model_path
    )

    print(
        f"\n✓ Final trained model saved to:\n"
        f"  {final_model_path}"
    )

    # --------------------------------------------------------
    # Save class names
    # --------------------------------------------------------

    class_names_path = (
        config.TRAINED_MODEL_DIR /
        "class_names.json"
    )

    with open(
        class_names_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            class_names,
            file,
            indent=4
        )

    print(
        f"✓ Class names saved to:\n"
        f"  {class_names_path}"
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    save_training_history(
        history
    )

    # --------------------------------------------------------
    # Create training graphs
    # --------------------------------------------------------

    print("\nSTEP 5: Creating training graphs")
    print("-" * 70)

    plot_accuracy(
        history
    )

    plot_loss(
        history
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STATUS: CAT DETECTOR TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        "\nThe model is now ready for the evaluation stage."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_model()