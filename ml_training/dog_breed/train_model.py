from pathlib import Path
import json

import matplotlib.pyplot as plt
import tensorflow as tf

import config
from data_loader import load_datasets
from model_builder import build_model


# ============================================================
# PawCare - Dog Breed Classifier
# Stage 5: Model Training
# ============================================================


def create_output_directories():
    """Create required output directories."""

    config.RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    config.TRAINED_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def save_class_names(class_names):
    """Save class names used during training."""

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
        f"\n✓ Class names saved to:\n"
        f"{output_path}"
    )


def save_training_history(history):
    """Save training history as JSON."""

    history_path = (
        config.RESULTS_DIR
        / "training_history.json"
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
        f"✓ Training history saved to:\n"
        f"{history_path}"
    )


def plot_training_curves(history):
    """Save training accuracy and loss curves."""

    history_data = history.history

    epochs = range(
        1,
        len(history_data["accuracy"]) + 1
    )

    # --------------------------------------------------------
    # Accuracy curve
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history_data["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history_data["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        "PawCare Dog Breed Classifier - Accuracy"
    )

    plt.legend()
    plt.grid(True)

    accuracy_path = (
        config.RESULTS_DIR
        / "training_accuracy.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Accuracy curve saved to:\n"
        f"{accuracy_path}"
    )

    # --------------------------------------------------------
    # Loss curve
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history_data["loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history_data["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "PawCare Dog Breed Classifier - Loss"
    )

    plt.legend()
    plt.grid(True)

    loss_path = (
        config.RESULTS_DIR
        / "training_loss.png"
    )

    plt.savefig(
        loss_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Loss curve saved to:\n"
        f"{loss_path}"
    )


def create_callbacks():
    """Create training callbacks."""

    best_model_path = (
        config.TRAINED_MODEL_DIR
        / "best_dog_breed_classifier.keras"
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor=config.MONITOR,
            mode=config.MONITOR_MODE,
            patience=config.PATIENCE,
            restore_best_weights=config.RESTORE_BEST_WEIGHTS,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor=config.MONITOR,
            mode=config.MONITOR_MODE,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]

    return callbacks


def train_model():
    """Train the PawCare dog breed classifier."""

    print("=" * 70)
    print("PAWCARE - DOG BREED CLASSIFIER")
    print("STAGE 5: MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    create_output_directories()

    print("\n✓ Output directories ready")

    # --------------------------------------------------------
    # Set random seeds
    # --------------------------------------------------------

    tf.random.set_seed(
        config.RANDOM_SEED
    )

    print(
        f"✓ Random seed set to "
        f"{config.RANDOM_SEED}"
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names
    ) = load_datasets()

    print(
        f"\n✓ Training dataset loaded"
    )

    print(
        f"✓ Validation dataset loaded"
    )

    print(
        f"✓ Test dataset loaded"
    )

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
    # Save class names
    # --------------------------------------------------------

    save_class_names(
        class_names
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BUILDING MODEL")
    print("=" * 70)

    model = build_model()

    print(
        "\n✓ Model created successfully"
    )

    # --------------------------------------------------------
    # Display model summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    model.summary()

    # --------------------------------------------------------
    # Create callbacks
    # --------------------------------------------------------

    callbacks = create_callbacks()

    print(
        "\n✓ Early stopping callback created"
    )

    print(
        "✓ Best-model checkpoint callback created"
    )

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)

    print(
        f"Model:              {config.MODEL_NAME}"
    )

    print(
        f"Base model:         {config.BASE_MODEL_NAME}"
    )

    print(
        f"Image size:         {config.IMAGE_SIZE}"
    )

    print(
        f"Batch size:         {config.BATCH_SIZE}"
    )

    print(
        f"Maximum epochs:     {config.EPOCHS}"
    )

    print(
        f"Learning rate:      {config.LEARNING_RATE}"
    )

    print(
        f"Optimizer:          {config.OPTIMIZER_NAME}"
    )

    print(
        f"Dropout:            {config.DROPOUT_RATE}"
    )

    print(
        f"Monitor:            {config.MONITOR}"
    )

    print(
        f"Patience:           {config.PATIENCE}"
    )

    # --------------------------------------------------------
    # Start training
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    print(
        "\nTraining may take some time on CPU..."
    )

    history = model.fit(

        train_dataset,

        validation_data=validation_dataset,

        epochs=config.EPOCHS,

        callbacks=callbacks,

        verbose=1
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    final_model_path = (
        config.TRAINED_MODEL_DIR
        / "dog_breed_classifier.keras"
    )

    model.save(
        final_model_path
    )

    print(
        f"\n✓ Final model saved to:\n"
        f"{final_model_path}"
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    save_training_history(
        history
    )

    # --------------------------------------------------------
    # Save training curves
    # --------------------------------------------------------

    plot_training_curves(
        history
    )

    # --------------------------------------------------------
    # Training summary
    # --------------------------------------------------------

    accuracy_values = (
        history.history["accuracy"]
    )

    validation_accuracy_values = (
        history.history["val_accuracy"]
    )

    loss_values = (
        history.history["loss"]
    )

    validation_loss_values = (
        history.history["val_loss"]
    )

    best_epoch = (
        validation_accuracy_values.index(
            max(validation_accuracy_values)
        ) + 1
    )

    best_training_accuracy = max(
        accuracy_values
    )

    best_validation_accuracy = max(
        validation_accuracy_values
    )

    best_validation_loss = min(
        validation_loss_values
    )

    total_epochs = len(
        accuracy_values
    )

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"\nEpochs completed:           "
        f"{total_epochs}/{config.EPOCHS}"
    )

    print(
        f"Best epoch by val_accuracy: "
        f"{best_epoch}"
    )

    print(
        f"Best training accuracy:     "
        f"{best_training_accuracy:.4f}"
    )

    print(
        f"Best validation accuracy:   "
        f"{best_validation_accuracy:.4f}"
    )

    print(
        f"Best validation loss:       "
        f"{best_validation_loss:.4f}"
    )

    print(
        "\nSaved files:"
    )

    print(
        f"  ✓ {final_model_path}"
    )

    print(
        f"  ✓ {config.TRAINED_MODEL_DIR / 'best_dog_breed_classifier.keras'}"
    )

    print(
        f"  ✓ {config.TRAINED_MODEL_DIR / 'class_names.json'}"
    )

    print(
        f"  ✓ {config.RESULTS_DIR / 'training_history.json'}"
    )

    print(
        f"  ✓ {config.RESULTS_DIR / 'training_accuracy.png'}"
    )

    print(
        f"  ✓ {config.RESULTS_DIR / 'training_loss.png'}"
    )

    print("\n" + "=" * 70)
    print("STATUS: DOG BREED MODEL TRAINING PASSED")
    print("=" * 70)


if __name__ == "__main__":

    train_model()