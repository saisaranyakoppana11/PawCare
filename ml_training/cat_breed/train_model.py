"""
PawCare - Cat Breed Classifier
Training Script

This script trains the Cat Breed classifier using:
    - EfficientNetB0 transfer learning
    - Training / validation datasets
    - Data augmentation
    - Adam optimizer
    - Early stopping
    - Model checkpointing

The existing PawCare model is NOT overwritten.
"""

import json
import sys
from pathlib import Path

import tensorflow as tf


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config
from data_loader import load_datasets
from model_builder import build_model


# ============================================================
# REPRODUCIBILITY
# ============================================================

tf.keras.utils.set_random_seed(config.RANDOM_SEED)


# ============================================================
# DIRECTORY SETUP
# ============================================================

config.OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

config.RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

config.MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CALLBACKS
# ============================================================

def create_callbacks():
    """
    Create callbacks used during training.
    """

    callbacks = []

    # --------------------------------------------------------
    # Model Checkpoint
    # --------------------------------------------------------

    if config.USE_MODEL_CHECKPOINT:

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODEL_PATH),
            monitor=config.MONITOR_METRIC,
            mode=config.MONITOR_MODE,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        callbacks.append(checkpoint)

    # --------------------------------------------------------
    # Early Stopping
    # --------------------------------------------------------

    if config.USE_EARLY_STOPPING:

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=config.MONITOR_METRIC,
            mode=config.MONITOR_MODE,
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=config.RESTORE_BEST_WEIGHTS,
            verbose=1
        )

        callbacks.append(early_stopping)

    return callbacks


# ============================================================
# SAVE CLASS NAMES
# ============================================================

def save_class_names(class_names):
    """
    Save class names to JSON so that the trained model
    can later map prediction indexes to breed names.
    """

    with open(
        config.CLASS_NAMES_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            class_names,
            file,
            indent=4
        )

    print()
    print(
        f"Class names saved to:\n"
        f"{config.CLASS_NAMES_PATH}"
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_training_history(history):
    """
    Save training accuracy and loss values as JSON.
    """

    history_data = {
        key: [
            float(value)
            for value in values
        ]
        for key, values in history.history.items()
    }

    with open(
        config.HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history_data,
            file,
            indent=4
        )

    print(
        f"Training history saved to:\n"
        f"{config.HISTORY_PATH}"
    )


# ============================================================
# PRINT TRAINING SUMMARY
# ============================================================

def print_training_summary(history):
    """
    Print the best training and validation results.
    """

    train_accuracy = history.history.get(
        "accuracy",
        []
    )

    validation_accuracy = history.history.get(
        "val_accuracy",
        []
    )

    train_loss = history.history.get(
        "loss",
        []
    )

    validation_loss = history.history.get(
        "val_loss",
        []
    )

    print()
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    print(
        f"Epochs completed     : "
        f"{len(history.epoch)}"
    )

    if train_accuracy:

        print(
            f"Best training accuracy   : "
            f"{max(train_accuracy):.4f}"
        )

    if validation_accuracy:

        best_val_index = validation_accuracy.index(
            max(validation_accuracy)
        )

        print(
            f"Best validation accuracy : "
            f"{max(validation_accuracy):.4f}"
        )

        print(
            f"Best validation epoch    : "
            f"{best_val_index + 1}"
        )

    if train_loss:

        print(
            f"Best training loss       : "
            f"{min(train_loss):.4f}"
        )

    if validation_loss:

        print(
            f"Best validation loss     : "
            f"{min(validation_loss):.4f}"
        )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_model():

    print()
    print("=" * 70)
    print("PAWCARE - CAT BREED CLASSIFIER TRAINING")
    print("=" * 70)

    print()
    print("Training configuration:")
    print("-" * 70)

    print(
        f"Base model          : "
        f"{config.BASE_MODEL_NAME}"
    )

    print(
        f"Image size          : "
        f"{config.IMAGE_HEIGHT} x "
        f"{config.IMAGE_WIDTH}"
    )

    print(
        f"Number of classes   : "
        f"{config.NUM_CLASSES}"
    )

    print(
        f"Batch size          : "
        f"{config.BATCH_SIZE}"
    )

    print(
        f"Maximum epochs      : "
        f"{config.EPOCHS}"
    )

    print(
        f"Learning rate       : "
        f"{config.LEARNING_RATE}"
    )

    print(
        f"Dropout rate        : "
        f"{config.DROPOUT_RATE}"
    )

    print(
        f"Early stopping      : "
        f"{config.USE_EARLY_STOPPING}"
    )

    print(
        f"Patience            : "
        f"{config.EARLY_STOPPING_PATIENCE}"
    )

    print()
    print("-" * 70)
    print("STEP 1: Loading datasets")
    print("-" * 70)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names
    ) = load_datasets()

    print()
    print(
        f"Training classes: {len(class_names)}"
    )

    print(
        f"Classes: {class_names}"
    )

    # --------------------------------------------------------
    # Save class names
    # --------------------------------------------------------

    save_class_names(class_names)

    print()
    print("-" * 70)
    print("STEP 2: Building model")
    print("-" * 70)

    model = build_model()

    print()
    print("Model successfully created.")

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("STEP 3: Preparing callbacks")
    print("-" * 70)

    callbacks = create_callbacks()

    print(
        f"Number of callbacks: {len(callbacks)}"
    )

    if config.USE_MODEL_CHECKPOINT:
        print(
            "✓ Model checkpoint enabled"
        )

    if config.USE_EARLY_STOPPING:
        print(
            "✓ Early stopping enabled"
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 4: STARTING TRAINING")
    print("=" * 70)

    print()
    print(
        "The existing PawCare model will NOT be overwritten."
    )

    print(
        f"New model location:\n"
        f"{config.MODEL_PATH}"
    )

    print()
    print(
        "Training can take some time on CPU."
    )

    print()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=config.VERBOSE
    )

    # --------------------------------------------------------
    # Save History
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("STEP 5: Saving training results")
    print("-" * 70)

    save_training_history(history)

    # --------------------------------------------------------
    # Save final model if checkpoint wasn't used
    # --------------------------------------------------------

    if not config.USE_MODEL_CHECKPOINT:

        model.save(
            config.MODEL_PATH
        )

        print(
            f"Model saved to:\n"
            f"{config.MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Training Summary
    # --------------------------------------------------------

    print_training_summary(history)

    # --------------------------------------------------------
    # Quick Validation Evaluation
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("STEP 6: Evaluating best model on validation data")
    print("-" * 70)

    best_model = tf.keras.models.load_model(
        config.MODEL_PATH
    )

    validation_loss, validation_accuracy = (
        best_model.evaluate(
            validation_dataset,
            verbose=1
        )
    )

    print()
    print(
        f"Validation loss     : "
        f"{validation_loss:.4f}"
    )

    print(
        f"Validation accuracy : "
        f"{validation_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Final Status
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CAT BREED TRAINING COMPLETED")
    print("=" * 70)

    print()
    print("Generated files:")

    print(
        f"✓ Model:\n"
        f"  {config.MODEL_PATH}"
    )

    print(
        f"✓ Class names:\n"
        f"  {config.CLASS_NAMES_PATH}"
    )

    print(
        f"✓ Training history:\n"
        f"  {config.HISTORY_PATH}"
    )

    print()
    print("STATUS: CAT BREED TRAINING COMPLETED")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        train_model()

    except KeyboardInterrupt:

        print()
        print("=" * 70)
        print("TRAINING INTERRUPTED BY USER")
        print("=" * 70)

        print()
        print(
            "The training process was stopped."
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("CAT BREED TRAINING FAILED")
        print("=" * 70)

        print()
        print("Error:")
        print(error)

        raise