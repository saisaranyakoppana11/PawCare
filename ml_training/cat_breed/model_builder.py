"""
PawCare - Cat Breed Classifier
Model Architecture Builder

Uses the local pretrained EfficientNetB0 weights stored
inside the PawCare project.

Architecture:

Input
  ↓
EfficientNetB0
  ↓
Global Average Pooling
  ↓
Dropout
  ↓
Dense 11
  ↓
Softmax
"""

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


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model():
    """
    Build the PawCare Cat Breed Classifier.

    EfficientNetB0 is used as the frozen feature extractor.
    The classification head is trained for 11 cat breeds.
    """

    print()
    print("=" * 70)
    print("PAWCARE - CAT BREED MODEL BUILDER")
    print("=" * 70)

    print()
    print("Model configuration:")
    print("-" * 70)

    print(f"Base model       : {config.BASE_MODEL_NAME}")
    print(f"Local weights    : {config.BASE_MODEL_WEIGHTS}")
    print(
        f"Image size       : "
        f"{config.IMAGE_HEIGHT} x {config.IMAGE_WIDTH}"
    )
    print(f"Number of classes: {config.NUM_CLASSES}")
    print(f"Pooling          : {config.POOLING}")
    print(f"Dropout rate     : {config.DROPOUT_RATE}")
    print(f"Optimizer        : {config.OPTIMIZER_NAME}")
    print(f"Learning rate    : {config.LEARNING_RATE}")
    print(f"Freeze base      : {config.FREEZE_BASE_MODEL}")

    # ========================================================
    # VERIFY LOCAL WEIGHTS
    # ========================================================

    print()
    print("-" * 70)
    print("Checking local EfficientNetB0 weights...")
    print("-" * 70)

    if not config.BASE_MODEL_WEIGHTS.exists():
        raise FileNotFoundError(
            "EfficientNetB0 weights were not found:\n"
            f"{config.BASE_MODEL_WEIGHTS}"
        )

    print("Local EfficientNetB0 weights found.")

    print(
        f"Weight file size: "
        f"{config.BASE_MODEL_WEIGHTS.stat().st_size:,} bytes"
    )

    # ========================================================
    # CREATE EFFICIENTNETB0 WITHOUT EXTERNAL DOWNLOAD
    # ========================================================

    print()
    print("-" * 70)
    print("Creating EfficientNetB0 architecture...")
    print("-" * 70)

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
            config.CHANNELS
        )
    )

    print("EfficientNetB0 architecture created.")

    # ========================================================
    # LOAD LOCAL PRETRAINED WEIGHTS
    # ========================================================

    print()
    print("-" * 70)
    print("Loading local pretrained weights...")
    print("-" * 70)

    base_model.load_weights(
        str(config.BASE_MODEL_WEIGHTS)
    )

    print(
        "Local pretrained weights loaded successfully."
    )

    # ========================================================
    # FREEZE BASE MODEL
    # ========================================================

    base_model.trainable = (
        not config.FREEZE_BASE_MODEL
    )

    print()
    print(
        f"Base model trainable: "
        f"{base_model.trainable}"
    )

    # ========================================================
    # INPUT LAYER
    # ========================================================

    inputs = tf.keras.Input(
        shape=(
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
            config.CHANNELS
        ),
        name="input_image"
    )

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    x = base_model(
        inputs,
        training=False
    )

    # ========================================================
    # GLOBAL AVERAGE POOLING
    # ========================================================

    if config.POOLING == "global_average":

        x = tf.keras.layers.GlobalAveragePooling2D(
            name="global_average_pooling"
        )(x)

    else:

        raise ValueError(
            f"Unsupported pooling type: "
            f"{config.POOLING}"
        )

    # ========================================================
    # DROPOUT
    # ========================================================

    x = tf.keras.layers.Dropout(
        config.DROPOUT_RATE,
        name="dropout"
    )(x)

    # ========================================================
    # OUTPUT LAYER
    # ========================================================

    outputs = tf.keras.layers.Dense(
        config.NUM_CLASSES,
        activation="softmax",
        name="cat_breed_output"
    )(x)

    # ========================================================
    # COMPLETE MODEL
    # ========================================================

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="PawCare_Cat_Breed_Classifier"
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    if config.OPTIMIZER_NAME.lower() == "adam":

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE
        )

    else:

        raise ValueError(
            f"Unsupported optimizer: "
            f"{config.OPTIMIZER_NAME}"
        )

    # ========================================================
    # COMPILE MODEL
    # ========================================================

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# MODEL INFORMATION
# ============================================================

def print_model_information(model):
    """
    Display model architecture and official Keras
    parameter counts.
    """

    print()
    print("=" * 70)
    print("MODEL ARCHITECTURE")
    print("=" * 70)

    model.summary()

    print()
    print("-" * 70)
    print("PARAMETER SUMMARY")
    print("-" * 70)

    print(
        f"Trainable parameters    : "
        f"{model.trainable_params:,}"
        if hasattr(model, "trainable_params")
        else "Trainable parameters    : calculated by Keras below"
    )

    # Keras' official parameter counts
    trainable_parameters = sum(
        variable.numpy().size
        for variable in model.trainable_variables
    )

    non_trainable_parameters = sum(
        variable.numpy().size
        for variable in model.non_trainable_variables
    )

    total_parameters = (
        trainable_parameters
        + non_trainable_parameters
    )

    print(
        f"Trainable parameters    : "
        f"{trainable_parameters:,}"
    )

    print(
        f"Non-trainable parameters: "
        f"{non_trainable_parameters:,}"
    )

    print(
        f"Total parameters        : "
        f"{total_parameters:,}"
    )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    try:

        model = build_model()

        print_model_information(model)

        print()
        print("=" * 70)
        print("MODEL BUILD TEST PASSED")
        print("=" * 70)

        print()
        print("The Cat Breed model successfully:")
        print("  ✓ Created EfficientNetB0 architecture")
        print("  ✓ Loaded local pretrained weights")
        print("  ✓ Frozen the EfficientNetB0 base")
        print("  ✓ Added Global Average Pooling")
        print("  ✓ Added Dropout")
        print("  ✓ Added 11-class Softmax output")
        print("  ✓ Compiled with Adam optimizer")

        print()
        print(
            "STATUS: LOCAL WEIGHT MODEL TEST PASSED"
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("MODEL BUILD TEST FAILED")
        print("=" * 70)

        print()
        print("Error:")
        print(error)

        raise