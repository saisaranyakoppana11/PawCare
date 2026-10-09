import tensorflow as tf

import config


# ============================================================
# PAWCARE - CAT VS NON-CAT MODEL BUILDER
# ============================================================


def build_model():

    print("\n" + "=" * 70)
    print("PAWCARE - CAT VS NON-CAT MODEL BUILDING")
    print("=" * 70)

    # --------------------------------------------------------
    # Check local EfficientNetB0 weights
    # --------------------------------------------------------

    if not config.BASE_MODEL_WEIGHTS.exists():

        raise FileNotFoundError(
            "\nLocal EfficientNetB0 weights were not found.\n"
            f"Expected path:\n"
            f"{config.BASE_MODEL_WEIGHTS}"
        )

    print("\nSTEP 1: Checking local pretrained weights")
    print("-" * 70)

    print("✓ Local EfficientNetB0 weights found")

    print(
        f"  File size: "
        f"{config.BASE_MODEL_WEIGHTS.stat().st_size:,} bytes"
    )

    # --------------------------------------------------------
    # Create EfficientNetB0 base model
    # --------------------------------------------------------

    print("\nSTEP 2: Creating EfficientNetB0 architecture")
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

    print("✓ EfficientNetB0 architecture created")

    # --------------------------------------------------------
    # Load local pretrained weights
    # --------------------------------------------------------

    print("\nSTEP 3: Loading local pretrained weights")
    print("-" * 70)

    base_model.load_weights(
        str(config.BASE_MODEL_WEIGHTS)
    )

    print("✓ Local pretrained weights loaded successfully")

    # --------------------------------------------------------
    # Freeze base model
    # --------------------------------------------------------

    base_model.trainable = config.FREEZE_BASE_MODEL

    # The configuration says True for freezing the base model.
    # Therefore explicitly freeze it.
    if config.FREEZE_BASE_MODEL:

        base_model.trainable = False

        print("\nSTEP 4: Freezing EfficientNetB0 base model")
        print("-" * 70)
        print("✓ Base model trainable: False")

    else:

        base_model.trainable = True

        print("\nSTEP 4: EfficientNetB0 base model")
        print("-" * 70)
        print("✓ Base model trainable: True")

    # --------------------------------------------------------
    # Build classification model
    # --------------------------------------------------------

    print("\nSTEP 5: Building classification head")
    print("-" * 70)

    inputs = tf.keras.Input(
        shape=(
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
            config.CHANNELS
        ),
        name="input_image"
    )

    # Keep the pretrained backbone in inference mode because
    # it is frozen.
    x = base_model(
        inputs,
        training=False
    )

    if config.POOLING == "global_average":

        x = tf.keras.layers.GlobalAveragePooling2D(
            name="global_average_pooling"
        )(x)

    else:

        raise ValueError(
            f"Unsupported pooling method: "
            f"{config.POOLING}"
        )

    x = tf.keras.layers.Dropout(
        config.DROPOUT_RATE,
        name="dropout"
    )(x)

    outputs = tf.keras.layers.Dense(
        config.NUM_CLASSES,
        activation="softmax",
        name="classifier"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="PawCare_Cat_Detector"
    )

    # --------------------------------------------------------
    # Compile model
    # --------------------------------------------------------

    print("\nSTEP 6: Compiling model")
    print("-" * 70)

    if config.OPTIMIZER_NAME == "Adam":

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE
        )

    else:

        raise ValueError(
            f"Unsupported optimizer: "
            f"{config.OPTIMIZER_NAME}"
        )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("✓ Model compiled successfully")

    # --------------------------------------------------------
    # Model summary
    # --------------------------------------------------------

    print("\nSTEP 7: Model summary")
    print("-" * 70)

    model.summary()

    # --------------------------------------------------------
    # Parameter counts
    # --------------------------------------------------------

    print("\nSTEP 8: Parameter information")
    print("-" * 70)

    trainable_parameters = sum(
        tf.keras.backend.count_params(variable)
        for variable in model.trainable_weights
    )

    non_trainable_parameters = sum(
        tf.keras.backend.count_params(variable)
        for variable in model.non_trainable_weights
    )

    total_parameters = (
        trainable_parameters
        + non_trainable_parameters
    )

    print(
        f"Trainable parameters     : "
        f"{trainable_parameters:,}"
    )

    print(
        f"Non-trainable parameters : "
        f"{non_trainable_parameters:,}"
    )

    print(
        f"Total parameters         : "
        f"{total_parameters:,}"
    )

    print("\n" + "=" * 70)
    print("STATUS: CAT DETECTOR MODEL BUILT SUCCESSFULLY")
    print("=" * 70)

    return model


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    model = build_model()

    print("\nModel name:")
    print(model.name)

    print("\nOutput shape:")
    print(model.output_shape)

    print("\nFinal status:")
    print("✓ Local EfficientNetB0 weights loaded")
    print("✓ Base model configured")
    print("✓ Classification head created")
    print("✓ Model compiled")
    print("✓ Model builder test completed")