import tensorflow as tf

import config


# ============================================================
# PawCare - Dog vs Non-Dog Detector
# Model Builder
# ============================================================


def build_model():

    print("=" * 70)
    print("PAWCARE - DOG DETECTOR MODEL BUILDER")
    print("=" * 70)

    print("\nCreating EfficientNetB0 base model...")

    # --------------------------------------------------------
    # Create EfficientNetB0 without classification head
    # --------------------------------------------------------

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

    print("\nLoading local EfficientNetB0 weights...")

    base_model.load_weights(
        config.BASE_MODEL_WEIGHTS
    )

    print("✓ Local EfficientNetB0 weights loaded")

    # --------------------------------------------------------
    # Freeze base model
    # --------------------------------------------------------

    base_model.trainable = (
        not config.FREEZE_BASE_MODEL
    )

    if config.FREEZE_BASE_MODEL:

        print("✓ EfficientNetB0 base model is frozen")

    else:

        print("✓ EfficientNetB0 base model is trainable")

    # --------------------------------------------------------
    # Build classification model
    # --------------------------------------------------------

    inputs = tf.keras.Input(
        shape=(
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
            config.CHANNELS
        ),
        name="input_image"
    )

    x = base_model(
        inputs,
        training=False
    )

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = tf.keras.layers.Dropout(
        config.DROPOUT_RATE,
        name="dropout"
    )(x)

    outputs = tf.keras.layers.Dense(
        config.NUM_CLASSES,
        activation="softmax",
        name="classification_output"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=config.MODEL_NAME
    )

    # --------------------------------------------------------
    # Compile model
    # --------------------------------------------------------

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("✓ Classification head created")

    print("✓ Model compiled")

    return model


def print_model_information(model):

    print("\n" + "=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)

    print(
        f"\nModel name: {model.name}"
    )

    print(
        f"Input shape: {model.input_shape}"
    )

    print(
        f"Output shape: {model.output_shape}"
    )

    print(
        f"Total parameters: {model.count_params():,}"
    )

    trainable_parameters = sum(
        tf.keras.backend.count_params(variable)
        for variable in model.trainable_variables
    )

    non_trainable_parameters = sum(
        tf.keras.backend.count_params(variable)
        for variable in model.non_trainable_variables
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Non-trainable parameters: "
        f"{non_trainable_parameters:,}"
    )

    print("\nModel summary:\n")

    model.summary()


def test_model():

    model = build_model()

    print_model_information(model)

    # --------------------------------------------------------
    # Verify input shape
    # --------------------------------------------------------

    expected_input_shape = (
        None,
        config.IMAGE_HEIGHT,
        config.IMAGE_WIDTH,
        config.CHANNELS
    )

    if model.input_shape != expected_input_shape:

        raise ValueError(
            "Unexpected model input shape.\n"
            f"Expected: {expected_input_shape}\n"
            f"Found: {model.input_shape}"
        )

    print(
        "\n✓ Input shape verification passed"
    )

    # --------------------------------------------------------
    # Verify output shape
    # --------------------------------------------------------

    expected_output_shape = (
        None,
        config.NUM_CLASSES
    )

    if model.output_shape != expected_output_shape:

        raise ValueError(
            "Unexpected model output shape.\n"
            f"Expected: {expected_output_shape}\n"
            f"Found: {model.output_shape}"
        )

    print(
        "✓ Output shape verification passed"
    )

    # --------------------------------------------------------
    # Verify base model freezing
    # --------------------------------------------------------

    if config.FREEZE_BASE_MODEL:

        base_model = None

        for layer in model.layers:

            if isinstance(
                layer,
                tf.keras.Model
            ):

                base_model = layer
                break

        if base_model is not None:

            if base_model.trainable:

                raise ValueError(
                    "EfficientNetB0 base model "
                    "should be frozen."
                )

            print(
                "✓ Base model freezing verification passed"
            )

    # --------------------------------------------------------
    # Verify output activation
    # --------------------------------------------------------

    output_layer = model.get_layer(
        "classification_output"
    )

    activation_name = (
        output_layer.activation.__name__
    )

    if activation_name != "softmax":

        raise ValueError(
            "Output layer must use softmax activation."
        )

    print(
        "✓ Softmax output verification passed"
    )

    # --------------------------------------------------------
    # Test prediction
    # --------------------------------------------------------

    dummy_input = tf.zeros(
        (
            1,
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
            config.CHANNELS
        ),
        dtype=tf.float32
    )

    prediction = model(
        dummy_input,
        training=False
    )

    print(
        "\nDummy prediction shape:",
        prediction.shape
    )

    if prediction.shape != (
        1,
        config.NUM_CLASSES
    ):

        raise ValueError(
            "Unexpected prediction shape."
        )

    print(
        "✓ Prediction shape verification passed"
    )

    # --------------------------------------------------------
    # Verify probability sum
    # --------------------------------------------------------

    probability_sum = tf.reduce_sum(
        prediction,
        axis=1
    ).numpy()[0]

    print(
        f"Dummy prediction probability sum: "
        f"{probability_sum:.6f}"
    )

    if abs(
        float(probability_sum) - 1.0
    ) > 0.001:

        raise ValueError(
            "Softmax probabilities do not sum to 1."
        )

    print(
        "✓ Softmax probability verification passed"
    )

    print("\n" + "=" * 70)
    print("STATUS: MODEL BUILDER TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":

    test_model()