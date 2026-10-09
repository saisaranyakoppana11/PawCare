import tensorflow as tf

import config


# ============================================================
# PawCare - Dog Breed Classifier
# Model Builder
# ============================================================


def build_model():

    print("=" * 70)
    print("PAWCARE - DOG BREED CLASSIFIER")
    print("MODEL BUILDER")
    print("=" * 70)

    # --------------------------------------------------------
    # Load EfficientNetB0 Architecture
    # --------------------------------------------------------

    print(
        "\nLoading EfficientNetB0 architecture..."
    )

    base_model = (
        tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(
                config.IMAGE_HEIGHT,
                config.IMAGE_WIDTH,
                config.CHANNELS
            )
        )
    )

    print(
        "✓ EfficientNetB0 architecture created"
    )

    # --------------------------------------------------------
    # Load Local Pretrained Weights
    # --------------------------------------------------------

    weights_path = config.BASE_MODEL_WEIGHTS

    print(
        f"\nLoading local pretrained weights:\n"
        f"{weights_path}"
    )

    base_model.load_weights(
        weights_path
    )

    print(
        "✓ Local EfficientNetB0 weights loaded"
    )

    # --------------------------------------------------------
    # Freeze Base Model
    # --------------------------------------------------------

    base_model.trainable = (
        not config.FREEZE_BASE_MODEL
    )

    if config.FREEZE_BASE_MODEL:

        print(
            "\n✓ EfficientNetB0 base model frozen"
        )

    else:

        print(
            "\n✓ EfficientNetB0 base model trainable"
        )

    # --------------------------------------------------------
    # Classification Head
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
        name="dog_breed_output"
    )(x)

    # --------------------------------------------------------
    # Create Model
    # --------------------------------------------------------

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=config.MODEL_NAME
    )

    # --------------------------------------------------------
    # Compile Model
    # --------------------------------------------------------

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(
        "\n✓ Model compiled successfully"
    )

    # --------------------------------------------------------
    # Model Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    model.summary()

    return model


# ============================================================
# Model Test
# ============================================================

def test_model():

    print("\n")
    print("=" * 70)
    print("MODEL BUILDER TEST")
    print("=" * 70)

    model = build_model()

    # --------------------------------------------------------
    # Verify Input Shape
    # --------------------------------------------------------

    expected_input_shape = (
        None,
        config.IMAGE_HEIGHT,
        config.IMAGE_WIDTH,
        config.CHANNELS
    )

    actual_input_shape = tuple(
        model.input_shape
    )

    print(
        f"\nInput shape:\n"
        f"{actual_input_shape}"
    )

    if actual_input_shape != expected_input_shape:

        raise ValueError(
            "\nUnexpected model input shape!\n"
            f"Expected: {expected_input_shape}\n"
            f"Found: {actual_input_shape}"
        )

    print(
        "✓ Input shape verified"
    )

    # --------------------------------------------------------
    # Verify Output Shape
    # --------------------------------------------------------

    expected_output_shape = (
        None,
        config.NUM_CLASSES
    )

    actual_output_shape = tuple(
        model.output_shape
    )

    print(
        f"\nOutput shape:\n"
        f"{actual_output_shape}"
    )

    if actual_output_shape != expected_output_shape:

        raise ValueError(
            "\nUnexpected model output shape!\n"
            f"Expected: {expected_output_shape}\n"
            f"Found: {actual_output_shape}"
        )

    print(
        "✓ Output shape verified"
    )

    # --------------------------------------------------------
    # Verify Base Model Freezing
    # --------------------------------------------------------

    base_model = None

    for layer in model.layers:

        if layer.name.startswith(
            "efficientnet"
        ):

            base_model = layer
            break

    if base_model is None:

        raise ValueError(
            "\nEfficientNetB0 base model "
            "could not be located."
        )

    print(
        f"\nBase model trainable: "
        f"{base_model.trainable}"
    )

    if config.FREEZE_BASE_MODEL:

        if base_model.trainable:

            raise ValueError(
                "\nBase model should be frozen."
            )

        print(
            "✓ Base model freezing verified"
        )

    # --------------------------------------------------------
    # Verify Parameters
    # --------------------------------------------------------

    total_parameters = model.count_params()

    trainable_parameters = sum(
        variable.shape.num_elements()
        for variable in model.trainable_variables
    )

    non_trainable_parameters = (
        total_parameters
        - trainable_parameters
    )

    print(
        f"\nTotal parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Non-trainable parameters: "
        f"{non_trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Dummy Prediction
    # --------------------------------------------------------

    print(
        "\nRunning dummy prediction..."
    )

    dummy_input = tf.random.uniform(
        shape=(
            1,
            config.IMAGE_HEIGHT,
            config.IMAGE_WIDTH,
            config.CHANNELS
        ),
        minval=0,
        maxval=255,
        dtype=tf.float32
    )

    prediction = model.predict(
        dummy_input,
        verbose=0
    )

    print(
        f"Prediction shape: "
        f"{prediction.shape}"
    )

    if prediction.shape != (
        1,
        config.NUM_CLASSES
    ):

        raise ValueError(
            "\nUnexpected prediction shape."
        )

    print(
        "✓ Prediction shape verified"
    )

    # --------------------------------------------------------
    # Verify Softmax
    # --------------------------------------------------------

    probability_sum = float(
        tf.reduce_sum(
            prediction[0]
        ).numpy()
    )

    print(
        f"\nSoftmax probability sum: "
        f"{probability_sum:.6f}"
    )

    if abs(
        probability_sum - 1.0
    ) > 0.001:

        raise ValueError(
            "\nSoftmax probabilities do not "
            "sum to approximately 1."
        )

    print(
        "✓ Softmax output verified"
    )

    # --------------------------------------------------------
    # Final Status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STATUS: MODEL BUILDER TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":

    test_model()