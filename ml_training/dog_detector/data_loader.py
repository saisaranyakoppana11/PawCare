import tensorflow as tf

import config


# ============================================================
# PawCare - Dog vs Non-Dog Detector
# Data Loader
# ============================================================


def create_augmentation():

    augmentation_layers = []

    if config.HORIZONTAL_FLIP:
        augmentation_layers.append(
            tf.keras.layers.RandomFlip(
                mode="horizontal"
            )
        )

    if config.ROTATION_FACTOR > 0:
        augmentation_layers.append(
            tf.keras.layers.RandomRotation(
                factor=config.ROTATION_FACTOR
            )
        )

    if config.ZOOM_FACTOR > 0:
        augmentation_layers.append(
            tf.keras.layers.RandomZoom(
                height_factor=config.ZOOM_FACTOR,
                width_factor=config.ZOOM_FACTOR
            )
        )

    if (
        config.TRANSLATION_HEIGHT > 0
        or config.TRANSLATION_WIDTH > 0
    ):
        augmentation_layers.append(
            tf.keras.layers.RandomTranslation(
                height_factor=config.TRANSLATION_HEIGHT,
                width_factor=config.TRANSLATION_WIDTH
            )
        )

    return tf.keras.Sequential(
        augmentation_layers,
        name="dog_detector_augmentation"
    )


def load_datasets():

    print("=" * 70)
    print("PAWCARE - DOG DETECTOR DATA LOADER")
    print("=" * 70)

    print("\nLoading training dataset...")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        class_names=config.EXPECTED_CLASSES,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE_TRAIN,
        seed=config.RANDOM_SEED
    )

    print("\nLoading validation dataset...")

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        config.VALIDATION_DIR,
        labels="inferred",
        label_mode="int",
        class_names=config.EXPECTED_CLASSES,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE_VALIDATION
    )

    print("\nLoading test dataset...")

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        labels="inferred",
        label_mode="int",
        class_names=config.EXPECTED_CLASSES,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE_TEST
    )

    class_names = train_dataset.class_names

    # --------------------------------------------------------
    # Verify class order
    # --------------------------------------------------------

    if class_names != config.EXPECTED_CLASSES:

        raise ValueError(
            "Class order mismatch.\n"
            f"Expected: {config.EXPECTED_CLASSES}\n"
            f"Found: {class_names}"
        )

    print("\nClass names:")
    print(class_names)

    # --------------------------------------------------------
    # Data augmentation
    # --------------------------------------------------------

    if config.USE_AUGMENTATION:

        augmentation = create_augmentation()

        train_dataset = train_dataset.map(
            lambda images, labels: (
                augmentation(images, training=True),
                labels
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        print("\n✓ Training augmentation enabled")

        print(
            "  Horizontal flip:",
            config.HORIZONTAL_FLIP
        )

        print(
            "  Rotation:",
            config.ROTATION_FACTOR
        )

        print(
            "  Zoom:",
            config.ZOOM_FACTOR
        )

        print(
            "  Translation height:",
            config.TRANSLATION_HEIGHT
        )

        print(
            "  Translation width:",
            config.TRANSLATION_WIDTH
        )

    else:

        print("\n✓ Training augmentation disabled")

    # --------------------------------------------------------
    # Prefetch
    # --------------------------------------------------------

    train_dataset = train_dataset.prefetch(
        config.PREFETCH_BUFFER
    )

    validation_dataset = validation_dataset.prefetch(
        config.PREFETCH_BUFFER
    )

    test_dataset = test_dataset.prefetch(
        config.PREFETCH_BUFFER
    )

    print("\n✓ Prefetch configured")

    print("\n" + "=" * 70)
    print("DATASETS LOADED SUCCESSFULLY")
    print("=" * 70)

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names
    )


def test_data_loader():

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names
    ) = load_datasets()

    print("\n" + "=" * 70)
    print("DATA LOADER VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset cardinality
    # --------------------------------------------------------

    train_batches = tf.data.experimental.cardinality(
        train_dataset
    ).numpy()

    validation_batches = tf.data.experimental.cardinality(
        validation_dataset
    ).numpy()

    test_batches = tf.data.experimental.cardinality(
        test_dataset
    ).numpy()

    print(
        f"\nTraining batches: {train_batches}"
    )

    print(
        f"Validation batches: {validation_batches}"
    )

    print(
        f"Test batches: {test_batches}"
    )

    # --------------------------------------------------------
    # Inspect one training batch
    # --------------------------------------------------------

    train_images, train_labels = next(
        iter(train_dataset)
    )

    print("\nTraining batch:")

    print(
        "  Images shape:",
        train_images.shape
    )

    print(
        "  Labels shape:",
        train_labels.shape
    )

    print(
        "  Image dtype:",
        train_images.dtype
    )

    print(
        "  Label dtype:",
        train_labels.dtype
    )

    # --------------------------------------------------------
    # Expected shapes
    # --------------------------------------------------------

    expected_image_shape = (
        config.BATCH_SIZE,
        config.IMAGE_HEIGHT,
        config.IMAGE_WIDTH,
        config.CHANNELS
    )

    expected_label_shape = (
        config.BATCH_SIZE,
    )

    if train_images.shape != expected_image_shape:

        raise ValueError(
            "Unexpected image shape.\n"
            f"Expected: {expected_image_shape}\n"
            f"Found: {train_images.shape}"
        )

    if train_labels.shape != expected_label_shape:

        raise ValueError(
            "Unexpected label shape.\n"
            f"Expected: {expected_label_shape}\n"
            f"Found: {train_labels.shape}"
        )

    print("\n✓ Image shape verification passed")

    print("✓ Label shape verification passed")

    # --------------------------------------------------------
    # Label range
    # --------------------------------------------------------

    min_label = tf.reduce_min(
        train_labels
    ).numpy()

    max_label = tf.reduce_max(
        train_labels
    ).numpy()

    print(
        f"\nLabel range: {min_label} to {max_label}"
    )

    if min_label < 0 or max_label >= config.NUM_CLASSES:

        raise ValueError(
            "Invalid label values detected."
        )

    print("✓ Label range verification passed")

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STATUS: DATA LOADER TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":

    test_data_loader()