from pathlib import Path

import tensorflow as tf

import config


# ============================================================
# PawCare - Dog Breed Classifier
# Data Loader
# ============================================================


# ============================================================
# Reproducibility
# ============================================================

tf.random.set_seed(
    config.RANDOM_SEED
)


# ============================================================
# Data Augmentation
# ============================================================

def create_augmentation():

    if not config.USE_AUGMENTATION:

        return tf.keras.Sequential(
            name="no_augmentation"
        )

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
        name="data_augmentation"
    )


# ============================================================
# Dataset Loader
# ============================================================

def load_datasets():

    print("=" * 70)
    print("PAWCARE - DOG BREED CLASSIFIER DATA LOADER")
    print("=" * 70)

    # --------------------------------------------------------
    # Training Dataset
    # --------------------------------------------------------

    print("\nLoading training dataset...")

    train_dataset = (
        tf.keras.utils.image_dataset_from_directory(
            config.TRAIN_DIR,
            labels="inferred",
            label_mode="int",
            class_names=config.EXPECTED_CLASSES,
            image_size=config.IMAGE_SIZE,
            batch_size=config.BATCH_SIZE,
            shuffle=config.SHUFFLE_TRAIN,
            seed=config.RANDOM_SEED
        )
    )

    # --------------------------------------------------------
    # Validation Dataset
    # --------------------------------------------------------

    print("\nLoading validation dataset...")

    validation_dataset = (
        tf.keras.utils.image_dataset_from_directory(
            config.VALIDATION_DIR,
            labels="inferred",
            label_mode="int",
            class_names=config.EXPECTED_CLASSES,
            image_size=config.IMAGE_SIZE,
            batch_size=config.BATCH_SIZE,
            shuffle=config.SHUFFLE_VALIDATION
        )
    )

    # --------------------------------------------------------
    # Test Dataset
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    test_dataset = (
        tf.keras.utils.image_dataset_from_directory(
            config.TEST_DIR,
            labels="inferred",
            label_mode="int",
            class_names=config.EXPECTED_CLASSES,
            image_size=config.IMAGE_SIZE,
            batch_size=config.BATCH_SIZE,
            shuffle=config.SHUFFLE_TEST
        )
    )

    # --------------------------------------------------------
    # Verify Class Names
    # --------------------------------------------------------

    class_names = (
        train_dataset.class_names
    )

    print(
        f"\nClass names:\n"
        f"{class_names}"
    )

    if class_names != config.EXPECTED_CLASSES:

        raise ValueError(
            "\nClass order mismatch!\n"
            f"Expected: "
            f"{config.EXPECTED_CLASSES}\n"
            f"Found: {class_names}"
        )

    if (
        validation_dataset.class_names
        != config.EXPECTED_CLASSES
    ):

        raise ValueError(
            "\nValidation class order mismatch!"
        )

    if (
        test_dataset.class_names
        != config.EXPECTED_CLASSES
    ):

        raise ValueError(
            "\nTest class order mismatch!"
        )

    print(
        "\n✓ Class order verified"
    )

    # --------------------------------------------------------
    # Data Augmentation
    # --------------------------------------------------------

    augmentation = (
        create_augmentation()
    )

    if config.USE_AUGMENTATION:

        print(
            "\n✓ Training augmentation enabled"
        )

        print(
            f"  Horizontal flip: "
            f"{config.HORIZONTAL_FLIP}"
        )

        print(
            f"  Rotation: "
            f"{config.ROTATION_FACTOR}"
        )

        print(
            f"  Zoom: "
            f"{config.ZOOM_FACTOR}"
        )

        print(
            f"  Translation height: "
            f"{config.TRANSLATION_HEIGHT}"
        )

        print(
            f"  Translation width: "
            f"{config.TRANSLATION_WIDTH}"
        )

    else:

        print(
            "\n✓ Training augmentation disabled"
        )

    # --------------------------------------------------------
    # Apply Augmentation to Training Only
    # --------------------------------------------------------

    train_dataset = train_dataset.map(
        lambda images, labels: (
            augmentation(images, training=True),
            labels
        ),
        num_parallel_calls=1
    )

    # --------------------------------------------------------
    # Validation and Test Datasets
    # --------------------------------------------------------
    # No augmentation is applied to validation/test data.
    # This ensures evaluation uses the original images.
    # --------------------------------------------------------

    print(
        "\n✓ Validation augmentation: disabled"
    )

    print(
        "✓ Test augmentation: disabled"
    )

    # --------------------------------------------------------
    # Prefetch
    # --------------------------------------------------------

    train_dataset = train_dataset.prefetch(
        config.PREFETCH_BUFFER
    )

    validation_dataset = (
        validation_dataset.prefetch(
            config.PREFETCH_BUFFER
        )
    )

    test_dataset = test_dataset.prefetch(
        config.PREFETCH_BUFFER
    )

    print(
        "\n✓ Prefetch configured"
    )

    print("\n" + "=" * 70)
    print("DATASETS LOADED SUCCESSFULLY")
    print("=" * 70)

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names
    )


# ============================================================
# Data Loader Test
# ============================================================

def test_data_loader():

    print("\n")
    print("=" * 70)
    print("DATA LOADER TEST")
    print("=" * 70)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names
    ) = load_datasets()

    # --------------------------------------------------------
    # Dataset Cardinality
    # --------------------------------------------------------

    train_batches = (
        tf.data.experimental.cardinality(
            train_dataset
        ).numpy()
    )

    validation_batches = (
        tf.data.experimental.cardinality(
            validation_dataset
        ).numpy()
    )

    test_batches = (
        tf.data.experimental.cardinality(
            test_dataset
        ).numpy()
    )

    print(
        f"\nTraining batches: "
        f"{train_batches}"
    )

    print(
        f"Validation batches: "
        f"{validation_batches}"
    )

    print(
        f"Test batches: "
        f"{test_batches}"
    )

    # --------------------------------------------------------
    # Inspect Training Batch
    # --------------------------------------------------------

    train_images, train_labels = (
        next(iter(train_dataset))
    )

    print(
        f"\nTraining image batch shape: "
        f"{train_images.shape}"
    )

    print(
        f"Training label batch shape: "
        f"{train_labels.shape}"
    )

    print(
        f"Image dtype: "
        f"{train_images.dtype}"
    )

    print(
        f"Label dtype: "
        f"{train_labels.dtype}"
    )

    # --------------------------------------------------------
    # Validate Image Shape
    # --------------------------------------------------------

    expected_image_shape = (
        config.BATCH_SIZE,
        config.IMAGE_HEIGHT,
        config.IMAGE_WIDTH,
        config.CHANNELS
    )

    actual_image_shape = (
        tuple(train_images.shape)
    )

    if actual_image_shape != expected_image_shape:

        raise ValueError(
            "\nUnexpected image batch shape!\n"
            f"Expected: "
            f"{expected_image_shape}\n"
            f"Found: {actual_image_shape}"
        )

    print(
        "\n✓ Image shape verified"
    )

    # --------------------------------------------------------
    # Validate Labels
    # --------------------------------------------------------

    minimum_label = int(
        tf.reduce_min(
            train_labels
        ).numpy()
    )

    maximum_label = int(
        tf.reduce_max(
            train_labels
        ).numpy()
    )

    print(
        f"\nLabel range: "
        f"{minimum_label} "
        f"to "
        f"{maximum_label}"
    )

    if minimum_label < 0:

        raise ValueError(
            "Invalid negative label detected."
        )

    if maximum_label >= config.NUM_CLASSES:

        raise ValueError(
            "Label exceeds configured "
            "number of classes."
        )

    print(
        "✓ Label range verified"
    )

    # --------------------------------------------------------
    # Validate Dataset Sizes
    # --------------------------------------------------------

    expected_train_images = 400
    expected_validation_images = 50
    expected_test_images = 50

    expected_train_batches = (
        (
            expected_train_images
            + config.BATCH_SIZE
            - 1
        )
        // config.BATCH_SIZE
    )

    expected_validation_batches = (
        (
            expected_validation_images
            + config.BATCH_SIZE
            - 1
        )
        // config.BATCH_SIZE
    )

    expected_test_batches = (
        (
            expected_test_images
            + config.BATCH_SIZE
            - 1
        )
        // config.BATCH_SIZE
    )

    if train_batches != expected_train_batches:

        raise ValueError(
            "\nUnexpected training "
            "batch count."
        )

    if (
        validation_batches
        != expected_validation_batches
    ):

        raise ValueError(
            "\nUnexpected validation "
            "batch count."
        )

    if (
        test_batches
        != expected_test_batches
    ):

        raise ValueError(
            "\nUnexpected test "
            "batch count."
        )

    print(
        "\n✓ Dataset sizes verified"
    )

    # --------------------------------------------------------
    # Final Status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STATUS: DATA LOADER TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":

    test_data_loader()