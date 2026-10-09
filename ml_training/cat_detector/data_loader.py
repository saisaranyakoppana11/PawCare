import tensorflow as tf

import config


# ============================================================
# PAWCARE - CAT VS NON-CAT DATA LOADER
# ============================================================


# ------------------------------------------------------------
# DATA AUGMENTATION
# ------------------------------------------------------------

def create_augmentation():

    if not config.USE_AUGMENTATION:

        print("\nData augmentation: DISABLED")

        return None

    print("\nData augmentation:")
    print("- Random horizontal flip")
    print(
        f"- Random rotation: "
        f"{config.ROTATION_FACTOR}"
    )
    print(
        f"- Random zoom: "
        f"{config.ZOOM_FACTOR}"
    )
    print(
        f"- Random translation: "
        f"height={config.TRANSLATION_HEIGHT}, "
        f"width={config.TRANSLATION_WIDTH}"
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
        name="cat_detector_augmentation"
    )


# ------------------------------------------------------------
# DATASET PERFORMANCE
# ------------------------------------------------------------

def prepare_dataset(dataset):

    return dataset.prefetch(
        buffer_size=config.PREFETCH_BUFFER
    )


# ------------------------------------------------------------
# LOAD DATASETS
# ------------------------------------------------------------

def load_datasets():

    print("\n" + "=" * 70)
    print("PAWCARE - CAT VS NON-CAT DATA LOADING")
    print("=" * 70)

    print("\nTraining directory:")
    print(config.TRAIN_DIR)

    print("\nValidation directory:")
    print(config.VALIDATION_DIR)

    print("\nTest directory:")
    print(config.TEST_DIR)

    # --------------------------------------------------------
    # Verify directories
    # --------------------------------------------------------

    if not config.TRAIN_DIR.exists():

        raise FileNotFoundError(
            f"Training directory not found:\n"
            f"{config.TRAIN_DIR}"
        )

    if not config.VALIDATION_DIR.exists():

        raise FileNotFoundError(
            f"Validation directory not found:\n"
            f"{config.VALIDATION_DIR}"
        )

    if not config.TEST_DIR.exists():

        raise FileNotFoundError(
            f"Test directory not found:\n"
            f"{config.TEST_DIR}"
        )

    # --------------------------------------------------------
    # Load training dataset
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("Loading training dataset...")
    print("-" * 70)

    train_dataset = (
        tf.keras.utils.image_dataset_from_directory(
            config.TRAIN_DIR,
            labels="inferred",
            label_mode="int",
            image_size=config.IMAGE_SIZE,
            batch_size=config.BATCH_SIZE,
            shuffle=config.SHUFFLE_TRAIN,
            seed=config.RANDOM_SEED
        )
    )

    class_names = train_dataset.class_names

    print("\nClasses detected:")

    for index, class_name in enumerate(class_names):

        print(
            f" {index}: {class_name}"
        )

    print(
        f"\nNumber of classes: "
        f"{len(class_names)}"
    )

    # --------------------------------------------------------
    # Verify class order
    # --------------------------------------------------------

    if class_names != config.EXPECTED_CLASSES:

        raise ValueError(
            "\nClass order mismatch!\n"
            f"Expected: {config.EXPECTED_CLASSES}\n"
            f"Found:    {class_names}"
        )

    print(
        "\nClass order verification: PASSED"
    )

    # --------------------------------------------------------
    # Load validation dataset
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("Loading validation dataset...")
    print("-" * 70)

    validation_dataset = (
        tf.keras.utils.image_dataset_from_directory(
            config.VALIDATION_DIR,
            labels="inferred",
            label_mode="int",
            class_names=class_names,
            image_size=config.IMAGE_SIZE,
            batch_size=config.BATCH_SIZE,
            shuffle=config.SHUFFLE_VALIDATION,
            seed=config.RANDOM_SEED
        )
    )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("Loading test dataset...")
    print("-" * 70)

    test_dataset = (
        tf.keras.utils.image_dataset_from_directory(
            config.TEST_DIR,
            labels="inferred",
            label_mode="int",
            class_names=class_names,
            image_size=config.IMAGE_SIZE,
            batch_size=config.BATCH_SIZE,
            shuffle=config.SHUFFLE_TEST,
            seed=config.RANDOM_SEED
        )
    )

    # --------------------------------------------------------
    # Data augmentation
    #
    # IMPORTANT:
    # Augmentation is applied ONLY to training data.
    # Validation and test data remain unchanged.
    # --------------------------------------------------------

    augmentation = create_augmentation()

    if augmentation is not None:

        train_dataset = train_dataset.map(
            lambda images, labels: (
                augmentation(
                    images,
                    training=True
                ),
                labels
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # --------------------------------------------------------
    # Prefetch datasets
    # --------------------------------------------------------

    train_dataset = prepare_dataset(
        train_dataset
    )

    validation_dataset = prepare_dataset(
        validation_dataset
    )

    test_dataset = prepare_dataset(
        test_dataset
    )

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("Dataset preparation completed")
    print("-" * 70)

    print(
        f"Training batches   : "
        f"{tf.data.experimental.cardinality(train_dataset).numpy()}"
    )

    print(
        f"Validation batches : "
        f"{tf.data.experimental.cardinality(validation_dataset).numpy()}"
    )

    print(
        f"Test batches       : "
        f"{tf.data.experimental.cardinality(test_dataset).numpy()}"
    )

    print(
        "\n✓ Augmentation applied only to training data"
    )

    print(
        "✓ Validation data kept unchanged"
    )

    print(
        "✓ Test data kept unchanged"
    )

    print(
        "✓ Dataset prefetching configured"
    )

    print("\n" + "=" * 70)
    print(
        "STATUS: DATA LOADER TEST PASSED"
    )
    print("=" * 70)

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names
    )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names
    ) = load_datasets()

    print("\n" + "=" * 70)
    print("DATA LOADER VERIFICATION")
    print("=" * 70)

    print(
        f"\nClasses: {class_names}"
    )

    # --------------------------------------------------------
    # Inspect one training batch
    # --------------------------------------------------------

    train_images, train_labels = next(
        iter(train_dataset)
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
        f"Training image dtype: "
        f"{train_images.dtype}"
    )

    print(
        f"Training label dtype: "
        f"{train_labels.dtype}"
    )

    # --------------------------------------------------------
    # Verify image dimensions
    # --------------------------------------------------------

    expected_shape = (
        config.BATCH_SIZE,
        config.IMAGE_HEIGHT,
        config.IMAGE_WIDTH,
        config.CHANNELS
    )

    actual_shape = train_images.shape

    print(
        f"\nExpected batch shape: "
        f"{expected_shape}"
    )

    print(
        f"Actual batch shape:   "
        f"{actual_shape}"
    )

    if (
        actual_shape[1] != config.IMAGE_HEIGHT
        or actual_shape[2] != config.IMAGE_WIDTH
        or actual_shape[3] != config.CHANNELS
    ):

        raise ValueError(
            "\nImage shape verification failed."
        )

    print(
        "\n✓ Image shape verification passed"
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "STATUS: DATA LOADER VERIFICATION PASSED"
    )
    print("=" * 70)