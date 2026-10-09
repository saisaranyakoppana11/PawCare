import tensorflow as tf

from config import (
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
    HORIZONTAL_FLIP,
    ROTATION_FACTOR,
    ZOOM_FACTOR,
    TRANSLATION_HEIGHT,
    TRANSLATION_WIDTH,
    NUM_PARALLEL_CALLS,
    PREFETCH_SIZE,
)


# ============================================================
# PAWCARE - CAT BREED CLASSIFIER
# DATA LOADING PIPELINE
# ============================================================


def create_data_augmentation():
    """
    Create the data augmentation pipeline.

    Augmentation is applied only to training images.
    """

    augmentation_layers = []

    if HORIZONTAL_FLIP:

        augmentation_layers.append(
            tf.keras.layers.RandomFlip(
                mode="horizontal"
            )
        )

    if ROTATION_FACTOR > 0:

        augmentation_layers.append(
            tf.keras.layers.RandomRotation(
                factor=ROTATION_FACTOR
            )
        )

    if ZOOM_FACTOR > 0:

        augmentation_layers.append(
            tf.keras.layers.RandomZoom(
                height_factor=ZOOM_FACTOR,
                width_factor=ZOOM_FACTOR
            )
        )

    if (
        TRANSLATION_HEIGHT > 0
        or TRANSLATION_WIDTH > 0
    ):

        augmentation_layers.append(
            tf.keras.layers.RandomTranslation(
                height_factor=TRANSLATION_HEIGHT,
                width_factor=TRANSLATION_WIDTH
            )
        )

    return tf.keras.Sequential(
        augmentation_layers,
        name="data_augmentation"
    )


def load_datasets():
    """
    Load training, validation, and test datasets.
    """

    print("\n" + "=" * 70)
    print("PAWCARE - CAT BREED DATA LOADING")
    print("=" * 70)

    print("\nTraining directory:")
    print(TRAIN_DIR)

    print("\nValidation directory:")
    print(VALIDATION_DIR)

    print("\nTest directory:")
    print(TEST_DIR)

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("Loading training dataset...")
    print("-" * 70)

    train_dataset = (
        tf.keras.utils.image_dataset_from_directory(
            TRAIN_DIR,
            labels="inferred",
            label_mode="int",
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=RANDOM_SEED,
        )
    )

    # --------------------------------------------------------
    # Get class names
    # --------------------------------------------------------

    class_names = train_dataset.class_names

    print("\nClasses detected:")

    for index, class_name in enumerate(class_names):

        print(
            f"{index:2}: {class_name}"
        )

    print(
        f"\nNumber of classes: "
        f"{len(class_names)}"
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("Loading validation dataset...")
    print("-" * 70)

    validation_dataset = (
        tf.keras.utils.image_dataset_from_directory(
            VALIDATION_DIR,
            labels="inferred",
            label_mode="int",
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    )

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("Loading test dataset...")
    print("-" * 70)

    test_dataset = (
        tf.keras.utils.image_dataset_from_directory(
            TEST_DIR,
            labels="inferred",
            label_mode="int",
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    )

    # --------------------------------------------------------
    # Verify class order
    # --------------------------------------------------------

    if validation_dataset.class_names != class_names:

        raise ValueError(
            "Training and validation class names do not match."
        )

    if test_dataset.class_names != class_names:

        raise ValueError(
            "Training and test class names do not match."
        )

    print("\nClass order verification: PASSED")

    # --------------------------------------------------------
    # Create augmentation
    # --------------------------------------------------------

    data_augmentation = create_data_augmentation()

    print("\nData augmentation:")

    if HORIZONTAL_FLIP:
        print("- Random horizontal flip")

    if ROTATION_FACTOR > 0:
        print(
            f"- Random rotation: "
            f"{ROTATION_FACTOR}"
        )

    if ZOOM_FACTOR > 0:
        print(
            f"- Random zoom: "
            f"{ZOOM_FACTOR}"
        )

    if (
        TRANSLATION_HEIGHT > 0
        or TRANSLATION_WIDTH > 0
    ):

        print(
            "- Random translation: "
            f"height={TRANSLATION_HEIGHT}, "
            f"width={TRANSLATION_WIDTH}"
        )

    # --------------------------------------------------------
    # Apply augmentation only to training data
    # --------------------------------------------------------

    def augment_images(images, labels):

        images = data_augmentation(
            images,
            training=True
        )

        return images, labels

    train_dataset = train_dataset.map(
        augment_images,
        num_parallel_calls=NUM_PARALLEL_CALLS
    )

    # --------------------------------------------------------
    # EfficientNetB0 preprocessing
    # --------------------------------------------------------
    #
    # We intentionally DO NOT use:
    #
    # tf.keras.layers.Rescaling(1./255)
    #
    # EfficientNetB0 includes its own preprocessing.
    #
    # Images remain in the 0-255 range.
    # --------------------------------------------------------

    print("\nEfficientNetB0 preprocessing:")
    print("- No manual Rescaling layer")
    print("- Images remain in 0-255 range")

    # --------------------------------------------------------
    # Prefetch
    # --------------------------------------------------------

    train_dataset = train_dataset.prefetch(
        PREFETCH_SIZE
    )

    validation_dataset = validation_dataset.prefetch(
        PREFETCH_SIZE
    )

    test_dataset = test_dataset.prefetch(
        PREFETCH_SIZE
    )

    print("\nPrefetch size:")
    print(PREFETCH_SIZE)

    print("\n" + "=" * 70)
    print("DATA LOADING COMPLETED")
    print("=" * 70)

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names,
    )


if __name__ == "__main__":

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names,
    ) = load_datasets()

    print("\nFinal dataset information:")
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
        f"Number of classes  : "
        f"{len(class_names)}"
    )

    print("\nSTATUS: DATA LOADER TEST PASSED")