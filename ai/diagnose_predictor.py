"""
PawCare AI Predictor Diagnostic

Purpose:
    Diagnose why different test images are producing
    identical predictions.

This file does NOT modify the existing predictor.
It directly checks:

    1. Image file contents
    2. Image dimensions
    3. Pixel statistics
    4. Cat detector raw predictions
    5. Dog detector raw predictions
    6. Cat breed raw predictions
    7. Dog breed raw predictions
"""

from pathlib import Path
import sys
import hashlib

import numpy as np
import tensorflow as tf


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_IMAGES_DIR = (
    PROJECT_ROOT
    / "ai"
    / "test_images"
)

ML_TRAINING_DIR = (
    PROJECT_ROOT
    / "ml_training"
)


# ============================================================
# Model Paths
# ============================================================

CAT_DETECTOR_MODEL = (
    ML_TRAINING_DIR
    / "cat_detector"
    / "trained_model"
    / "cat_detector.keras"
)

CAT_BREED_MODEL = (
    ML_TRAINING_DIR
    / "cat_breed"
    / "trained_model"
    / "cat_breed_classifier.keras"
)

DOG_DETECTOR_MODEL = (
    ML_TRAINING_DIR
    / "dog_detector"
    / "trained_model"
    / "dog_detector.keras"
)

DOG_BREED_MODEL = (
    ML_TRAINING_DIR
    / "dog_breed"
    / "trained_model"
    / "dog_breed_classifier.keras"
)


# ============================================================
# Class Names
# ============================================================

CAT_DETECTOR_CLASSES = [
    "cat",
    "non_cat"
]

CAT_BREED_CLASSES = [
    "Abyssinian",
    "Bengal",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx"
]

DOG_DETECTOR_CLASSES = [
    "dog",
    "non_dog"
]

DOG_BREED_CLASSES = [
    "beagle",
    "boxer",
    "German_shepherd",
    "golden_retriever",
    "pug",
    "shitzu",
    "Siberian_husky",
    "standard_poodle",
    "whippet",
    "Yorkshire_terrier"
]


# ============================================================
# Test Images
# ============================================================

TEST_IMAGES = [
    (
        "CAT TEST",
        TEST_IMAGES_DIR / "cat_test.jpg"
    ),
    (
        "DOG TEST",
        TEST_IMAGES_DIR / "dog_test.jpg"
    ),
    (
        "NON-PET TEST",
        TEST_IMAGES_DIR / "non_pet_test.jpg"
    )
]


# ============================================================
# Utility Functions
# ============================================================

def check_file(path, description):

    if not path.exists():

        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )


def file_hash(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def load_image(image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224),
        color_mode="rgb"
    )

    array = tf.keras.utils.img_to_array(
        image
    )

    array = np.asarray(
        array,
        dtype=np.float32
    )

    batch = np.expand_dims(
        array,
        axis=0
    )

    return array, batch


def print_prediction(
    title,
    predictions,
    class_names
):

    predictions = np.asarray(
        predictions,
        dtype=np.float32
    ).reshape(-1)

    print()
    print(title)
    print("-" * 70)

    for index, probability in enumerate(
        predictions
    ):

        if index < len(class_names):

            class_name = class_names[index]

        else:

            class_name = f"INDEX_{index}"

        print(
            f"{class_name:<25} "
            f"{float(probability):.10f}"
        )

    predicted_index = int(
        np.argmax(predictions)
    )

    if predicted_index < len(class_names):

        predicted_class = (
            class_names[predicted_index]
        )

    else:

        predicted_class = (
            f"INDEX_{predicted_index}"
        )

    print()
    print(
        f"Predicted class : {predicted_class}"
    )

    print(
        f"Confidence      : "
        f"{float(predictions[predicted_index]):.10f}"
    )


# ============================================================
# Main Diagnostic
# ============================================================

def main():

    print("=" * 70)
    print("PAWCARE STAGE 7C")
    print("RAW MODEL PREDICTION DIAGNOSTIC")
    print("=" * 70)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Test image directory:")
    print(TEST_IMAGES_DIR)

    # --------------------------------------------------------
    # Check model files
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKING MODEL FILES")
    print("=" * 70)

    model_paths = [
        (
            "Cat Detector",
            CAT_DETECTOR_MODEL
        ),
        (
            "Cat Breed Classifier",
            CAT_BREED_MODEL
        ),
        (
            "Dog Detector",
            DOG_DETECTOR_MODEL
        ),
        (
            "Dog Breed Classifier",
            DOG_BREED_MODEL
        )
    ]

    for name, path in model_paths:

        check_file(
            path,
            name
        )

        print(
            f"{name}: OK"
        )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING MODELS")
    print("=" * 70)

    print("Loading Cat Detector...")
    cat_detector = tf.keras.models.load_model(
        CAT_DETECTOR_MODEL
    )

    print("Loading Cat Breed Classifier...")
    cat_breed = tf.keras.models.load_model(
        CAT_BREED_MODEL
    )

    print("Loading Dog Detector...")
    dog_detector = tf.keras.models.load_model(
        DOG_DETECTOR_MODEL
    )

    print("Loading Dog Breed Classifier...")
    dog_breed = tf.keras.models.load_model(
        DOG_BREED_MODEL
    )

    print()
    print("ALL FOUR MODELS LOADED SUCCESSFULLY")

    # --------------------------------------------------------
    # Print model input/output information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL INPUT / OUTPUT INFORMATION")
    print("=" * 70)

    models = [
        ("Cat Detector", cat_detector),
        ("Cat Breed Classifier", cat_breed),
        ("Dog Detector", dog_detector),
        ("Dog Breed Classifier", dog_breed)
    ]

    for name, model in models:

        print()
        print(name)

        print(
            f"Input shape : {model.input_shape}"
        )

        print(
            f"Output shape: {model.output_shape}"
        )

    # --------------------------------------------------------
    # Process every test image
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("IMAGE AND RAW PREDICTION DIAGNOSTIC")
    print("=" * 70)

    all_arrays = {}

    for test_name, image_path in TEST_IMAGES:

        print()
        print("=" * 70)
        print(test_name)
        print("=" * 70)

        check_file(
            image_path,
            test_name
        )

        print()
        print("File:")
        print(image_path)

        print()
        print(
            f"File size: "
            f"{image_path.stat().st_size} bytes"
        )

        print()
        print("SHA-256:")
        print(
            file_hash(image_path)
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image_array, batch = load_image(
            image_path
        )

        all_arrays[test_name] = image_array

        print()
        print("Processed image information")

        print(
            f"Shape : {image_array.shape}"
        )

        print(
            f"Dtype : {image_array.dtype}"
        )

        print(
            f"Min   : {float(image_array.min()):.6f}"
        )

        print(
            f"Max   : {float(image_array.max()):.6f}"
        )

        print(
            f"Mean  : {float(image_array.mean()):.6f}"
        )

        print(
            f"Std   : {float(image_array.std()):.6f}"
        )

        # ----------------------------------------------------
        # Cat detector
        # ----------------------------------------------------

        cat_detector_prediction = (
            cat_detector.predict(
                batch,
                verbose=0
            )[0]
        )

        print_prediction(
            "CAT DETECTOR RAW PREDICTION",
            cat_detector_prediction,
            CAT_DETECTOR_CLASSES
        )

        # ----------------------------------------------------
        # Cat breed
        # ----------------------------------------------------

        cat_breed_prediction = (
            cat_breed.predict(
                batch,
                verbose=0
            )[0]
        )

        print_prediction(
            "CAT BREED RAW PREDICTION",
            cat_breed_prediction,
            CAT_BREED_CLASSES
        )

        # ----------------------------------------------------
        # Dog detector
        # ----------------------------------------------------

        dog_detector_prediction = (
            dog_detector.predict(
                batch,
                verbose=0
            )[0]
        )

        print_prediction(
            "DOG DETECTOR RAW PREDICTION",
            dog_detector_prediction,
            DOG_DETECTOR_CLASSES
        )

        # ----------------------------------------------------
        # Dog breed
        # ----------------------------------------------------

        dog_breed_prediction = (
            dog_breed.predict(
                batch,
                verbose=0
            )[0]
        )

        print_prediction(
            "DOG BREED RAW PREDICTION",
            dog_breed_prediction,
            DOG_BREED_CLASSES
        )

    # --------------------------------------------------------
    # Compare processed images
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PROCESSED IMAGE COMPARISON")
    print("=" * 70)

    names = list(all_arrays.keys())

    for i in range(len(names)):

        for j in range(i + 1, len(names)):

            first_name = names[i]
            second_name = names[j]

            first_array = all_arrays[
                first_name
            ]

            second_array = all_arrays[
                second_name
            ]

            same = np.array_equal(
                first_array,
                second_array
            )

            mean_difference = float(
                np.mean(
                    np.abs(
                        first_array
                        - second_array
                    )
                )
            )

            print()
            print(
                f"{first_name} vs "
                f"{second_name}"
            )

            print(
                f"Exactly identical: {same}"
            )

            print(
                f"Mean pixel difference: "
                f"{mean_difference:.10f}"
            )

    print()
    print("=" * 70)
    print("STAGE 7C DIAGNOSTIC COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()