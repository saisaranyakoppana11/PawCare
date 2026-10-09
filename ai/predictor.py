"""
PawCare Unified Pet AI Predictor

Purpose:
    1. Detect whether an uploaded image contains a cat.
    2. If it is not a cat, detect whether it contains a dog.
    3. If it is a cat, classify the cat breed.
    4. If it is a dog, classify the dog breed.
    5. Reject images that are neither cats nor dogs.

Models:
    Cat Detector
    Cat Breed Classifier
    Dog Detector
    Dog Breed Classifier

This module is designed to be used later by the Flask application.
"""

from pathlib import Path
import json

import numpy as np
import tensorflow as tf


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ML_TRAINING_DIR = PROJECT_ROOT / "ml_training"


# ============================================================
# Model Paths
# ============================================================

CAT_DETECTOR_MODEL_PATH = (
    ML_TRAINING_DIR
    / "cat_detector"
    / "trained_model"
    / "cat_detector.keras"
)

CAT_DETECTOR_CLASSES_PATH = (
    ML_TRAINING_DIR
    / "cat_detector"
    / "trained_model"
    / "class_names.json"
)


CAT_BREED_MODEL_PATH = (
    ML_TRAINING_DIR
    / "cat_breed"
    / "trained_model"
    / "cat_breed_classifier.keras"
)

CAT_BREED_CLASSES_PATH = (
    ML_TRAINING_DIR
    / "cat_breed"
    / "trained_model"
    / "class_names.json"
)


DOG_DETECTOR_MODEL_PATH = (
    ML_TRAINING_DIR
    / "dog_detector"
    / "trained_model"
    / "dog_detector.keras"
)

DOG_DETECTOR_CLASSES_PATH = (
    ML_TRAINING_DIR
    / "dog_detector"
    / "trained_model"
    / "class_names.json"
)


DOG_BREED_MODEL_PATH = (
    ML_TRAINING_DIR
    / "dog_breed"
    / "trained_model"
    / "dog_breed_classifier.keras"
)

DOG_BREED_CLASSES_PATH = (
    ML_TRAINING_DIR
    / "dog_breed"
    / "trained_model"
    / "class_names.json"
)


# ============================================================
# Image Configuration
# ============================================================

IMAGE_SIZE = (224, 224)


# ============================================================
# Model Loading Helpers
# ============================================================

def check_file_exists(file_path: Path, description: str):
    """
    Verify that a required model/class file exists.
    """

    if not file_path.exists():

        raise FileNotFoundError(
            f"{description} not found:\n"
            f"{file_path}"
        )


def load_class_names(class_names_path: Path):
    """
    Load class names from a JSON file.
    """

    check_file_exists(
        class_names_path,
        "Class names file"
    )

    with open(
        class_names_path,
        "r",
        encoding="utf-8"
    ) as file:

        class_names = json.load(file)

    if not isinstance(class_names, list):

        raise ValueError(
            f"Invalid class names format in:\n"
            f"{class_names_path}"
        )

    if not class_names:

        raise ValueError(
            f"Class names file is empty:\n"
            f"{class_names_path}"
        )

    return class_names


def load_model(model_path: Path, model_name: str):
    """
    Load a trained Keras model.
    """

    check_file_exists(
        model_path,
        f"{model_name} model"
    )

    print(
        f"Loading {model_name}..."
    )

    model = tf.keras.models.load_model(
        model_path
    )

    print(
        f"{model_name} loaded successfully."
    )

    return model


# ============================================================
# Load All PawCare Models
# ============================================================

print("=" * 70)
print("PAWCARE UNIFIED AI INITIALIZATION")
print("=" * 70)


cat_detector_model = load_model(
    CAT_DETECTOR_MODEL_PATH,
    "Cat Detector"
)

cat_detector_class_names = load_class_names(
    CAT_DETECTOR_CLASSES_PATH
)


cat_breed_model = load_model(
    CAT_BREED_MODEL_PATH,
    "Cat Breed Classifier"
)

cat_breed_class_names = load_class_names(
    CAT_BREED_CLASSES_PATH
)


dog_detector_model = load_model(
    DOG_DETECTOR_MODEL_PATH,
    "Dog Detector"
)

dog_detector_class_names = load_class_names(
    DOG_DETECTOR_CLASSES_PATH
)


dog_breed_model = load_model(
    DOG_BREED_MODEL_PATH,
    "Dog Breed Classifier"
)

dog_breed_class_names = load_class_names(
    DOG_BREED_CLASSES_PATH
)


print()
print("Loaded Cat Detector classes:")
print(cat_detector_class_names)

print()
print("Loaded Cat Breed classes:")
print(cat_breed_class_names)

print()
print("Loaded Dog Detector classes:")
print(dog_detector_class_names)

print()
print("Loaded Dog Breed classes:")
print(dog_breed_class_names)

print("=" * 70)
print("ALL PAWCARE AI MODELS LOADED")
print("=" * 70)


# ============================================================
# Image Preprocessing
# ============================================================

def preprocess_image(image_path):
    """
    Load and preprocess an image for PawCare models.

    EfficientNetB0 in this project receives images in the
    normal 0-255 pixel range because its built-in preprocessing
    is used.
    """

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n"
            f"{image_path}"
        )

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = image_array.astype(
        np.float32
    )

    return image_array


# ============================================================
# Generic Prediction Helper
# ============================================================

def predict_class(
    model,
    class_names,
    image_array
):
    """
    Predict the highest-probability class.

    Returns:
        {
            "class_name": str,
            "confidence": float,
            "probabilities": dict
        }
    """

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    probabilities = {}

    for index, class_name in enumerate(
        class_names
    ):

        probabilities[class_name] = float(
            predictions[index]
        )

    return {
        "class_name": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities
    }


# ============================================================
# Detector Helper
# ============================================================

def detect_species(
    model,
    class_names,
    image_array,
    positive_class
):
    """
    Detect whether the image belongs to the
    requested species.

    Example:
        positive_class = "cat"

    Returns:
        {
            "detected": True/False,
            "confidence": float,
            "predicted_class": str
        }
    """

    result = predict_class(
        model,
        class_names,
        image_array
    )

    predicted_class = result[
        "class_name"
    ]

    return {
        "detected": (
            predicted_class == positive_class
        ),
        "confidence": result[
            "confidence"
        ],
        "predicted_class": predicted_class,
        "probabilities": result[
            "probabilities"
        ]
    }


# ============================================================
# Cat Prediction
# ============================================================

def predict_cat(image_array):
    """
    Run cat detection followed by cat breed classification.
    """

    detection = detect_species(
        cat_detector_model,
        cat_detector_class_names,
        image_array,
        "cat"
    )

    if not detection["detected"]:

        return {
            "species": "cat",
            "is_pet": False,
            "detected": False,
            "message": "The image was not detected as a cat.",
            "detector_confidence": detection[
                "confidence"
            ],
            "breed": None,
            "breed_confidence": None,
            "top_predictions": []
        }

    breed_result = predict_class(
        cat_breed_model,
        cat_breed_class_names,
        image_array
    )

    sorted_predictions = sorted(
        breed_result["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True
    )

    top_predictions = []

    for breed, confidence in sorted_predictions[:3]:

        top_predictions.append(
            {
                "breed": breed,
                "confidence": float(
                    confidence
                )
            }
        )

    return {
        "species": "cat",
        "is_pet": True,
        "detected": True,
        "message": "Cat detected successfully.",
        "detector_confidence": detection[
            "confidence"
        ],
        "breed": breed_result[
            "class_name"
        ],
        "breed_confidence": breed_result[
            "confidence"
        ],
        "top_predictions": top_predictions
    }


# ============================================================
# Dog Prediction
# ============================================================

def predict_dog(image_array):
    """
    Run dog detection followed by dog breed classification.
    """

    detection = detect_species(
        dog_detector_model,
        dog_detector_class_names,
        image_array,
        "dog"
    )

    if not detection["detected"]:

        return {
            "species": "dog",
            "is_pet": False,
            "detected": False,
            "message": "The image was not detected as a dog.",
            "detector_confidence": detection[
                "confidence"
            ],
            "breed": None,
            "breed_confidence": None,
            "top_predictions": []
        }

    breed_result = predict_class(
        dog_breed_model,
        dog_breed_class_names,
        image_array
    )

    sorted_predictions = sorted(
        breed_result["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True
    )

    top_predictions = []

    for breed, confidence in sorted_predictions[:3]:

        top_predictions.append(
            {
                "breed": breed,
                "confidence": float(
                    confidence
                )
            }
        )

    return {
        "species": "dog",
        "is_pet": True,
        "detected": True,
        "message": "Dog detected successfully.",
        "detector_confidence": detection[
            "confidence"
        ],
        "breed": breed_result[
            "class_name"
        ],
        "breed_confidence": breed_result[
            "confidence"
        ],
        "top_predictions": top_predictions
    }


# ============================================================
# Unified Prediction
# ============================================================

def predict_pet(image_path):
    """
    Complete PawCare unified prediction pipeline.

    Flow:

        Image
          |
          v
      Cat Detector
          |
       Cat? -------- Yes --------> Cat Breed
          |
          No
          |
          v
      Dog Detector
          |
       Dog? -------- Yes --------> Dog Breed
          |
          No
          |
          v
       Reject
    """

    image_array = preprocess_image(
        image_path
    )

    # --------------------------------------------------------
    # Step 1: Cat Detection
    # --------------------------------------------------------

    cat_detection = detect_species(
        cat_detector_model,
        cat_detector_class_names,
        image_array,
        "cat"
    )

    if cat_detection["detected"]:

        cat_result = predict_cat(
            image_array
        )

        cat_result["image_path"] = str(
            image_path
        )

        return cat_result

    # --------------------------------------------------------
    # Step 2: Dog Detection
    # --------------------------------------------------------

    dog_detection = detect_species(
        dog_detector_model,
        dog_detector_class_names,
        image_array,
        "dog"
    )

    if dog_detection["detected"]:

        dog_result = predict_dog(
            image_array
        )

        dog_result["image_path"] = str(
            image_path
        )

        return dog_result

    # --------------------------------------------------------
    # Step 3: Neither Cat nor Dog
    # --------------------------------------------------------

    return {
        "species": "unknown",
        "is_pet": False,
        "detected": False,
        "message": (
            "The image was not detected as a cat or dog."
        ),
        "detector_confidence": None,
        "breed": None,
        "breed_confidence": None,
        "top_predictions": [],
        "image_path": str(image_path)
    }


# ============================================================
# Direct Test
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("PAWCARE UNIFIED AI PREDICTOR TEST")
    print("=" * 70)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Cat Detector:")
    print(CAT_DETECTOR_MODEL_PATH)

    print()
    print("Cat Breed Classifier:")
    print(CAT_BREED_MODEL_PATH)

    print()
    print("Dog Detector:")
    print(DOG_DETECTOR_MODEL_PATH)

    print()
    print("Dog Breed Classifier:")
    print(DOG_BREED_MODEL_PATH)

    print()
    print("=" * 70)
    print("PREDICTOR INITIALIZATION TEST PASSED")
    print("=" * 70)