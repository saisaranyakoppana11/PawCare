# ============================================================
# PAWCARE UNIFIED PET PREDICTION SYSTEM
# ============================================================
#
# Pipeline:
#
#                    IMAGE
#                      |
#              CAT vs NON-CAT
#                  DETECTOR
#                      |
#             +--------+--------+
#             |                 |
#            CAT            NON-CAT
#             |                 |
#      CAT BREED MODEL      DOG vs NON-DOG
#             |               DETECTOR
#             |                  |
#             |          +-------+-------+
#             |          |               |
#             |         DOG           NON-DOG
#             |          |               |
#             |     DOG BREED MODEL   NEITHER
#             |          |
#             +----------+
#
# ============================================================

import os
import json
import numpy as np
import tensorflow as tf

from PIL import Image
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

# Current file:
#
# PawCare/
# └── scripts/
#     └── unified_pet_predict.py
#
# Going up:
# scripts -> PawCare
#
# Therefore:
# ".."

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


# ============================================================
# GENERAL SETTINGS
# ============================================================

IMG_SIZE = 224


# ============================================================
# CAT MODEL PATHS
# ============================================================

CAT_DETECTOR_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "cat_detector.keras"
)

CAT_DETECTOR_CLASSES_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "cat_detector_class_names.json"
)

CAT_BREED_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "cat_breed_classifier.keras"
)

CAT_BREED_CLASSES_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "class_names.json"
)


# ============================================================
# DOG MODEL PATHS
# ============================================================

DOG_DETECTOR_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "dog_breed",
    "dog_detector",
    "models",
    "dog_detector.keras"
)

DOG_DETECTOR_CLASSES_PATH = os.path.join(
    PROJECT_ROOT,
    "dog_breed",
    "dog_detector",
    "models",
    "dog_detector_class_names.json"
)

DOG_BREED_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "dog_breed",
    "models",
    "dog_breed_classifier.keras"
)

DOG_BREED_CLASSES_PATH = os.path.join(
    PROJECT_ROOT,
    "dog_breed",
    "models",
    "dog_class_names.json"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 75)
print("PAWCARE UNIFIED PET AI SYSTEM")
print("=" * 75)

print()
print("Project root:")
print(PROJECT_ROOT)


# ============================================================
# CHECK ALL REQUIRED FILES
# ============================================================

required_files = {

    "Cat Detector Model":
        CAT_DETECTOR_MODEL_PATH,

    "Cat Detector Classes":
        CAT_DETECTOR_CLASSES_PATH,

    "Cat Breed Model":
        CAT_BREED_MODEL_PATH,

    "Cat Breed Classes":
        CAT_BREED_CLASSES_PATH,

    "Dog Detector Model":
        DOG_DETECTOR_MODEL_PATH,

    "Dog Detector Classes":
        DOG_DETECTOR_CLASSES_PATH,

    "Dog Breed Model":
        DOG_BREED_MODEL_PATH,

    "Dog Breed Classes":
        DOG_BREED_CLASSES_PATH
}


print()
print("=" * 75)
print("CHECKING REQUIRED MODEL FILES")
print("=" * 75)

all_files_found = True

for name, path in required_files.items():

    if os.path.exists(path):

        print(f"[FOUND] {name}")
        print(f"        {path}")

    else:

        print(f"[MISSING] {name}")
        print(f"          {path}")

        all_files_found = False


if not all_files_found:

    print()
    print("=" * 75)
    print("ERROR: SOME REQUIRED FILES ARE MISSING")
    print("=" * 75)

    raise FileNotFoundError(
        "One or more required model files were not found."
    )


print()
print("All required files found successfully!")


# ============================================================
# LOAD CAT DETECTOR
# ============================================================

print()
print("=" * 75)
print("LOADING CAT VS NON-CAT DETECTOR")
print("=" * 75)

cat_detector_model = tf.keras.models.load_model(
    CAT_DETECTOR_MODEL_PATH
)

with open(
    CAT_DETECTOR_CLASSES_PATH,
    "r"
) as file:

    cat_detector_classes = json.load(file)


print()
print("Cat detector loaded successfully!")
print("Classes:", cat_detector_classes)


# ============================================================
# LOAD CAT BREED CLASSIFIER
# ============================================================

print()
print("=" * 75)
print("LOADING CAT BREED CLASSIFIER")
print("=" * 75)

cat_breed_model = tf.keras.models.load_model(
    CAT_BREED_MODEL_PATH
)

with open(
    CAT_BREED_CLASSES_PATH,
    "r"
) as file:

    cat_breed_classes = json.load(file)


print()
print("Cat breed classifier loaded successfully!")
print("Number of breeds:", len(cat_breed_classes))
print("Breeds:")

for breed in cat_breed_classes:

    print(
        " -",
        breed.replace("_", " ")
    )


# ============================================================
# LOAD DOG DETECTOR
# ============================================================

print()
print("=" * 75)
print("LOADING DOG VS NON-DOG DETECTOR")
print("=" * 75)

dog_detector_model = tf.keras.models.load_model(
    DOG_DETECTOR_MODEL_PATH
)

with open(
    DOG_DETECTOR_CLASSES_PATH,
    "r"
) as file:

    dog_detector_classes = json.load(file)


print()
print("Dog detector loaded successfully!")
print("Classes:", dog_detector_classes)


# ============================================================
# LOAD DOG BREED CLASSIFIER
# ============================================================

print()
print("=" * 75)
print("LOADING DOG BREED CLASSIFIER")
print("=" * 75)

dog_breed_model = tf.keras.models.load_model(
    DOG_BREED_MODEL_PATH
)

with open(
    DOG_BREED_CLASSES_PATH,
    "r"
) as file:

    dog_breed_classes = json.load(file)


print()
print("Dog breed classifier loaded successfully!")
print("Number of breeds:", len(dog_breed_classes))
print("Breeds:")

for breed in dog_breed_classes:

    print(
        " -",
        breed.replace("_", " ")
    )


# ============================================================
# OPEN IMAGE SELECTOR
# ============================================================

print()
print("=" * 75)
print("SELECT AN IMAGE")
print("=" * 75)

root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(

    title="Select a Pet Image",

    filetypes=[
        (
            "Image Files",
            "*.jpg *.jpeg *.png *.bmp *.webp"
        )
    ]
)


# ============================================================
# NO IMAGE SELECTED
# ============================================================

if not image_path:

    print()
    print("No image selected.")
    print("Prediction cancelled.")

    root.destroy()
    raise SystemExit


print()
print("Selected image:")
print(image_path)


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    original_image = Image.open(
        image_path
    ).convert("RGB")

except Exception as error:

    root.destroy()

    raise RuntimeError(
        f"Unable to open image: {error}"
    )


# ============================================================
# DISPLAY ORIGINAL IMAGE
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.imshow(
    original_image
)

plt.axis("off")

plt.title(
    "Selected Image",
    fontsize=14
)

plt.show()


# ============================================================
# PREPROCESS IMAGE
# ============================================================

image = original_image.resize(
    (
        IMG_SIZE,
        IMG_SIZE
    )
)

image_array = np.array(
    image,
    dtype=np.float32
)

image_array = np.expand_dims(
    image_array,
    axis=0
)


# ============================================================
# STAGE 1
# CAT VS NON-CAT
# ============================================================

print()
print("=" * 75)
print("STAGE 1 - CAT VS NON-CAT DETECTION")
print("=" * 75)

cat_detector_predictions = (
    cat_detector_model.predict(
        image_array,
        verbose=0
    )[0]
)


cat_detector_index = np.argmax(
    cat_detector_predictions
)


cat_detected_class = (
    cat_detector_classes[
        cat_detector_index
    ]
)


cat_detector_confidence = (
    cat_detector_predictions[
        cat_detector_index
    ] * 100
)


print()
print(
    "Cat detector result:",
    cat_detected_class
)

print(
    f"Confidence: "
    f"{cat_detector_confidence:.2f}%"
)


print()
print("Cat detector probabilities:")

for index, class_name in enumerate(
    cat_detector_classes
):

    probability = (
        cat_detector_predictions[
            index
        ] * 100
    )

    print(
        f"{class_name:10s} : "
        f"{probability:.2f}%"
    )


# ============================================================
# IF CAT
# ============================================================

if cat_detected_class == "cat":

    print()
    print("=" * 75)
    print("CAT DETECTED")
    print("=" * 75)

    print()
    print("Running Cat Breed Classifier...")


    # --------------------------------------------------------
    # CAT BREED PREDICTION
    # --------------------------------------------------------

    cat_breed_predictions = (
        cat_breed_model.predict(
            image_array,
            verbose=0
        )[0]
    )


    # --------------------------------------------------------
    # TOP 3 CAT BREEDS
    # --------------------------------------------------------

    cat_top_indices = np.argsort(
        cat_breed_predictions
    )[::-1][:3]


    print()
    print("Top 3 Cat Breed Predictions:")
    print()


    for rank, index in enumerate(
        cat_top_indices,
        start=1
    ):

        breed = (
            cat_breed_classes[
                index
            ]
        )

        confidence = (
            cat_breed_predictions[
                index
            ] * 100
        )

        print(
            f"{rank}. "
            f"{breed.replace('_', ' ')} "
            f"- {confidence:.2f}%"
        )


    # --------------------------------------------------------
    # BEST CAT BREED
    # --------------------------------------------------------

    best_cat_index = (
        cat_top_indices[0]
    )

    best_cat_breed = (
        cat_breed_classes[
            best_cat_index
        ]
    )

    best_cat_confidence = (
        cat_breed_predictions[
            best_cat_index
        ] * 100
    )


    # --------------------------------------------------------
    # FINAL CAT RESULT
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("FINAL PAWCARE RESULT")
    print("=" * 75)

    print()
    print("Animal           : CAT")

    print(
        f"Cat Confidence   : "
        f"{cat_detector_confidence:.2f}%"
    )

    print(
        f"Predicted Breed  : "
        f"{best_cat_breed.replace('_', ' ')}"
    )

    print(
        f"Breed Confidence : "
        f"{best_cat_confidence:.2f}%"
    )


    # --------------------------------------------------------
    # DISPLAY FINAL CAT RESULT WITH IMAGE
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 7)
    )

    plt.imshow(
        original_image
    )

    plt.axis("off")

    plt.title(
        "🐱 CAT DETECTED\n"
        f"Breed: "
        f"{best_cat_breed.replace('_', ' ')}\n"
        f"Breed Confidence: "
        f"{best_cat_confidence:.2f}%",
        fontsize=14
    )

    plt.show()


# ============================================================
# IF NOT CAT → CHECK DOG
# ============================================================

else:

    print()
    print("=" * 75)
    print("IMAGE IS NOT DETECTED AS A CAT")
    print("=" * 75)

    print()
    print("Moving to Dog vs Non-Dog Detector...")


    # ========================================================
    # STAGE 2
    # DOG VS NON-DOG
    # ========================================================

    print()
    print("=" * 75)
    print("STAGE 2 - DOG VS NON-DOG DETECTION")
    print("=" * 75)


    dog_detector_predictions = (
        dog_detector_model.predict(
            image_array,
            verbose=0
        )[0]
    )


    dog_detector_index = np.argmax(
        dog_detector_predictions
    )


    dog_detected_class = (
        dog_detector_classes[
            dog_detector_index
        ]
    )


    dog_detector_confidence = (
        dog_detector_predictions[
            dog_detector_index
        ] * 100
    )


    print()
    print(
        "Dog detector result:",
        dog_detected_class
    )

    print(
        f"Confidence: "
        f"{dog_detector_confidence:.2f}%"
    )


    print()
    print("Dog detector probabilities:")

    for index, class_name in enumerate(
        dog_detector_classes
    ):

        probability = (
            dog_detector_predictions[
                index
            ] * 100
        )

        print(
            f"{class_name:10s} : "
            f"{probability:.2f}%"
        )


    # ========================================================
    # IF DOG
    # ========================================================

    if dog_detected_class == "dog":

        print()
        print("=" * 75)
        print("DOG DETECTED")
        print("=" * 75)

        print()
        print("Running Dog Breed Classifier...")


        # ----------------------------------------------------
        # DOG BREED PREDICTION
        # ----------------------------------------------------

        dog_breed_predictions = (
            dog_breed_model.predict(
                image_array,
                verbose=0
            )[0]
        )


        # ----------------------------------------------------
        # TOP 3 DOG BREEDS
        # ----------------------------------------------------

        dog_top_indices = np.argsort(
            dog_breed_predictions
        )[::-1][:3]


        print()
        print("Top 3 Dog Breed Predictions:")
        print()


        for rank, index in enumerate(
            dog_top_indices,
            start=1
        ):

            breed = (
                dog_breed_classes[
                    index
                ]
            )

            confidence = (
                dog_breed_predictions[
                    index
                ] * 100
            )

            print(
                f"{rank}. "
                f"{breed.replace('_', ' ')} "
                f"- {confidence:.2f}%"
            )


        # ----------------------------------------------------
        # BEST DOG BREED
        # ----------------------------------------------------

        best_dog_index = (
            dog_top_indices[0]
        )

        best_dog_breed = (
            dog_breed_classes[
                best_dog_index
            ]
        )

        best_dog_confidence = (
            dog_breed_predictions[
                best_dog_index
            ] * 100
        )


        # ----------------------------------------------------
        # FINAL DOG RESULT
        # ----------------------------------------------------

        print()
        print("=" * 75)
        print("FINAL PAWCARE RESULT")
        print("=" * 75)

        print()
        print("Animal           : DOG")

        print(
            f"Dog Confidence   : "
            f"{dog_detector_confidence:.2f}%"
        )

        print(
            f"Predicted Breed  : "
            f"{best_dog_breed.replace('_', ' ')}"
        )

        print(
            f"Breed Confidence : "
            f"{best_dog_confidence:.2f}%"
        )


        # ----------------------------------------------------
        # DISPLAY FINAL DOG RESULT WITH IMAGE
        # ----------------------------------------------------

        plt.figure(
            figsize=(7, 7)
        )

        plt.imshow(
            original_image
        )

        plt.axis("off")

        plt.title(
            "🐶 DOG DETECTED\n"
            f"Breed: "
            f"{best_dog_breed.replace('_', ' ')}\n"
            f"Breed Confidence: "
            f"{best_dog_confidence:.2f}%",
            fontsize=14
        )

        plt.show()


    # ========================================================
    # NEITHER CAT NOR DOG
    # ========================================================

    else:

        print()
        print("=" * 75)
        print("FINAL PAWCARE RESULT")
        print("=" * 75)

        print()
        print("❌ Image is neither a cat nor a dog.")

        print()
        print(
            "Cat detector confidence: "
            f"{cat_detector_confidence:.2f}%"
        )

        print(
            "Dog detector confidence: "
            f"{dog_detector_confidence:.2f}%"
        )


        # ----------------------------------------------------
        # DISPLAY NON-PET RESULT
        # ----------------------------------------------------

        plt.figure(
            figsize=(7, 7)
        )

        plt.imshow(
            original_image
        )

        plt.axis("off")

        plt.title(
            "❌ NEITHER CAT NOR DOG\n"
            "Breed classification skipped.",
            fontsize=14
        )

        plt.show()


# ============================================================
# CLEANUP
# ============================================================

root.destroy()


print()
print("=" * 75)
print("PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 75)

print()
print("PawCare Unified Pet AI finished.")