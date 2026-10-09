import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox


# ============================================================
# PROJECT PATHS
# ============================================================

# Current file:
# PawCare/
# └── dog_breed/
#     └── scripts/
#         └── pawcare_dog_predict.py
#
# Going up two levels:
# scripts -> dog_breed -> PawCare

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


# ============================================================
# DOG DETECTOR PATHS
# ============================================================

DETECTOR_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "dog_breed",
    "dog_detector",
    "models",
    "dog_detector.keras"
)

DETECTOR_CLASSES_PATH = os.path.join(
    PROJECT_ROOT,
    "dog_breed",
    "dog_detector",
    "models",
    "dog_detector_class_names.json"
)


# ============================================================
# DOG BREED CLASSIFIER PATHS
# ============================================================

BREED_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "dog_breed",
    "models",
    "dog_breed_classifier.keras"
)

BREED_CLASSES_PATH = os.path.join(
    PROJECT_ROOT,
    "dog_breed",
    "models",
    "dog_class_names.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 224


# ============================================================
# START
# ============================================================

print("=" * 70)
print("PAWCARE DOG AI SYSTEM")
print("=" * 70)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    DETECTOR_MODEL_PATH,
    DETECTOR_CLASSES_PATH,
    BREED_MODEL_PATH,
    BREED_CLASSES_PATH
]

print("\nChecking required files...")

for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )

    print("FOUND:")
    print(file_path)


# ============================================================
# LOAD DOG VS NON-DOG DETECTOR
# ============================================================

print("\n" + "=" * 70)
print("LOADING DOG VS NON-DOG DETECTOR")
print("=" * 70)

detector_model = tf.keras.models.load_model(
    DETECTOR_MODEL_PATH
)

with open(
    DETECTOR_CLASSES_PATH,
    "r"
) as f:

    detector_classes = json.load(f)


print("\nDog detector loaded successfully!")

print(
    "Detector classes:",
    detector_classes
)


# ============================================================
# LOAD DOG BREED CLASSIFIER
# ============================================================

print("\n" + "=" * 70)
print("LOADING DOG BREED CLASSIFIER")
print("=" * 70)

breed_model = tf.keras.models.load_model(
    BREED_MODEL_PATH
)

with open(
    BREED_CLASSES_PATH,
    "r"
) as f:

    breed_classes = json.load(f)


print("\nDog breed classifier loaded successfully!")

print(
    "Breed classes:",
    breed_classes
)


# ============================================================
# SELECT IMAGE
# ============================================================

print("\nOpening image selector...")

root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[
        (
            "Image Files",
            "*.jpg *.jpeg *.png *.bmp *.webp"
        )
    ]
)


# ============================================================
# CHECK IMAGE SELECTION
# ============================================================

if not image_path:

    print("\nNo image selected.")

    root.destroy()

    exit()


print("\nSelected image:")
print(image_path)


# ============================================================
# LOAD IMAGE
# ============================================================

original_image = Image.open(
    image_path
).convert("RGB")


# ============================================================
# PREPARE IMAGE
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
# DOG VS NON-DOG DETECTION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1 - DOG VS NON-DOG DETECTION")
print("=" * 70)


detector_predictions = detector_model.predict(
    image_array,
    verbose=0
)[0]


detector_index = np.argmax(
    detector_predictions
)

detected_class = detector_classes[
    detector_index
]

detector_confidence = (
    detector_predictions[
        detector_index
    ] * 100
)


print(
    "\nDetection:",
    detected_class
)

print(
    f"Confidence: "
    f"{detector_confidence:.2f}%"
)


# ============================================================
# SHOW DETECTOR PROBABILITIES
# ============================================================

print("\nDetector probabilities:")

for index, class_name in enumerate(
    detector_classes
):

    probability = (
        detector_predictions[index]
        * 100
    )

    print(
        f"{class_name:10s} : "
        f"{probability:.2f}%"
    )


# ============================================================
# IF NON-DOG → STOP
# ============================================================

if detected_class != "dog":

    print("\n" + "-" * 70)
    print("RESULT: NON-DOG IMAGE")
    print("-" * 70)

    print(
        "Dog breed classification skipped."
    )


    result_message = (
        "❌ NON-DOG IMAGE\n\n"
        f"Detector Confidence: "
        f"{detector_confidence:.2f}%\n\n"
        "Dog breed classification "
        "was skipped."
    )


    messagebox.showinfo(
        "PawCare - Dog AI",
        result_message
    )


    root.destroy()

    exit()


# ============================================================
# STAGE 2
# DOG BREED CLASSIFICATION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 - DOG BREED CLASSIFICATION")
print("=" * 70)


breed_predictions = breed_model.predict(
    image_array,
    verbose=0
)[0]


# ============================================================
# GET TOP 3 BREEDS
# ============================================================

top_indices = np.argsort(
    breed_predictions
)[::-1][:3]


# ============================================================
# DISPLAY TOP 3
# ============================================================

print("\nTop 3 Dog Breed Predictions:")

for rank, index in enumerate(
    top_indices,
    start=1
):

    breed = breed_classes[
        index
    ]

    confidence = (
        breed_predictions[index]
        * 100
    )

    print(
        f"{rank}. "
        f"{breed.replace('_', ' ')} "
        f"- {confidence:.2f}%"
    )


# ============================================================
# BEST BREED
# ============================================================

best_index = top_indices[0]

best_breed = breed_classes[
    best_index
]

best_confidence = (
    breed_predictions[
        best_index
    ] * 100
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL PAWCARE DOG AI RESULT")
print("=" * 70)

print(
    "\nAnimal          : Dog"
)

print(
    f"Dog Confidence  : "
    f"{detector_confidence:.2f}%"
)

print(
    f"Predicted Breed : "
    f"{best_breed.replace('_', ' ')}"
)

print(
    f"Breed Confidence: "
    f"{best_confidence:.2f}%"
)


# ============================================================
# CREATE RESULT MESSAGE
# ============================================================

result_message = (
    "🐶 DOG DETECTED\n\n"
    f"Dog Confidence: "
    f"{detector_confidence:.2f}%\n\n"
    "Predicted Breed:\n"
    f"{best_breed.replace('_', ' ')}\n\n"
    f"Breed Confidence: "
    f"{best_confidence:.2f}%\n\n"
    "Top 3 Predictions:\n"
)


for rank, index in enumerate(
    top_indices,
    start=1
):

    breed = breed_classes[
        index
    ].replace(
        "_",
        " "
    )

    confidence = (
        breed_predictions[index]
        * 100
    )

    result_message += (
        f"{rank}. "
        f"{breed} - "
        f"{confidence:.2f}%\n"
    )


# ============================================================
# SHOW RESULT
# ============================================================

messagebox.showinfo(
    "PawCare - Dog AI Result",
    result_message
)


# ============================================================
# CLOSE TKINTER
# ============================================================

root.destroy()

print("\nPrediction completed successfully!")