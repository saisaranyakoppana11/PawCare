import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox

# ============================================================
# PATHS
# ============================================================

DETECTOR_MODEL_PATH = "models/cat_detector.keras"
DETECTOR_CLASSES_PATH = "models/cat_detector_class_names.json"

BREED_MODEL_PATH = "models/cat_breed_classifier.keras"
BREED_CLASSES_PATH = "models/class_names.json"

IMG_SIZE = 224


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    DETECTOR_MODEL_PATH,
    DETECTOR_CLASSES_PATH,
    BREED_MODEL_PATH,
    BREED_CLASSES_PATH
]

for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(DETECTOR_CLASSES_PATH, "r") as file:

    detector_classes = json.load(file)


with open(BREED_CLASSES_PATH, "r") as file:

    breed_classes = json.load(file)


# ============================================================
# LOAD MODELS
# ============================================================

print("\n" + "=" * 60)
print("PAWCARE AI PET IMAGE ANALYSIS")
print("=" * 60)

print("\nLoading Cat vs Non-Cat detector...")

detector_model = tf.keras.models.load_model(
    DETECTOR_MODEL_PATH
)

print("Cat detector loaded successfully!")


print("\nLoading Cat Breed classifier...")

breed_model = tf.keras.models.load_model(
    BREED_MODEL_PATH
)

print("Cat breed classifier loaded successfully!")


# ============================================================
# SELECT IMAGE
# ============================================================

root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select Pet Image",
    filetypes=[
        (
            "Image files",
            "*.jpg *.jpeg *.png *.bmp *.webp"
        ),
        (
            "JPG files",
            "*.jpg *.jpeg"
        ),
        (
            "PNG files",
            "*.png"
        ),
        (
            "All files",
            "*.*"
        )
    ]
)


if not image_path:

    print("\nNo image selected.")
    print("Prediction cancelled.")

    root.destroy()
    exit()


print("\nSelected image:")
print(image_path)


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(image_path)

print("\nOriginal image size:")
print(image.size)

image = image.convert("RGB")

image = image.resize(
    (IMG_SIZE, IMG_SIZE)
)


# ============================================================
# PREPARE IMAGE
# ============================================================

image_array = np.array(
    image,
    dtype=np.float32
)

image_array = np.expand_dims(
    image_array,
    axis=0
)


# ============================================================
# STAGE 1 — CAT VS NON-CAT
# ============================================================

print("\n" + "-" * 60)
print("STAGE 1: CAT VS NON-CAT DETECTION")
print("-" * 60)

detector_predictions = detector_model.predict(
    image_array,
    verbose=0
)

detector_index = np.argmax(
    detector_predictions[0]
)

detector_result = detector_classes[
    detector_index
]

detector_confidence = (
    float(
        detector_predictions[0][detector_index]
    ) * 100
)

print("\nDetector prediction:")
print(detector_result)

print("\nDetector confidence:")
print(
    f"{detector_confidence:.2f}%"
)


# ============================================================
# IF NON-CAT → STOP
# ============================================================

if detector_result == "non_cat":

    print("\n" + "=" * 60)
    print("PAWCARE RESULT")
    print("=" * 60)

    print("\n❌ This image is not detected as a cat.")

    print(
        "\nPlease upload a clear image of a cat "
        "to identify its breed."
    )

    message = (
        "This image is not detected as a cat.\n\n"
        f"Detector confidence: "
        f"{detector_confidence:.2f}%\n\n"
        "Please upload a clear cat image."
    )

    messagebox.showinfo(
        "PawCare - Cat Detection",
        message
    )

    root.destroy()
    exit()


# ============================================================
# STAGE 2 — CAT BREED CLASSIFICATION
# ============================================================

print("\n" + "-" * 60)
print("STAGE 2: CAT BREED CLASSIFICATION")
print("-" * 60)

breed_predictions = breed_model.predict(
    image_array,
    verbose=0
)

top_indices = np.argsort(
    breed_predictions[0]
)[::-1][:3]

predicted_index = top_indices[0]

predicted_breed = breed_classes[
    predicted_index
]

breed_confidence = (
    float(
        breed_predictions[0][predicted_index]
    ) * 100
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("PAWCARE FINAL RESULT")
print("=" * 60)

print("\nAnimal:")
print("CAT")

print("\nPredicted Breed:")
print(predicted_breed)

print("\nBreed Confidence:")
print(
    f"{breed_confidence:.2f}%"
)


print("\nTop 3 Breed Predictions:")
print("-" * 40)

for rank, index in enumerate(
    top_indices,
    start=1
):

    breed = breed_classes[index]

    confidence = (
        float(
            breed_predictions[0][index]
        ) * 100
    )

    print(
        f"{rank}. "
        f"{breed:20s} "
        f"{confidence:.2f}%"
    )


# ============================================================
# POPUP
# ============================================================

message = (
    "🐱 CAT DETECTED\n\n"

    f"Predicted Breed:\n"
    f"{predicted_breed}\n\n"

    f"Breed Confidence:\n"
    f"{breed_confidence:.2f}%\n\n"

    "Top 3 Predictions:\n"

    f"1. {breed_classes[top_indices[0]]} - "
    f"{breed_predictions[0][top_indices[0]] * 100:.2f}%\n"

    f"2. {breed_classes[top_indices[1]]} - "
    f"{breed_predictions[0][top_indices[1]] * 100:.2f}%\n"

    f"3. {breed_classes[top_indices[2]]} - "
    f"{breed_predictions[0][top_indices[2]] * 100:.2f}%"
)

messagebox.showinfo(
    "PawCare - Cat Breed Prediction",
    message
)

root.destroy()