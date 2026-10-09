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

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".."
    )
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "dog_breed",
    "dog_detector",
    "models",
    "dog_detector.keras"
)

CLASS_NAMES_PATH = os.path.join(
    PROJECT_ROOT,
    "dog_breed",
    "dog_detector",
    "models",
    "dog_detector_class_names.json"
)

IMG_SIZE = 224


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Dog vs Non-Dog Detector...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

with open(
    CLASS_NAMES_PATH,
    "r"
) as f:

    class_names = json.load(f)


print("Model loaded successfully!")

print(
    "Classes:",
    class_names
)


# ============================================================
# SELECT IMAGE
# ============================================================

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


if not image_path:

    print("\nNo image selected.")

    root.destroy()

    exit()


print("\nSelected image:")
print(image_path)


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(
    image_path
).convert("RGB")


# ============================================================
# PREPARE IMAGE
# ============================================================

image = image.resize(
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
# PREDICTION
# ============================================================

predictions = model.predict(
    image_array,
    verbose=0
)[0]


predicted_index = np.argmax(
    predictions
)

predicted_class = class_names[
    predicted_index
]

confidence = (
    predictions[predicted_index]
    * 100
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 55)
print("DOG VS NON-DOG RESULT")
print("=" * 55)

print(
    f"Prediction : {predicted_class}"
)

print(
    f"Confidence : {confidence:.2f}%"
)


# ============================================================
# BOTH CLASS PROBABILITIES
# ============================================================

print("\nClass probabilities:")

for index, class_name in enumerate(
    class_names
):

    probability = (
        predictions[index]
        * 100
    )

    print(
        f"{class_name:10s} : "
        f"{probability:.2f}%"
    )


print("=" * 55)


# ============================================================
# USER-FRIENDLY MESSAGE
# ============================================================

if predicted_class == "dog":

    result_message = (
        "🐶 DOG DETECTED\n\n"
        f"Confidence: {confidence:.2f}%\n\n"
        "This image can now be sent "
        "to the Dog Breed Classifier."
    )

else:

    result_message = (
        "❌ NON-DOG DETECTED\n\n"
        f"Confidence: {confidence:.2f}%\n\n"
        "This image will not be sent "
        "to the Dog Breed Classifier."
    )


messagebox.showinfo(
    "PawCare - Dog Detector",
    result_message
)


root.destroy()