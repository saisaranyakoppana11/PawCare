import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox

MODEL_PATH = "models/cat_detector.keras"
CLASS_NAMES_PATH = "models/cat_detector_class_names.json"

IMG_SIZE = 224

# --------------------------------------------------
# CHECK FILES
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names not found: {CLASS_NAMES_PATH}"
    )

# --------------------------------------------------
# LOAD CLASS NAMES
# --------------------------------------------------

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

print("\nClass names:")
print(class_names)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("\nLoading Cat vs Non-Cat model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")

# --------------------------------------------------
# SELECT IMAGE
# --------------------------------------------------

root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[
        (
            "Image files",
            "*.jpg *.jpeg *.png *.bmp *.webp"
        ),
        ("All files", "*.*")
    ]
)

if not image_path:

    print("\nNo image selected.")
    root.destroy()
    exit()

print("\nSelected image:")
print(image_path)

# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

image = Image.open(image_path)

print("\nOriginal image size:")
print(image.size)

image = image.convert("RGB")

image = image.resize(
    (IMG_SIZE, IMG_SIZE)
)

# --------------------------------------------------
# PREPARE IMAGE
# --------------------------------------------------

image_array = np.array(
    image,
    dtype=np.float32
)

image_array = np.expand_dims(
    image_array,
    axis=0
)

# --------------------------------------------------
# PREDICT
# --------------------------------------------------

print("\nAnalyzing image...")

predictions = model.predict(
    image_array,
    verbose=0
)

predicted_index = np.argmax(
    predictions[0]
)

predicted_class = class_names[
    predicted_index
]

confidence = (
    float(predictions[0][predicted_index])
    * 100
)

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("PAWCARE CAT DETECTOR")
print("=" * 60)

print("\nPrediction:")
print(predicted_class)

print("\nConfidence:")
print(f"{confidence:.2f}%")

print("\nClass probabilities:")

for index, class_name in enumerate(class_names):

    probability = (
        float(predictions[0][index])
        * 100
    )

    print(
        f"{class_name:10s}: "
        f"{probability:.2f}%"
    )

print("\n" + "=" * 60)

# --------------------------------------------------
# POPUP
# --------------------------------------------------

message = (
    f"Prediction:\n"
    f"{predicted_class.upper()}\n\n"
    f"Confidence:\n"
    f"{confidence:.2f}%\n\n"
    f"Cat probability:\n"
    f"{predictions[0][class_names.index('cat')] * 100:.2f}%\n\n"
    f"Non-Cat probability:\n"
    f"{predictions[0][class_names.index('non_cat')] * 100:.2f}%"
)

messagebox.showinfo(
    "PawCare - Cat Detector",
    message
)

root.destroy()