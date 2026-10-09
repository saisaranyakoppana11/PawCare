import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox


# ============================================================
# PAWCARE - CAT BREED IMAGE PREDICTION
# ============================================================

MODEL_PATH = "models/cat_breed_classifier.keras"
CLASS_NAMES_PATH = "models/class_names.json"

IMG_SIZE = 224


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading PawCare cat breed model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ============================================================
# OPEN IMAGE SELECTION WINDOW
# ============================================================

root = tk.Tk()

root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select a Cat Image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.bmp"),
        ("JPG files", "*.jpg"),
        ("PNG files", "*.png"),
        ("All files", "*.*")
    ]
)


# ============================================================
# IF USER CANCELS
# ============================================================

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
# CONVERT IMAGE
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
# PREDICT
# ============================================================

print("\nAnalyzing cat image...")

predictions = model.predict(
    image_array,
    verbose=0
)


# ============================================================
# GET TOP 3 PREDICTIONS
# ============================================================

top_indices = np.argsort(
    predictions[0]
)[::-1][:3]


# ============================================================
# MAIN PREDICTION
# ============================================================

predicted_index = top_indices[0]

predicted_breed = class_names[
    predicted_index
]

confidence = (
    float(predictions[0][predicted_index])
    * 100
)


# ============================================================
# DISPLAY RESULT IN TERMINAL
# ============================================================

print("\n" + "=" * 60)
print("PAWCARE CAT BREED PREDICTION")
print("=" * 60)

print("\nPredicted Breed:")
print(predicted_breed)

print("\nConfidence:")
print(f"{confidence:.2f}%")

print("\nTop 3 Predictions:")
print("-" * 40)

for rank, index in enumerate(
    top_indices,
    start=1
):

    breed = class_names[index]

    breed_confidence = (
        float(predictions[0][index])
        * 100
    )

    print(
        f"{rank}. {breed:20s} "
        f"{breed_confidence:.2f}%"
    )


# ============================================================
# POPUP RESULT
# ============================================================

message = (
    f"Predicted Breed:\n"
    f"{predicted_breed}\n\n"
    f"Confidence:\n"
    f"{confidence:.2f}%\n\n"
    f"Top 3 Predictions:\n"
    f"1. {class_names[top_indices[0]]} - "
    f"{predictions[0][top_indices[0]] * 100:.2f}%\n"
    f"2. {class_names[top_indices[1]]} - "
    f"{predictions[0][top_indices[1]] * 100:.2f}%\n"
    f"3. {class_names[top_indices[2]]} - "
    f"{predictions[0][top_indices[2]] * 100:.2f}%"
)

messagebox.showinfo(
    "PawCare - Cat Breed Prediction",
    message
)

root.destroy()