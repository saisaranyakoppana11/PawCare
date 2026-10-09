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
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "dog_breed",
    "models",
    "dog_breed_classifier.keras"
)

CLASS_NAMES_PATH = os.path.join(
    PROJECT_ROOT,
    "dog_breed",
    "models",
    "dog_class_names.json"
)

IMG_SIZE = 224


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Dog Breed Classifier...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("Model loaded successfully!")
print("Classes:", class_names)


# ============================================================
# SELECT IMAGE
# ============================================================

root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select a Dog Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")
    ]
)

if not image_path:
    print("\nNo image selected.")
    root.destroy()
    exit()


print("\nSelected image:")
print(image_path)


# ============================================================
# LOAD AND PREPARE IMAGE
# ============================================================

image = Image.open(image_path).convert("RGB")

display_image = image.copy()

image = image.resize((IMG_SIZE, IMG_SIZE))

image_array = np.array(image, dtype=np.float32)

# EfficientNetB0 preprocessing is already included
# inside the Keras EfficientNet model.
image_array = np.expand_dims(image_array, axis=0)


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(image_array, verbose=0)[0]

top_indices = np.argsort(predictions)[::-1][:3]


# ============================================================
# TOP 3 RESULTS
# ============================================================

print("\n" + "=" * 50)
print("DOG BREED PREDICTION")
print("=" * 50)

for rank, index in enumerate(top_indices, start=1):

    breed = class_names[index]
    confidence = predictions[index] * 100

    print(
        f"{rank}. {breed.replace('_', ' ')} "
        f"- {confidence:.2f}%"
    )


# ============================================================
# BEST PREDICTION
# ============================================================

best_index = top_indices[0]

best_breed = class_names[best_index]
best_confidence = predictions[best_index] * 100

print("\n" + "-" * 50)
print(
    f"Predicted Breed : {best_breed.replace('_', ' ')}"
)
print(
    f"Confidence      : {best_confidence:.2f}%"
)
print("-" * 50)


# ============================================================
# SHOW RESULT
# ============================================================

result_message = (
    f"Predicted Breed:\n"
    f"{best_breed.replace('_', ' ')}\n\n"
    f"Confidence:\n"
    f"{best_confidence:.2f}%\n\n"
    f"Top 3 Predictions:\n"
)

for rank, index in enumerate(top_indices, start=1):

    breed = class_names[index].replace("_", " ")
    confidence = predictions[index] * 100

    result_message += (
        f"{rank}. {breed} - {confidence:.2f}%\n"
    )

messagebox.showinfo(
    "PawCare - Dog Breed Prediction",
    result_message
)

root.destroy()