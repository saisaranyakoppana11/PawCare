import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PawCare AI",
    page_icon="🐾",
    layout="centered"
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)


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


IMG_SIZE = 224


# ============================================================
# CHECK MODEL FILES
# ============================================================

required_files = {
    "Cat Detector": CAT_DETECTOR_MODEL_PATH,
    "Cat Detector Classes": CAT_DETECTOR_CLASSES_PATH,
    "Cat Breed Classifier": CAT_BREED_MODEL_PATH,
    "Cat Breed Classes": CAT_BREED_CLASSES_PATH,
    "Dog Detector": DOG_DETECTOR_MODEL_PATH,
    "Dog Detector Classes": DOG_DETECTOR_CLASSES_PATH,
    "Dog Breed Classifier": DOG_BREED_MODEL_PATH,
    "Dog Breed Classes": DOG_BREED_CLASSES_PATH
}

missing_files = []

for name, path in required_files.items():

    if not os.path.exists(path):
        missing_files.append(
            f"{name}: {path}"
        )


if missing_files:

    st.error("Some required model files are missing.")

    for file in missing_files:
        st.write(file)

    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    # --------------------------------------------------------
    # CAT DETECTOR
    # --------------------------------------------------------

    cat_detector = tf.keras.models.load_model(
        CAT_DETECTOR_MODEL_PATH
    )

    with open(
        CAT_DETECTOR_CLASSES_PATH,
        "r"
    ) as file:

        cat_detector_classes = json.load(file)


    # --------------------------------------------------------
    # CAT BREED CLASSIFIER
    # --------------------------------------------------------

    cat_breed_model = tf.keras.models.load_model(
        CAT_BREED_MODEL_PATH
    )

    with open(
        CAT_BREED_CLASSES_PATH,
        "r"
    ) as file:

        cat_breed_classes = json.load(file)


    # --------------------------------------------------------
    # DOG DETECTOR
    # --------------------------------------------------------

    dog_detector = tf.keras.models.load_model(
        DOG_DETECTOR_MODEL_PATH
    )

    with open(
        DOG_DETECTOR_CLASSES_PATH,
        "r"
    ) as file:

        dog_detector_classes = json.load(file)


    # --------------------------------------------------------
    # DOG BREED CLASSIFIER
    # --------------------------------------------------------

    dog_breed_model = tf.keras.models.load_model(
        DOG_BREED_MODEL_PATH
    )

    with open(
        DOG_BREED_CLASSES_PATH,
        "r"
    ) as file:

        dog_breed_classes = json.load(file)


    return (
        cat_detector,
        cat_detector_classes,
        cat_breed_model,
        cat_breed_classes,
        dog_detector,
        dog_detector_classes,
        dog_breed_model,
        dog_breed_classes
    )


# ============================================================
# LOAD ALL MODELS
# ============================================================

with st.spinner("Loading PawCare AI models..."):

    (
        cat_detector,
        cat_detector_classes,
        cat_breed_model,
        cat_breed_classes,
        dog_detector,
        dog_detector_classes,
        dog_breed_model,
        dog_breed_classes
    ) = load_models()


# ============================================================
# HEADER
# ============================================================

st.title("🐾 PawCare AI")

st.subheader(
    "Unified Cat & Dog Breed Prediction"
)

st.write(
    "Upload a pet image and PawCare AI will "
    "detect whether it is a cat or dog and "
    "predict its breed."
)


st.divider()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload a pet image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    st.image(
        original_image,
        caption="Uploaded Image",
        width="stretch"
    )


    # --------------------------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------------------------

    resized_image = original_image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # ========================================================
    # START PREDICTION
    # ========================================================

    with st.spinner("Analyzing image..."):

        # ----------------------------------------------------
        # STAGE 1
        # CAT VS NON-CAT
        # ----------------------------------------------------

        cat_predictions = (
            cat_detector.predict(
                image_array,
                verbose=0
            )[0]
        )

        cat_index = np.argmax(
            cat_predictions
        )

        cat_class = cat_detector_classes[
            cat_index
        ]

        cat_confidence = (
            cat_predictions[
                cat_index
            ] * 100
        )


        # ====================================================
        # CAT PATH
        # ====================================================

        if cat_class == "cat":

            st.success("🐱 Cat detected!")

            st.metric(
                "Cat Detection Confidence",
                f"{cat_confidence:.2f}%"
            )


            # ------------------------------------------------
            # CAT BREED CLASSIFICATION
            # ------------------------------------------------

            cat_breed_predictions = (
                cat_breed_model.predict(
                    image_array,
                    verbose=0
                )[0]
            )

            top_indices = np.argsort(
                cat_breed_predictions
            )[::-1][:3]


            best_index = top_indices[0]

            best_breed = cat_breed_classes[
                best_index
            ]

            best_confidence = (
                cat_breed_predictions[
                    best_index
                ] * 100
            )


            st.subheader("🐱 Cat Breed Prediction")

            st.success(
                f"Predicted Breed: "
                f"{best_breed.replace('_', ' ')}"
            )

            st.metric(
                "Breed Confidence",
                f"{best_confidence:.2f}%"
            )


            # ------------------------------------------------
            # TOP 3
            # ------------------------------------------------

            st.subheader("Top 3 Predictions")

            for rank, index in enumerate(
                top_indices,
                start=1
            ):

                breed = cat_breed_classes[
                    index
                ].replace("_", " ")

                confidence = (
                    cat_breed_predictions[
                        index
                    ] * 100
                )

                st.write(
                    f"**{rank}. {breed}** "
                    f"— {confidence:.2f}%"
                )

                st.progress(
                    float(
                        cat_breed_predictions[
                            index
                        ]
                    )
                )


        # ====================================================
        # NON-CAT → DOG PATH
        # ====================================================

        else:

            st.info(
                "The image is not detected as a cat. "
                "Checking for dog..."
            )


            # ------------------------------------------------
            # DOG DETECTOR
            # ------------------------------------------------

            dog_predictions = (
                dog_detector.predict(
                    image_array,
                    verbose=0
                )[0]
            )

            dog_index = np.argmax(
                dog_predictions
            )

            dog_class = dog_detector_classes[
                dog_index
            ]

            dog_confidence = (
                dog_predictions[
                    dog_index
                ] * 100
            )


            # =================================================
            # DOG PATH
            # =================================================

            if dog_class == "dog":

                st.success("🐶 Dog detected!")

                st.metric(
                    "Dog Detection Confidence",
                    f"{dog_confidence:.2f}%"
                )


                # ---------------------------------------------
                # DOG BREED CLASSIFICATION
                # ---------------------------------------------

                dog_breed_predictions = (
                    dog_breed_model.predict(
                        image_array,
                        verbose=0
                    )[0]
                )

                top_indices = np.argsort(
                    dog_breed_predictions
                )[::-1][:3]


                best_index = top_indices[0]

                best_breed = dog_breed_classes[
                    best_index
                ]

                best_confidence = (
                    dog_breed_predictions[
                        best_index
                    ] * 100
                )


                st.subheader(
                    "🐶 Dog Breed Prediction"
                )

                st.success(
                    f"Predicted Breed: "
                    f"{best_breed.replace('_', ' ')}"
                )

                st.metric(
                    "Breed Confidence",
                    f"{best_confidence:.2f}%"
                )


                # ---------------------------------------------
                # TOP 3
                # ---------------------------------------------

                st.subheader("Top 3 Predictions")

                for rank, index in enumerate(
                    top_indices,
                    start=1
                ):

                    breed = dog_breed_classes[
                        index
                    ].replace("_", " ")

                    confidence = (
                        dog_breed_predictions[
                            index
                        ] * 100
                    )

                    st.write(
                        f"**{rank}. {breed}** "
                        f"— {confidence:.2f}%"
                    )

                    st.progress(
                        float(
                            dog_breed_predictions[
                                index
                            ]
                        )
                    )


            # =================================================
            # NEITHER CAT NOR DOG
            # =================================================

            else:

                st.warning(
                    "❌ This image is neither "
                    "detected as a cat nor a dog."
                )

                st.write(
                    "Breed classification was skipped."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🐾 PawCare AI • Cat & Dog Breed Classification"
)