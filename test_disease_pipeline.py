from pathlib import Path
from ultralytics import YOLO


# ============================================================
# PAWCARE - DISEASE MODEL VALIDATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# MODEL PATHS
# ============================================================

SPECIES_MODEL_PATH = (
    BASE_DIR / "disease_models" / "yolo26n.pt"
)

CAT_DISEASE_MODEL_PATH = (
    BASE_DIR / "disease_models" / "cat" / "best.pt"
)

DOG_DISEASE_MODEL_PATH = (
    BASE_DIR / "disease_models" / "dog" / "best.pt"
)


# ============================================================
# IMAGE PATHS
# ============================================================

TEST_IMAGES_DIR = BASE_DIR / "test_images"

CAT_DIR = TEST_IMAGES_DIR / "cat"
DOG_DIR = TEST_IMAGES_DIR / "dog"
NON_PET_DIR = TEST_IMAGES_DIR / "non_pet"

OUTPUT_DIR = BASE_DIR / "test_outputs"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


# ============================================================
# GET CLASS NAME
# ============================================================

def get_class_name(model, class_id):

    names = model.names

    if isinstance(names, dict):
        return names.get(class_id, str(class_id))

    if 0 <= class_id < len(names):
        return names[class_id]

    return str(class_id)


# ============================================================
# CHECK FILE
# ============================================================

def check_file(path, description):

    if not path.exists():

        print(f"\n❌ {description} not found:")
        print(f"   {path}")

        return False

    print(f"✅ {description}")
    print(f"   {path}")

    return True


# ============================================================
# SHOW MODEL CLASSES
# ============================================================

def show_classes(model, title):

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    names = model.names

    if isinstance(names, dict):

        for class_id, class_name in names.items():
            print(f"{class_id}: {class_name}")

    else:

        for class_id, class_name in enumerate(names):
            print(f"{class_id}: {class_name}")


# ============================================================
# SPECIES DETECTION
# ============================================================

def detect_species(model, image_path):

    results = model.predict(
        source=str(image_path),
        conf=0.25,
        verbose=False
    )

    if not results:
        return None, 0.0

    result = results[0]

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):
        return None, 0.0

    best_species = None
    best_confidence = 0.0

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = get_class_name(
            model,
            class_id
        )

        class_name = class_name.lower()

        if class_name in {"cat", "dog"}:

            if confidence > best_confidence:

                best_species = class_name
                best_confidence = confidence

    return best_species, best_confidence


# ============================================================
# RUN DISEASE MODEL
# ============================================================

def run_disease_model(
    model,
    image_path,
    output_folder
):

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    results = model.predict(
        source=str(image_path),
        conf=0.10,
        save=True,
        project=str(output_folder),
        name="prediction",
        exist_ok=True,
        verbose=False
    )

    if not results:
        return []

    result = results[0]

    predictions = []

    # --------------------------------------------------------
    # Bounding-box predictions
    # --------------------------------------------------------

    if (
        result.boxes is not None
        and len(result.boxes) > 0
    ):

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = get_class_name(
                model,
                class_id
            )

            coordinates = [
                round(value, 2)
                for value in box.xyxy[0].tolist()
            ]

            predictions.append(
                {
                    "class": class_name,
                    "confidence": confidence,
                    "box": coordinates
                }
            )

    # --------------------------------------------------------
    # Classification probabilities
    # --------------------------------------------------------

    if result.probs is not None:

        top_class_id = int(
            result.probs.top1
        )

        top_confidence = float(
            result.probs.top1conf
        )

        top_class_name = get_class_name(
            model,
            top_class_id
        )

        predictions.append(
            {
                "class": top_class_name,
                "confidence": top_confidence,
                "box": None
            }
        )

    return predictions


# ============================================================
# TEST ONE ANIMAL
# ============================================================

def test_animal(
    species_model,
    disease_model,
    image_path,
    expected_species,
    expected_health,
    output_folder
):

    print("\n")
    print("=" * 70)

    if expected_species == "cat":
        print("🐱 CAT TEST")
    else:
        print("🐶 DOG TEST")

    print("=" * 70)

    print(f"Image: {image_path.name}")

    # --------------------------------------------------------
    # Species
    # --------------------------------------------------------

    species, species_confidence = detect_species(
        species_model,
        image_path
    )

    if species is None:

        print("\n❌ Species detection failed.")
        return

    print(
        f"\nSpecies detected: "
        f"{species.upper()}"
    )

    print(
        f"Species confidence: "
        f"{species_confidence:.4f}"
    )

    if species != expected_species:

        print(
            f"\n❌ Expected {expected_species}, "
            f"but detected {species}."
        )

        return

    print("✅ Species detection correct.")

    # --------------------------------------------------------
    # Disease model
    # --------------------------------------------------------

    print(
        f"\n🧬 Running {expected_species.upper()} "
        f"disease model..."
    )

    predictions = run_disease_model(
        disease_model,
        image_path,
        output_folder
    )

    if not predictions:

        print(
            "\n⚠️ No prediction returned."
        )

        return

    # --------------------------------------------------------
    # Display predictions
    # --------------------------------------------------------

    print("\nPredictions:")

    for index, prediction in enumerate(
        predictions,
        start=1
    ):

        print(
            f"\n  Prediction {index}"
        )

        print(
            f"  Class      : "
            f"{prediction['class']}"
        )

        print(
            f"  Confidence : "
            f"{prediction['confidence']:.4f}"
        )

        if prediction["box"] is not None:

            print(
                f"  Bounding Box: "
                f"{prediction['box']}"
            )

        else:

            print(
                "  Bounding Box: "
                "Not applicable"
            )

    # --------------------------------------------------------
    # Health expectation
    # --------------------------------------------------------

    best_prediction = max(
        predictions,
        key=lambda item: item["confidence"]
    )

    predicted_class = (
        best_prediction["class"].lower()
    )

    if expected_health:

        if predicted_class in {
            "health",
            "healthy"
        }:

            print(
                "\n✅ HEALTHY TEST PASSED"
            )

        else:

            print(
                "\n⚠️ HEALTHY TEST NEEDS REVIEW"
            )

            print(
                f"Expected a healthy class, "
                f"but top prediction was: "
                f"{best_prediction['class']}"
            )

    else:

        print(
            "\nℹ️ Diseased-image test."
        )

        print(
            "The predicted disease should be "
            "compared with the actual condition "
            "shown in the image."
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    prediction_folder = (
        output_folder / "prediction"
    )

    if prediction_folder.exists():

        annotated_images = [
            file
            for file in prediction_folder.iterdir()
            if (
                file.is_file()
                and file.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]

        if annotated_images:

            print(
                "\n🖼️ Annotated image:"
            )

            print(
                f"   {annotated_images[0]}"
            )


# ============================================================
# NON-PET TEST
# ============================================================

def test_non_pet(
    species_model,
    image_path
):

    print("\n")
    print("=" * 70)
    print("🖼️ NON-PET TEST")
    print("=" * 70)

    print(
        f"Image: {image_path.name}"
    )

    species, confidence = detect_species(
        species_model,
        image_path
    )

    if species is None:

        print(
            "\n✅ NON-PET TEST PASSED"
        )

        print(
            "No cat or dog was detected."
        )

    else:

        print(
            "\n⚠️ NON-PET TEST NEEDS REVIEW"
        )

        print(
            f"Detected: {species}"
        )

        print(
            f"Confidence: {confidence:.4f}"
        )


# ============================================================
# FIND IMAGE
# ============================================================

def find_image(folder, filename):

    path = folder / filename

    if path.exists():
        return path

    print(
        f"\n❌ Missing image:"
    )

    print(
        f"   {path}"
    )

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("🐾 PAWCARE")
    print("HEALTHY & DISEASED ANIMAL MODEL VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check models
    # --------------------------------------------------------

    print("\n📦 Checking model files...")

    required_models = [
        (
            SPECIES_MODEL_PATH,
            "Species detection model"
        ),
        (
            CAT_DISEASE_MODEL_PATH,
            "Cat disease model"
        ),
        (
            DOG_DISEASE_MODEL_PATH,
            "Dog disease model"
        )
    ]

    for path, description in required_models:

        if not check_file(
            path,
            description
        ):

            print(
                "\n❌ Required model is missing."
            )

            return

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print("\n🤖 Loading models...")

    try:

        species_model = YOLO(
            str(SPECIES_MODEL_PATH)
        )

        cat_model = YOLO(
            str(CAT_DISEASE_MODEL_PATH)
        )

        dog_model = YOLO(
            str(DOG_DISEASE_MODEL_PATH)
        )

    except Exception as error:

        print(
            "\n❌ Model loading failed:"
        )

        print(error)

        return

    print(
        "✅ All models loaded successfully."
    )

    # --------------------------------------------------------
    # Show classes
    # --------------------------------------------------------

    show_classes(
        species_model,
        "SPECIES MODEL"
    )

    show_classes(
        cat_model,
        "CAT DISEASE MODEL"
    )

    show_classes(
        dog_model,
        "DOG DISEASE MODEL"
    )

    # ========================================================
    # CAT TESTS
    # ========================================================

    healthy_cat = find_image(
        CAT_DIR,
        "healthy_cat.jpg"
    )

    diseased_cat = find_image(
        CAT_DIR,
        "diseased_cat.jpg"
    )

    if healthy_cat:

        test_animal(
            species_model,
            cat_model,
            healthy_cat,
            "cat",
            True,
            OUTPUT_DIR / "cat" / "healthy"
        )

    if diseased_cat:

        test_animal(
            species_model,
            cat_model,
            diseased_cat,
            "cat",
            False,
            OUTPUT_DIR / "cat" / "diseased"
        )

    # ========================================================
    # DOG TESTS
    # ========================================================

    healthy_dog = find_image(
        DOG_DIR,
        "healthy_dog.jpg"
    )

    diseased_dog = find_image(
        DOG_DIR,
        "diseased_dog.jpg"
    )

    if healthy_dog:

        test_animal(
            species_model,
            dog_model,
            healthy_dog,
            "dog",
            True,
            OUTPUT_DIR / "dog" / "healthy"
        )

    if diseased_dog:

        test_animal(
            species_model,
            dog_model,
            diseased_dog,
            "dog",
            False,
            OUTPUT_DIR / "dog" / "diseased"
        )

    # ========================================================
    # NON-PET
    # ========================================================

    non_pet = find_image(
        NON_PET_DIR,
        "non_pet_test.jpg"
    )

    if non_pet:

        test_non_pet(
            species_model,
            non_pet
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("🏁 VALIDATION COMPLETED")
    print("=" * 70)

    print(
        f"\nAnnotated outputs:"
    )

    print(
        f"   {OUTPUT_DIR}"
    )

    print(
        "\n⚠️ Existing PawCare files were not modified."
    )

    print(
        "⚠️ Flask integration has not started."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()