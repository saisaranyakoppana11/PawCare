"""
PawCare Disease Detection Module

Connects the teammate's YOLO-based cat and dog skin-disease
models to the PawCare application.

This module does NOT replace the existing breed prediction system.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DISEASE_MODELS_DIR = PROJECT_ROOT / "disease_models"

SPECIES_MODEL_PATH = DISEASE_MODELS_DIR / "yolo26n.pt"
CAT_DISEASE_MODEL_PATH = DISEASE_MODELS_DIR / "cat" / "best.pt"
DOG_DISEASE_MODEL_PATH = DISEASE_MODELS_DIR / "dog" / "best.pt"


# ============================================================
# SETTINGS
# ============================================================

SPECIES_CONFIDENCE_THRESHOLD = 0.35
DISEASE_CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640


# ============================================================
# MODEL STORAGE
# ============================================================

_species_model: Optional[YOLO] = None
_cat_disease_model: Optional[YOLO] = None
_dog_disease_model: Optional[YOLO] = None


# ============================================================
# MODEL LOADING
# ============================================================

def _check_model_files() -> None:
    """Verify that all required model files exist."""

    missing_models = []

    if not SPECIES_MODEL_PATH.exists():
        missing_models.append(str(SPECIES_MODEL_PATH))

    if not CAT_DISEASE_MODEL_PATH.exists():
        missing_models.append(str(CAT_DISEASE_MODEL_PATH))

    if not DOG_DISEASE_MODEL_PATH.exists():
        missing_models.append(str(DOG_DISEASE_MODEL_PATH))

    if missing_models:
        raise FileNotFoundError(
            "Required PawCare disease model file(s) not found:\n"
            + "\n".join(missing_models)
        )


def _load_species_model() -> YOLO:
    """Load the cat/dog species detection model."""

    global _species_model

    if _species_model is None:
        _check_model_files()
        _species_model = YOLO(str(SPECIES_MODEL_PATH))

    return _species_model


def _load_cat_disease_model() -> YOLO:
    """Load the cat disease detection model."""

    global _cat_disease_model

    if _cat_disease_model is None:
        _check_model_files()
        _cat_disease_model = YOLO(str(CAT_DISEASE_MODEL_PATH))

    return _cat_disease_model


def _load_dog_disease_model() -> YOLO:
    """Load the dog disease detection model."""

    global _dog_disease_model

    if _dog_disease_model is None:
        _check_model_files()
        _dog_disease_model = YOLO(str(DOG_DISEASE_MODEL_PATH))

    return _dog_disease_model


# ============================================================
# INPUT VALIDATION
# ============================================================

def _validate_image(image_path: str) -> None:
    """Validate that the supplied path contains a readable image."""

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as error:
        raise ValueError(
            f"Unable to read the supplied image: {error}"
        ) from error


# ============================================================
# SPECIES DETECTION
# ============================================================

def detect_species(image_path: str) -> Dict[str, Any]:
    """
    Detect whether the image contains a cat or dog.

    Returns the detected species and confidence.
    """

    _validate_image(image_path)

    model = _load_species_model()

    results = model.predict(
        source=image_path,
        conf=SPECIES_CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )

    if not results:
        return {
            "species": "unknown",
            "confidence": 0.0,
            "detected": False,
        }

    result = results[0]

    cat_confidence = 0.0
    dog_confidence = 0.0

    if result.boxes is not None and len(result.boxes) > 0:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = str(
                result.names[class_id]
            ).lower()

            if class_name == "cat":
                cat_confidence = max(
                    cat_confidence,
                    confidence,
                )

            elif class_name == "dog":
                dog_confidence = max(
                    dog_confidence,
                    confidence,
                )

    if cat_confidence >= dog_confidence and cat_confidence >= SPECIES_CONFIDENCE_THRESHOLD:
        return {
            "species": "cat",
            "confidence": cat_confidence,
            "detected": True,
        }

    if dog_confidence > cat_confidence and dog_confidence >= SPECIES_CONFIDENCE_THRESHOLD:
        return {
            "species": "dog",
            "confidence": dog_confidence,
            "detected": True,
        }

    return {
        "species": "unknown",
        "confidence": 0.0,
        "detected": False,
    }


# ============================================================
# DISEASE DETECTION
# ============================================================

def detect_disease(
    image_path: str,
    species: str,
) -> Dict[str, Any]:
    """
    Run the appropriate disease detector for the specified species.
    """

    _validate_image(image_path)

    normalized_species = str(species).strip().lower()

    if normalized_species == "cat":
        model = _load_cat_disease_model()

    elif normalized_species == "dog":
        model = _load_dog_disease_model()

    else:
        raise ValueError(
            "Disease detection supports only cat and dog."
        )

    results = model.predict(
        source=image_path,
        conf=DISEASE_CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )

    if not results:
        return {
            "detected": False,
            "species": normalized_species,
            "detections": [],
            "annotated_image": None,
        }

    result = results[0]

    detections: List[Dict[str, Any]] = []

    if result.boxes is not None and len(result.boxes) > 0:

        for index, box in enumerate(
            result.boxes,
            start=1,
        ):

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            disease_name = str(
                model.names[class_id]
            )

            coordinates = [
                float(value)
                for value in box.xyxy[0].tolist()
            ]

            detections.append(
                {
                    "detection_number": index,
                    "disease": disease_name,
                    "confidence": confidence,
                    "confidence_percent": round(
                        confidence * 100,
                        1,
                    ),
                    "bounding_box": {
                        "x1": coordinates[0],
                        "y1": coordinates[1],
                        "x2": coordinates[2],
                        "y2": coordinates[3],
                    },
                }
            )

    annotated_image = result.plot()

    return {
        "detected": len(detections) > 0,
        "species": normalized_species,
        "detections": detections,
        "annotated_image": annotated_image,
    }


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def predict_pet_disease(image_path: str) -> Dict[str, Any]:
    """
    Complete PawCare disease-detection pipeline.

    1. Detect cat/dog.
    2. Select the appropriate disease model.
    3. Detect affected regions/conditions.
    4. Return structured results.
    """

    try:

        species_result = detect_species(image_path)

        if not species_result["detected"]:

            return {
                "status": "unsupported",
                "species": "unknown",
                "species_confidence": 0.0,
                "disease_detected": False,
                "detections": [],
                "annotated_image": None,
                "message": (
                    "A clearly identifiable cat or dog "
                    "was not detected in the image."
                ),
            }

        species = species_result["species"]

        disease_result = detect_disease(
            image_path,
            species,
        )

        if disease_result["detected"]:

            message = (
                f"{len(disease_result['detections'])} "
                f"possible condition detection(s) found."
            )

        else:

            message = (
                "No skin condition was detected by "
                "the current disease model."
            )

        return {
            "status": "success",
            "species": species,
            "species_confidence": round(
                species_result["confidence"] * 100,
                1,
            ),
            "disease_detected": disease_result["detected"],
            "detections": disease_result["detections"],
            "annotated_image": disease_result[
                "annotated_image"
            ],
            "message": message,
        }

    except Exception as error:

        return {
            "status": "error",
            "species": "unknown",
            "species_confidence": 0.0,
            "disease_detected": False,
            "detections": [],
            "annotated_image": None,
            "message": (
                "Disease detection could not be completed."
            ),
            "error": str(error),
        }


# ============================================================
# DIRECT MODULE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PAWCARE DISEASE DETECTION MODULE")
    print("=" * 70)
    print()

    print("Project root:")
    print(PROJECT_ROOT)
    print()

    print("Species model:")
    print(SPECIES_MODEL_PATH)
    print()

    print("Cat disease model:")
    print(CAT_DISEASE_MODEL_PATH)
    print()

    print("Dog disease model:")
    print(DOG_DISEASE_MODEL_PATH)
    print()

    try:
        _check_model_files()

        print("Model files: PASSED")
        print("Ultralytics import: PASSED")
        print()
        print("STATUS: DISEASE DETECTION MODULE READY")

    except Exception as error:

        print("STATUS: FAILED")
        print(error)

    print("=" * 70)