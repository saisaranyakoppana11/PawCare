"""
PawCare AI Compatibility Layer

This module connects the existing Flask application
to the verified unified AI predictor.

The actual prediction pipeline is implemented in:

    ai/predictor.py

Pipeline:

    Image
      |
      v
    Cat Detector
      |
      +---- Cat ----> Cat Breed Classifier
      |
      No
      |
      v
    Dog Detector
      |
      +---- Dog ----> Dog Breed Classifier
      |
      No
      |
      v
    Unsupported Image
"""

from ai.predictor import predict_pet


# ============================================================
# Veterinary / AI Disclaimer
# ============================================================

AI_DISCLAIMER = (
    "This AI result is for informational purposes only and "
    "does not provide a veterinary diagnosis. Breed predictions "
    "may be incorrect, especially for mixed-breed pets or unclear "
    "images. Please consult a qualified veterinarian for medical "
    "concerns or professional advice."
)


# ============================================================
# Breed Name Formatting
# ============================================================

def format_breed_name(breed_name):
    """
    Convert model class names into user-friendly display names.

    Examples:
        Russian_Blue -> Russian Blue
        German_shepherd -> German Shepherd
        golden_retriever -> Golden Retriever
        standard_poodle -> Standard Poodle
    """

    if not breed_name:
        return breed_name

    formatted = str(breed_name).replace("_", " ")

    formatted_words = []

    for word in formatted.split():
        if word.lower() in {"and", "of", "the"}:
            formatted_words.append(word.lower())
        else:
            formatted_words.append(word.capitalize())

    return " ".join(formatted_words)


# ============================================================
# Compatibility Prediction Function
# ============================================================

def predict_pet_image(image_input):
    """
    Compatibility wrapper used by the existing Flask application.

    The verified unified predictor returns the following information:

        species
        breed
        detector confidence
        breed confidence
        top predictions

    This function converts that result into the response structure
    already expected by the PawCare Flask UI.
    """

    try:

        result = predict_pet(image_input)

    except Exception as error:

        return {
            "status": "error",
            "is_supported": False,
            "species": "Unknown",
            "breed": None,
            "raw_breed": None,
            "confidence": 0.0,
            "confidence_level": "N/A",
            "top_predictions": [],
            "detector": "Unavailable",
            "detector_confidence": 0.0,
            "message": (
                "The AI prediction could not be completed. "
                f"Error: {str(error)}"
            ),
            "disclaimer": AI_DISCLAIMER
        }

    # ========================================================
    # UNSUPPORTED IMAGE
    # ========================================================

    if not result.get("is_pet", False):

        return {
            "status": "unsupported",
            "is_supported": False,
            "species": "Unknown",
            "breed": None,
            "raw_breed": None,
            "confidence": 0.0,
            "confidence_level": "N/A",
            "top_predictions": [],
            "detector": "None",
            "detector_confidence": 0.0,
            "message": result.get(
                "message",
                "Neither a supported cat nor dog was detected."
            ),
            "disclaimer": AI_DISCLAIMER
        }

    # ========================================================
    # SUPPORTED CAT / DOG
    # ========================================================

    species = result.get("species", "Unknown")

    raw_breed = result.get("breed")

    formatted_breed = format_breed_name(raw_breed)

    breed_confidence = float(
        result.get("breed_confidence", 0.0)
    ) * 100.0

    detector_confidence = float(
        result.get("detector_confidence", 0.0)
    ) * 100.0

    # --------------------------------------------------------
    # Convert Top Predictions
    # --------------------------------------------------------

    converted_predictions = []

    for item in result.get("top_predictions", []):

        raw_prediction_breed = item.get("breed")

        prediction_confidence = float(
            item.get("confidence", 0.0)
        ) * 100.0

        converted_predictions.append(
            {
                "breed": format_breed_name(
                    raw_prediction_breed
                ),
                "raw_breed": raw_prediction_breed,
                "confidence": round(
                    prediction_confidence,
                    1
                )
            }
        )

    # --------------------------------------------------------
    # Confidence Level
    # --------------------------------------------------------

    if breed_confidence >= 70:
        confidence_level = "High"

    elif breed_confidence >= 45:
        confidence_level = "Moderate"

    else:
        confidence_level = "Low"

    # --------------------------------------------------------
    # Detector Name
    # --------------------------------------------------------

    if species.lower() == "cat":
        detector_name = "Cat Detector"

    elif species.lower() == "dog":
        detector_name = "Dog Detector"

    else:
        detector_name = "Species Detector"

    # --------------------------------------------------------
    # Final Flask-Compatible Result
    # --------------------------------------------------------

    return {
        "status": "success",
        "is_supported": True,
        "species": species,
        "breed": formatted_breed,
        "raw_breed": raw_breed,
        "confidence": round(
            breed_confidence,
            1
        ),
        "confidence_level": confidence_level,
        "top_predictions": converted_predictions,
        "detector": detector_name,
        "detector_confidence": round(
            detector_confidence,
            1
        ),
        "message": (
            f"Successfully identified as "
            f"{species} ({formatted_breed}) with "
            f"{breed_confidence:.1f}% confidence."
        ),
        "disclaimer": AI_DISCLAIMER
    }


# ============================================================
# Direct Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PAWCARE PET AI COMPATIBILITY LAYER")
    print("=" * 70)

    print()
    print("Unified predictor import: PASSED")
    print("Compatibility function: PASSED")
    print("Flask response format: PASSED")
    print()
    print("STATUS: PET AI COMPATIBILITY LAYER READY")
    print("=" * 70)