"""
PawCare - Teammate Health & Disease Model Integration Adapter
=============================================================
This module defines the architectural placeholder and standardized pluggable adapter
for integrating the teammate's forthcoming clinical disease/health model.

STRICT CONSTRAINTS ADHERED TO:
- Does NOT fabricate or invent disease predictions.
- Does NOT create fake disease classes or arbitrary probabilities.
- Does NOT assume model framework, input dimensions, or architecture.
- Provides a clean, modular integration point so the teammate's model
  can be attached without redesigning the platform.

Mandated End-to-End Pipeline Flow:
Image
  ↓
Species detection (pet_ai.predict_pet_image)
  ↓
Breed prediction (pet_ai.predict_pet_image)
  ↓
Teammate health/disease model (via BaseTeammateHealthAdapter)
  ↓
Disease/condition prediction (upon teammate integration)
  ↓
Affected location (upon teammate integration)
  ↓
Confidence/result (upon teammate integration)
"""

import os
from abc import ABC, abstractmethod
from pet_ai import predict_pet_image, AI_DISCLAIMER


class BaseTeammateHealthAdapter(ABC):
    """
    Abstract base class for the teammate's clinical health and disease model.
    When the teammate is ready to connect their model, they can subclass this
    and implement the inference hooks.
    """

    @abstractmethod
    def is_ready(self) -> bool:
        """Returns True if the model weights are loaded and ready for inference."""
        pass

    @abstractmethod
    def predict_disease(self, image_input, species: str, breed: str, **kwargs) -> dict:
        """
        Executes disease/health prediction on the given image.
        Expected return format:
        {
            "conditions": list[dict],       # Predicted conditions or findings
            "affected_location": dict|str,  # Localization/heatmap/region
            "confidence": float,            # Overall confidence score
            "notes": str                    # Clinical observations
        }
        """
        pass

    @abstractmethod
    def get_metadata(self) -> dict:
        """Returns model information (framework, classes, version, author)."""
        pass


class TeammateHealthModelPlaceholder(BaseTeammateHealthAdapter):
    """
    Default placeholder adapter active while the teammate develops the health model.
    Confirms that the integration pipeline is active, without fabricating disease results.
    """

    def is_ready(self) -> bool:
        return False

    def predict_disease(self, image_input, species: str, breed: str, **kwargs) -> dict:
        raise NotImplementedError(
            "Teammate health/disease model is not yet attached. "
            "PawCare strictly avoids generating fabricated disease predictions."
        )

    def get_metadata(self) -> dict:
        return {
            "status": "awaiting_teammate_model",
            "ready": False,
            "message": "Teammate disease model integration slot is reserved and ready.",
            "pipeline_stage": "Stage 12 Placeholder"
        }


# Global active health adapter instance (defaults to placeholder)
_CURRENT_HEALTH_ADAPTER: BaseTeammateHealthAdapter = TeammateHealthModelPlaceholder()


def register_health_model(adapter: BaseTeammateHealthAdapter):
    """
    Registers a custom health model adapter provided by the teammate.
    """
    global _CURRENT_HEALTH_ADAPTER
    if not isinstance(adapter, BaseTeammateHealthAdapter):
        raise TypeError("Adapter must inherit from BaseTeammateHealthAdapter")
    _CURRENT_HEALTH_ADAPTER = adapter


def reset_health_model():
    """Resets the health model adapter back to the default placeholder."""
    global _CURRENT_HEALTH_ADAPTER
    _CURRENT_HEALTH_ADAPTER = TeammateHealthModelPlaceholder()


def get_health_model() -> BaseTeammateHealthAdapter:
    """Returns the currently active health model adapter."""
    return _CURRENT_HEALTH_ADAPTER


def is_health_model_ready() -> bool:
    """Returns whether the teammate's model has been connected and is ready."""
    return _CURRENT_HEALTH_ADAPTER.is_ready()


def run_health_analysis_pipeline(image_input, pet_metadata: dict = None) -> dict:
    """
    Executes the unified 5-stage health pipeline:
    1. Image Preprocessing
    2. Species Detection (Dog or Cat)
    3. Breed Classification
    4. Teammate Disease Model (if registered)
    5. Final Diagnostic preparation output

    If the teammate model is not yet registered:
    - Retains species and breed detection from existing models.
    - Accurately reports pipeline progress up to breed prediction.
    - Explicitly marks disease prediction and affected location as awaiting teammate model.
    - Preserves strict anti-fabrication safety.
    """
    # -------------------------------------------------------------
    # Stage 1 & 2: Species & Breed Classification (Live Models)
    # -------------------------------------------------------------
    vision_result = predict_pet_image(image_input)

    if not vision_result.get("is_supported", False):
        return {
            "status": "unsupported_image",
            "is_supported": False,
            "pipeline_stage_reached": "species_detection",
            "species": vision_result.get("species", "Unknown"),
            "breed": None,
            "breed_confidence": 0.0,
            "disease_model_attached": False,
            "disease_prediction": None,
            "affected_location": None,
            "message": vision_result.get("message", "Could not identify a supported pet species."),
            "disclaimer": AI_DISCLAIMER
        }

    species = vision_result.get("species")
    breed = vision_result.get("breed")
    breed_conf = vision_result.get("confidence", 0.0)

    # -------------------------------------------------------------
    # Stage 3: Teammate Disease Model Integration Slot
    # -------------------------------------------------------------
    adapter = get_health_model()

    if adapter.is_ready():
        try:
            disease_output = adapter.predict_disease(
                image_input=image_input,
                species=species,
                breed=breed,
                pet_metadata=pet_metadata or {}
            )
            return {
                "status": "success",
                "is_supported": True,
                "pipeline_stage_reached": "disease_prediction_complete",
                "species": species,
                "breed": breed,
                "breed_confidence": breed_conf,
                "top_breed_predictions": vision_result.get("top_predictions", []),
                "disease_model_attached": True,
                "disease_prediction": disease_output.get("conditions"),
                "affected_location": disease_output.get("affected_location"),
                "confidence": disease_output.get("confidence"),
                "notes": disease_output.get("notes"),
                "message": "Full pipeline completed including teammate disease analysis.",
                "disclaimer": AI_DISCLAIMER
            }
        except Exception as e:
            return {
                "status": "error_in_teammate_model",
                "is_supported": True,
                "pipeline_stage_reached": "teammate_model_execution",
                "species": species,
                "breed": breed,
                "breed_confidence": breed_conf,
                "disease_model_attached": True,
                "disease_prediction": None,
                "affected_location": None,
                "error_details": str(e),
                "message": f"Teammate model encountered an error: {str(e)}",
                "disclaimer": AI_DISCLAIMER
            }

    # -------------------------------------------------------------
    # Default State: Teammate Model Placeholder
    # -------------------------------------------------------------
    return {
        "status": "awaiting_teammate_model",
        "is_supported": True,
        "pipeline_stage_reached": "breed_prediction",
        "species": species,
        "breed": breed,
        "breed_confidence": breed_conf,
        "top_breed_predictions": vision_result.get("top_predictions", []),
        "disease_model_attached": False,
        "disease_prediction": None,
        "affected_location": None,
        "message": (
            f"Species ({species}) and breed ({breed}) successfully identified with {breed_conf:.1f}% confidence. "
            "Teammate disease and health analysis model integration point is ready."
        ),
        "disclaimer": AI_DISCLAIMER,
        "metadata": adapter.get_metadata()
    }
