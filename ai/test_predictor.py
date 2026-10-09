from pathlib import Path
import sys


# ============================================================
# PAWCARE REAL IMAGE PREDICTION TEST
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_IMAGES_DIR = PROJECT_ROOT / "ai" / "test_images"


# Make sure the project root can be imported
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ai.predictor import predict_pet


TEST_IMAGES = [
    ("CAT TEST", TEST_IMAGES_DIR / "cat_test.jpg"),
    ("DOG TEST", TEST_IMAGES_DIR / "dog_test.jpg"),
    ("NON-PET TEST", TEST_IMAGES_DIR / "non_pet_test.jpg"),
]


def print_prediction(test_name, image_path):
    print()
    print("=" * 70)
    print(test_name)
    print("=" * 70)

    print(f"Image: {image_path}")

    if not image_path.exists():
        print("STATUS: IMAGE NOT FOUND")
        print(f"Expected path: {image_path}")
        return False

    try:
        result = predict_pet(image_path)

        print()
        print("Prediction Result")
        print("-" * 70)

        print(f"Species             : {result.get('species')}")
        print(f"Is Pet              : {result.get('is_pet')}")
        print(f"Detected            : {result.get('detected')}")
        print(f"Message             : {result.get('message')}")
        print(
            f"Detector Confidence: "
            f"{result.get('detector_confidence')}"
        )
        print(f"Breed               : {result.get('breed')}")
        print(
            f"Breed Confidence    : "
            f"{result.get('breed_confidence')}"
        )

        top_predictions = result.get("top_predictions", [])

        print()
        print("Top Predictions")
        print("-" * 70)

        if top_predictions:
            for index, prediction in enumerate(
                top_predictions,
                start=1
            ):
                print(
                    f"{index}. "
                    f"{prediction.get('class')} "
                    f"-> "
                    f"{prediction.get('confidence')}"
                )
        else:
            print("No breed predictions available.")

        print()
        print("STATUS: PREDICTION COMPLETED")

        return True

    except Exception as error:

        print()
        print("STATUS: PREDICTION FAILED")
        print(f"Error: {error}")

        return False


def main():

    print("=" * 70)
    print("PAWCARE STAGE 7B")
    print("REAL IMAGE UNIFIED AI PREDICTION TEST")
    print("=" * 70)

    print()
    print(f"Project root:")
    print(PROJECT_ROOT)

    print()
    print(f"Test images directory:")
    print(TEST_IMAGES_DIR)

    print()
    print("Expected test images:")
    for test_name, image_path in TEST_IMAGES:
        print(f"- {test_name}: {image_path.name}")

    successful_tests = 0

    for test_name, image_path in TEST_IMAGES:

        success = print_prediction(
            test_name,
            image_path
        )

        if success:
            successful_tests += 1

    print()
    print("=" * 70)
    print("STAGE 7B TEST SUMMARY")
    print("=" * 70)

    print(
        f"Successful image tests: "
        f"{successful_tests}/{len(TEST_IMAGES)}"
    )

    if successful_tests == len(TEST_IMAGES):
        print()
        print("STATUS: STAGE 7B PREDICTION TEST COMPLETED")
    else:
        print()
        print("STATUS: STAGE 7B NEEDS ATTENTION")

    print("=" * 70)


if __name__ == "__main__":
    main()