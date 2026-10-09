"""
PawCare Stage 7D
Unified Predictor Test

Purpose:
    Test the actual predict_pet() function from ai/predictor.py.

Expected:
    1. Cat image -> cat + breed
    2. Dog image -> dog + breed
    3. Non-pet image -> unknown/rejected
"""

from pathlib import Path
import sys


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AI_DIR = PROJECT_ROOT / "ai"

TEST_IMAGE_DIR = AI_DIR / "test_images"


# ============================================================
# MAKE AI DIRECTORY IMPORTABLE
# ============================================================

if str(AI_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(AI_DIR)
    )


# ============================================================
# IMPORT UNIFIED PREDICTOR
# ============================================================

print("=" * 70)
print("PAWCARE STAGE 7D")
print("UNIFIED PREDICTOR TEST")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Test image directory:")
print(TEST_IMAGE_DIR)


try:

    from predictor import predict_pet

except Exception as error:

    print()
    print("=" * 70)
    print("ERROR: COULD NOT IMPORT UNIFIED PREDICTOR")
    print("=" * 70)

    print()
    print(error)

    raise


print()
print("=" * 70)
print("UNIFIED PREDICTOR IMPORTED SUCCESSFULLY")
print("=" * 70)


# ============================================================
# TEST IMAGES
# ============================================================

TEST_IMAGES = [
    (
        "CAT TEST",
        TEST_IMAGE_DIR / "cat_test.jpg"
    ),
    (
        "DOG TEST",
        TEST_IMAGE_DIR / "dog_test.jpg"
    ),
    (
        "NON-PET TEST",
        TEST_IMAGE_DIR / "non_pet_test.jpg"
    )
]


# ============================================================
# VERIFY TEST FILES
# ============================================================

print()
print("=" * 70)
print("CHECKING TEST IMAGES")
print("=" * 70)


for test_name, image_path in TEST_IMAGES:

    print()
    print(test_name)

    print("File:")
    print(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Required test image not found:\n"
            f"{image_path}"
        )

    print("Status: OK")


# ============================================================
# RUN UNIFIED PREDICTION
# ============================================================

results = {}


for test_name, image_path in TEST_IMAGES:

    print()
    print("=" * 70)
    print(test_name)
    print("=" * 70)

    print()
    print("Running:")
    print("predict_pet()")

    print()
    print("Image:")
    print(image_path)

    try:

        result = predict_pet(
            image_path
        )

        results[test_name] = result

    except Exception as error:

        print()
        print("=" * 70)
        print("PREDICTION ERROR")
        print("=" * 70)

        print()
        print(error)

        raise

    print()
    print("Prediction completed.")

    print()
    print("-" * 70)
    print("RESULT")
    print("-" * 70)

    print()

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )


# ============================================================
# VALIDATE CAT RESULT
# ============================================================

print()
print("=" * 70)
print("VALIDATING CAT RESULT")
print("=" * 70)

cat_result = results["CAT TEST"]

cat_passed = True


if cat_result.get("species") != "cat":

    print(
        "FAIL: Expected species = cat"
    )

    cat_passed = False


if cat_result.get("detected") is not True:

    print(
        "FAIL: Cat should be detected"
    )

    cat_passed = False


if not cat_result.get("breed"):

    print(
        "FAIL: Cat breed should be present"
    )

    cat_passed = False


if cat_passed:

    print(
        "CAT TEST: PASSED"
    )


# ============================================================
# VALIDATE DOG RESULT
# ============================================================

print()
print("=" * 70)
print("VALIDATING DOG RESULT")
print("=" * 70)

dog_result = results["DOG TEST"]

dog_passed = True


if dog_result.get("species") != "dog":

    print(
        "FAIL: Expected species = dog"
    )

    dog_passed = False


if dog_result.get("detected") is not True:

    print(
        "FAIL: Dog should be detected"
    )

    dog_passed = False


if not dog_result.get("breed"):

    print(
        "FAIL: Dog breed should be present"
    )

    dog_passed = False


if dog_passed:

    print(
        "DOG TEST: PASSED"
    )


# ============================================================
# VALIDATE NON-PET RESULT
# ============================================================

print()
print("=" * 70)
print("VALIDATING NON-PET RESULT")
print("=" * 70)

non_pet_result = results["NON-PET TEST"]

non_pet_passed = True


if non_pet_result.get("species") != "unknown":

    print(
        "FAIL: Expected species = unknown"
    )

    non_pet_passed = False


if non_pet_result.get("detected") is not False:

    print(
        "FAIL: Non-pet image should not be detected"
    )

    non_pet_passed = False


if non_pet_passed:

    print(
        "NON-PET TEST: PASSED"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("STAGE 7D FINAL VALIDATION")
print("=" * 70)

print()

print(
    f"Cat test     : {'PASSED' if cat_passed else 'FAILED'}"
)

print(
    f"Dog test     : {'PASSED' if dog_passed else 'FAILED'}"
)

print(
    f"Non-pet test : {'PASSED' if non_pet_passed else 'FAILED'}"
)


all_passed = (
    cat_passed
    and dog_passed
    and non_pet_passed
)


print()

if all_passed:

    print("=" * 70)
    print("STATUS: STAGE 7D UNIFIED PREDICTOR TEST PASSED")
    print("=" * 70)

else:

    print("=" * 70)
    print("STATUS: STAGE 7D UNIFIED PREDICTOR TEST FAILED")
    print("=" * 70)

    raise RuntimeError(
        "One or more unified predictor tests failed."
    )