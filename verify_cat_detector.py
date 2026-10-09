import os
from PIL import Image

DATASET_DIR = r"C:\PawCareData\cat_detector"

SPLITS = ["train", "validation", "test"]
CLASSES = ["cat", "non_cat"]

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

total_images = 0
total_valid = 0
total_invalid = 0

print("=" * 60)
print("PAWCARE CAT VS NON-CAT DATASET VERIFICATION")
print("=" * 60)

for split in SPLITS:

    print(f"\n{split.upper()}")
    print("-" * 60)

    for class_name in CLASSES:

        folder = os.path.join(
            DATASET_DIR,
            split,
            class_name
        )

        if not os.path.exists(folder):

            print(f"{class_name}: FOLDER NOT FOUND")
            continue

        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(VALID_EXTENSIONS)
        ]

        valid = 0
        invalid = 0

        for filename in files:

            image_path = os.path.join(
                folder,
                filename
            )

            try:

                with Image.open(image_path) as image:

                    image.verify()

                valid += 1

            except Exception as error:

                invalid += 1

                print(
                    f"\nINVALID IMAGE:"
                    f"\n{image_path}"
                    f"\nError: {error}"
                )

        total_images += len(files)
        total_valid += valid
        total_invalid += invalid

        print(
            f"{class_name:8s}: "
            f"{len(files):4d} images | "
            f"Valid: {valid:4d} | "
            f"Invalid: {invalid:4d}"
        )

print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)

print(f"\nTotal images:   {total_images}")
print(f"Valid images:   {total_valid}")
print(f"Invalid images: {total_invalid}")

if total_invalid == 0:

    print("\n✅ ALL IMAGES ARE VALID")
    print("Dataset is ready for model training.")

else:

    print("\n⚠️ INVALID IMAGES FOUND")
    print("Do NOT train yet.")
    print("We need to remove or replace the invalid images.")

print("\n" + "=" * 60)