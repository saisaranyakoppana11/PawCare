import os

DATASET_PATH = r"C:\PawCareData\cat_breed_split"

print("=" * 60)
print("PAWCARE CAT DATASET CHECK")
print("=" * 60)

if not os.path.exists(DATASET_PATH):
    print("\nERROR: Dataset not found!")
    print(DATASET_PATH)
    exit()

print("\nDataset found:")
print(DATASET_PATH)

for split in ["train", "validation", "test"]:

    split_path = os.path.join(DATASET_PATH, split)

    print(f"\n{split.upper()}")

    if not os.path.exists(split_path):
        print("  Split not found!")
        continue

    total = 0

    for breed in sorted(os.listdir(split_path)):

        breed_path = os.path.join(split_path, breed)

        if not os.path.isdir(breed_path):
            continue

        image_count = len([
            file for file in os.listdir(breed_path)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            )
        ])

        print(f"  {breed:20s}: {image_count}")

        total += image_count

    print(f"  Total: {total}")

print("\n" + "=" * 60)
print("CHECK COMPLETED")
print("=" * 60)