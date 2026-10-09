import os

BASE_DIR = r"C:\PawCareData\cat_detector"

splits = ["train", "validation", "test"]
classes = ["cat", "non_cat"]

print("=" * 60)
print("CREATING PAWCARE CAT DETECTOR DATASET")
print("=" * 60)

for split in splits:
    for class_name in classes:

        folder = os.path.join(
            BASE_DIR,
            split,
            class_name
        )

        os.makedirs(folder, exist_ok=True)

        print(f"Created: {folder}")

print("\n" + "=" * 60)
print("DATASET FOLDERS CREATED SUCCESSFULLY")
print("=" * 60)

print("\nDataset structure:")
print(BASE_DIR)
print("├── train")
print("│   ├── cat")
print("│   └── non_cat")
print("├── validation")
print("│   ├── cat")
print("│   └── non_cat")
print("└── test")
print("    ├── cat")
print("    └── non_cat")