from pathlib import Path

# New copied dataset
DATASET = Path(r"C:\Users\saisa\Documents\PawCare\dog_disease_dataset_v2")

# Old YOLO class -> new YOLO class
CLASS_MAP = {
    0: 0,  # demodicosis
    1: 1,  # Dermatitis
    2: 2,  # Fungal_infections
    4: 3,  # Hypersensitivity
    5: 4,  # ringworm
}

splits = ["train", "valid", "test"]

print("Starting dog disease label remapping...\n")

for split in splits:
    label_dir = DATASET / "labels" / split

    if not label_dir.exists():
        print(f"Skipping {split}: label folder not found")
        continue

    changed = 0

    for label_file in label_dir.rglob("*.txt"):
        lines = label_file.read_text(encoding="utf-8").splitlines()

        new_lines = []

        for line in lines:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            try:
                old_class = int(parts[0])
            except ValueError:
                raise ValueError(
                    f"Invalid class ID in {label_file}: {parts[0]}"
                )

            if old_class == 3:
                raise ValueError(
                    f"Unexpected Healthy class found in:\n{label_file}\n"
                    "Healthy should NOT have YOLO bounding-box labels."
                )

            if old_class not in CLASS_MAP:
                raise ValueError(
                    f"Unknown class ID {old_class} found in:\n{label_file}"
                )

            parts[0] = str(CLASS_MAP[old_class])
            new_lines.append(" ".join(parts))

        label_file.write_text(
            "\n".join(new_lines) + ("\n" if new_lines else ""),
            encoding="utf-8"
        )

        changed += 1

    print(f"{split}: remapped {changed} label files")

# Write corrected dataset.yaml
yaml_file = DATASET / "dataset.yaml"

yaml_content = f"""path: {DATASET.as_posix()}

train: images/train
val: images/valid
test: images/test

names:
  0: demodicosis
  1: Dermatitis
  2: Fungal_infections
  3: Hypersensitivity
  4: ringworm
"""

yaml_file.write_text(yaml_content, encoding="utf-8")

print("\nDone!")
print(f"Dataset: {DATASET}")
print(f"YAML: {yaml_file}")
print("\nNew classes:")
print("0 = demodicosis")
print("1 = Dermatitis")
print("2 = Fungal_infections")
print("3 = Hypersensitivity")
print("4 = ringworm")