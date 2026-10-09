from pathlib import Path
from ultralytics import YOLO

PROJECT = Path(r"C:\Users\saisa\Documents\PawCare")

DATASET = PROJECT / "dog_disease_final_dataset"
YAML_FILE = DATASET / "dataset.yaml"

BASE_MODEL = PROJECT / "disease_models" / "yolo26n.pt"

OUTPUT = PROJECT / "disease_training_outputs"

print("=" * 60)
print("PAWCARE IMPROVED DOG DISEASE MODEL")
print("=" * 60)

model = YOLO(str(BASE_MODEL))

model.train(
    data=str(YAML_FILE),

    epochs=30,
    imgsz=512,
    batch=8,

    device="cpu",
    workers=0,
    cache=False,

    patience=10,

    # stronger augmentation
    degrees=10,
    translate=0.15,
    scale=0.5,
    fliplr=0.5,

    mosaic=1.0,
    close_mosaic=10,

    project=str(OUTPUT),
    name="dog_improved",
    exist_ok=True,

    save=True,
    save_period=5,

    verbose=True
)

best_model = (
    OUTPUT /
    "dog_improved" /
    "weights" /
    "best.pt"
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("Best model:")
print(best_model)

if not best_model.exists():
    raise FileNotFoundError(
        f"best.pt was not created: {best_model}"
    )

print("\nRunning TEST evaluation...\n")

trained = YOLO(str(best_model))

metrics = trained.val(
    data=str(YAML_FILE),
    split="test",
    imgsz=512,
    batch=8,
    workers=0,
    device="cpu"
)

print("\n" + "=" * 60)
print("IMPROVED DOG MODEL - TEST RESULTS")
print("=" * 60)

print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")
print(f"mAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")

print("\nModel:")
print(best_model)

print("\nExisting disease_models/dog/best.pt was NOT modified.")