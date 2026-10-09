from pathlib import Path
from ultralytics import YOLO

PROJECT = Path(r"C:\Users\saisa\Documents\PawCare")

DATASET = PROJECT / "dog_disease_dataset_v2"
YAML_FILE = DATASET / "dataset.yaml"

BASE_MODEL = PROJECT / "disease_models" / "yolo26n.pt"

OUTPUT_DIR = PROJECT / "disease_training_outputs"


print("=" * 60)
print("PAWCARE DOG DISEASE MODEL TRAINING")
print("=" * 60)

print(f"\nDataset : {DATASET}")
print(f"Model   : {BASE_MODEL}")
print(f"Output  : {OUTPUT_DIR}")

# Load pretrained YOLO model
model = YOLO(str(BASE_MODEL))

print("\nStarting training...")
print("Existing disease_models/dog/best.pt will NOT be changed.\n")

results = model.train(
    data=str(YAML_FILE),

    # Training
    epochs=40,
    imgsz=640,
    batch=8,

    # Windows CPU
    workers=0,
    device="cpu",
    cache=False,

    # Training control
    patience=10,

    # Output
    project=str(OUTPUT_DIR),
    name="dog_v2",
    exist_ok=True,

    # Save checkpoints
    save=True,
    save_period=5,

    verbose=True
)

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

best_model = OUTPUT_DIR / "dog_v2" / "weights" / "best.pt"

print(f"\nBest model:")
print(best_model)

# Load best model
if best_model.exists():

    print("\nRunning final TEST evaluation...")

    trained_model = YOLO(str(best_model))

    metrics = trained_model.val(
        data=str(YAML_FILE),
        split="test",
        imgsz=640,
        batch=8,
        workers=0,
        device="cpu"
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    print(f"mAP50     : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")

else:
    print("\nERROR: best.pt was not created.")