from pathlib import Path
from ultralytics import YOLO

ROOT = Path(r"C:\Users\saisa\Documents\PawCare")

MODELS = {
    "CAT": ROOT / "disease_models" / "cat" / "best.pt",
    "DOG": ROOT / "disease_models" / "dog" / "best.pt",
}


def inspect_model(name, path):
    print("\n" + "=" * 70)
    print(f"{name} DISEASE MODEL")
    print("=" * 70)

    if not path.exists():
        print(f"❌ Model not found: {path}")
        return

    print(f"Model file : {path}")
    print(f"File size  : {path.stat().st_size / (1024 * 1024):.2f} MB")

    try:
        model = YOLO(str(path))

        print(f"Task       : {model.task}")
        print(f"Model type : {type(model.model).__name__}")

        print("\nClasses:")
        for class_id, class_name in model.names.items():
            print(f"  {class_id}: {class_name}")

        print("\nModel configuration:")

        if hasattr(model, "overrides"):
            print(f"Overrides  : {model.overrides}")

        if hasattr(model.model, "yaml"):
            print(f"Model YAML : {model.model.yaml}")

        print("\n✅ Model inspection completed.")

    except Exception as e:
        print(f"❌ Error loading model: {e}")


print("=" * 70)
print("🐾 PAWCARE DISEASE MODEL INSPECTION")
print("=" * 70)

inspect_model("CAT", MODELS["CAT"])
inspect_model("DOG", MODELS["DOG"])

print("\n" + "=" * 70)
print("🏁 INSPECTION COMPLETED")
print("=" * 70)