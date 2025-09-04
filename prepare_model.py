from transformers import ViTForImageClassification
from config import BASE_MODEL_PATH, MODEL_NAME, NUM_LABELS

def prepare_base_model():
    """Prepares and saves the base Vision Transformer model."""
    if not BASE_MODEL_PATH.exists():
        print("Preparing base model...")
        base_model = ViTForImageClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True
        )
        base_model.save_pretrained(BASE_MODEL_PATH)
        print(f"Base model saved in {BASE_MODEL_PATH}")
    else:
        print("Base model already prepared.")

if __name__ == "__main__":
    prepare_base_model()