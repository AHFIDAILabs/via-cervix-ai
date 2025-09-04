import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification

from config import DEVICE, TRAINED_MODEL_PATH, CLASS_NAMES

class Predictor:
    def __init__(self, model_path):
        self.model = ViTForImageClassification.from_pretrained(
            "artifacts/base_model" # Load the model structure
        )
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path):
        """Predicts the class of a single image."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class_index = torch.argmax(probabilities, dim=1).item()
            predicted_class = CLASS_NAMES[predicted_class_index]
            confidence = probabilities[0][predicted_class_index].item()

        return predicted_class, confidence

if __name__ == "__main__":
    predictor = Predictor(TRAINED_MODEL_PATH)
    # Example usage:
    predicted_class, confidence = predictor.predict("path/to/your/image.jpg")
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.3f}")