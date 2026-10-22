import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from model.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    DEVICE,
    IMAGE_SIZE,
)
from model.model import PlantDiseaseCNN


PREDICT_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
OUTPUT_WEIGHT_KEY = "classifier.11.weight"


def predict_image(
    image: Image.Image,
    model: torch.nn.Module,
    class_names: list[str],
) -> dict[str, Any]:
    """Apply the training preprocessing and return the top three classes."""
    image_tensor = PREDICT_TRANSFORM(image.convert("RGB")).unsqueeze(0)
    model_device = next(model.parameters()).device
    image_tensor = image_tensor.to(model_device)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    if probabilities.numel() != len(class_names):
        raise ValueError("Model output does not match the configured class names.")

    top_count = min(3, len(class_names))
    top_probabilities, top_indices = torch.topk(probabilities, k=top_count)
    predicted_index = top_indices[0].item()

    return {
        "predicted_class": class_names[predicted_index],
        "confidence": round(probabilities[predicted_index].item() * 100, 2),
        "top_predictions": [
            {
                "class_name": class_names[index.item()],
                "confidence": round(probability.item() * 100, 2),
            }
            for probability, index in zip(top_probabilities, top_indices)
        ],
    }


class PredictionService:
    """Load the trained model once and reuse it for image predictions."""

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: list[str],
    ) -> None:
        self.model = model.eval()
        self.class_names = class_names

    @classmethod
    def from_files(
        cls,
        checkpoint_path: Path = BEST_MODEL_PATH,
        class_names_path: Path = CLASS_NAMES_PATH,
    ) -> "PredictionService":
        if not checkpoint_path.is_file():
            raise FileNotFoundError("Trained model checkpoint is missing.")
        if not class_names_path.is_file():
            raise FileNotFoundError("Model class names file is missing.")

        with class_names_path.open("r", encoding="utf-8") as class_file:
            class_names = json.load(class_file)
        if (
            not isinstance(class_names, list)
            or not class_names
            or not all(isinstance(name, str) and name for name in class_names)
        ):
            raise ValueError("Class names file must contain a non-empty list of names.")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("Model checkpoint has an unsupported format.")

        checkpoint_names = checkpoint.get("class_names")
        if checkpoint_names is not None and checkpoint_names != class_names:
            raise ValueError("Checkpoint and class names file do not match.")
        if checkpoint.get("num_classes") != len(class_names):
            raise ValueError("Checkpoint output count and class names do not match.")

        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Model checkpoint does not contain model weights.")
        output_weights = state_dict.get(OUTPUT_WEIGHT_KEY)
        if not isinstance(output_weights, torch.Tensor):
            raise ValueError("Model checkpoint is missing its classifier output layer.")
        if output_weights.shape[0] != len(class_names):
            raise ValueError("Model output layer and class names do not match.")

        model = PlantDiseaseCNN(num_classes=len(class_names)).to(DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        return cls(model=model, class_names=class_names)

    def predict(self, image: Image.Image) -> dict[str, Any]:
        return predict_image(image, self.model, self.class_names)
