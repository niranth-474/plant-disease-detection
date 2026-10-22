from pathlib import Path

import torch
import torchvision
from PIL import Image

from backend.services.prediction_service import PredictionService
from model.config import BEST_MODEL_PATH, CLASS_NAMES_PATH


def main() -> None:
    print(f"Python: {__import__('sys').version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    if not BEST_MODEL_PATH.is_file() or not CLASS_NAMES_PATH.is_file():
        print("Inference not checked: trained model files are missing.")
        return

    service = PredictionService.from_files()
    result = service.predict(Image.new("RGB", (224, 224), "green"))
    print(f"Checkpoint: {Path(BEST_MODEL_PATH).resolve()}")
    print(f"Classes: {len(service.class_names)}")
    print(f"Sample prediction: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2f}%")


if __name__ == "__main__":
    main()
