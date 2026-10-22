from pathlib import Path

from PIL import Image

from backend.services.prediction_service import (
    PredictionService,
    predict_image,
)


def load_model():
    """Load the trained CNN and class names."""
    service = PredictionService.from_files()
    return service.model, service.class_names


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to a leaf image")
    args = parser.parse_args()

    path = Path(args.image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    model, class_names = load_model()

    with Image.open(path) as image:
        result = predict_image(image, model, class_names)

    print(json.dumps(result, indent=2))