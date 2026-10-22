
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from model.config import (
    DEVICE,
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    SAVE_DIR,
)

from model.model import PlantDiseaseCNN
from model.dataset import create_dataloaders


def evaluate_model(model, test_loader, device):
    """Evaluate the trained model on the test dataset."""

    model.eval()

    all_labels = []
    all_predictions = []
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)

            total_loss += loss.item() * images.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    average_loss = total_loss / len(test_loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)

    return (
        average_loss,
        accuracy,
        np.array(all_labels),
        np.array(all_predictions),
    )


def main():
    print(f"Device: {DEVICE}")

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {BEST_MODEL_PATH}\n"
            "Train the model first."
        )

    # Load the saved class names
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            class_names = json.load(file)
    else:
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}\n"
            "Make sure training has saved class_names.json."
        )

    # Load datasets
    _, _, test_loader, dataset_class_names = create_dataloaders()

    if class_names != dataset_class_names:
        raise ValueError(
            "Class names do not match the dataset. "
            "Check the class order used during training."
        )

    # Create model
    model = PlantDiseaseCNN(num_classes=len(class_names)).to(DEVICE)

    # Load trained weights
    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded model: {BEST_MODEL_PATH}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Test images: {len(test_loader.dataset)}")

    # Evaluate
    test_loss, accuracy, labels, predictions = evaluate_model(
        model, test_loader, DEVICE
    )

    print("\n========== TEST RESULTS ==========")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            labels,
            predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(class_names))),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(figsize=(20, 20))

    display.plot(
        ax=ax,
        xticks_rotation="vertical",
        colorbar=False,
        values_format="d",
    )

    plt.title("Plant Disease Detection - Confusion Matrix")
    plt.tight_layout()

    confusion_path = SAVE_DIR / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nConfusion matrix saved to: {confusion_path}")

    # Save evaluation metrics
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(accuracy),
        "classification_report": report,
    }

    results_path = SAVE_DIR / "evaluation_results.json"

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"Evaluation results saved to: {results_path}")


if __name__ == "__main__":
    main()
