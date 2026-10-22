
import json

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.model import PlantDiseaseCNN
from model.dataset import create_dataloaders
from model.config import (
    BEST_MODEL_PATH as MODEL_PATH,
    CLASS_NAMES_PATH,
    DEVICE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE as PATIENCE,
    LEARNING_RATE,
    SAVE_DIR,
    WEIGHT_DECAY,
)


# -------------------------
# Configuration
# -------------------------

# -------------------------
# Training function
# -------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    use_amp = device.type == "cuda"

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# -------------------------
# Validation function
# -------------------------

def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += batch_size

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# -------------------------
# Main training pipeline
# -------------------------

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    device = DEVICE

    print("=" * 50)
    print("PLANT DISEASE DETECTION - TRAINING")
    print("=" * 50)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Load dataset
    train_loader, val_loader, test_loader, class_names = (
        create_dataloaders()
    )

    num_classes = len(class_names)

    print("Number of classes:", num_classes)
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Test images:", len(test_loader.dataset))

    # Save class names in their correct label order
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=4)

    # Initialize custom CNN from scratch
    model = PlantDiseaseCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}%"
        )
        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )
        print(f"Learning Rate: {current_lr:.6f}")

        # Save the checkpoint with the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_accuracy": best_val_accuracy,
                    "num_classes": num_classes,
                    "class_names": class_names,
                },
                MODEL_PATH
            )

            print("Best model saved!")

        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print("Best epoch:", best_epoch)
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print("Model saved to:", MODEL_PATH)
    print("Class names saved to:", CLASS_NAMES_PATH)


if __name__ == "__main__":
    main()
