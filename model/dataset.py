
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model.config import (
    BATCH_SIZE,
    DATASET_DIR,
    IMAGE_SIZE,
    SEED,
    TRAIN_SPLIT,
    VAL_SPLIT,
)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, eval_transform


def create_datasets():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_DIR}\n"
            "Download and extract PlantVillage into data/plantvillage."
        )

    train_transform, eval_transform = get_transforms()

    base_dataset = datasets.ImageFolder(DATASET_DIR)

    if len(base_dataset.classes) < 2:
        raise ValueError(
            "Expected at least two class folders containing images."
        )

    class_names = base_dataset.classes
    total_size = len(base_dataset)

  
    train_size = int(TRAIN_SPLIT * total_size)
    val_size = int(VAL_SPLIT * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(SEED)

    train_subset, val_subset, test_subset = random_split(
        base_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    train_dataset = datasets.ImageFolder(
        DATASET_DIR, transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        DATASET_DIR, transform=eval_transform
    )
    test_dataset = datasets.ImageFolder(
        DATASET_DIR, transform=eval_transform
    )

    # Keep the same indices in each split.
    train_dataset = torch.utils.data.Subset(
        train_dataset, train_subset.indices
    )
    val_dataset = torch.utils.data.Subset(
        val_dataset, val_subset.indices
    )
    test_dataset = torch.utils.data.Subset(
        test_dataset, test_subset.indices
    )

    return train_dataset, val_dataset, test_dataset, class_names


def create_dataloaders():
    train_dataset, val_dataset, test_dataset, class_names = (
        create_datasets()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = (
        create_dataloaders()
    )

    images, labels = next(iter(train_loader))

    print("Classes:", len(class_names))
    print("Class names:", class_names)
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Test images:", len(test_loader.dataset))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Labels in first batch:", labels[:10].tolist())
