"""Train the breast MRI classifier expected by mri_module.py.

Run from the project directory:
    python train_mri_local.py

The script uses the dataset already downloaded under Downloads by default and
writes mri_model.pt next to this file when training finishes.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.models import DenseNet201_Weights, densenet201


class MRIClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = densenet201(weights=DenseNet201_Weights.DEFAULT)
        self.features = backbone.features
        num_features = backbone.classifier.in_features
        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, 1024),
            nn.SiLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, images):
        return self.head(self.features(images))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "Downloads" / "archive" / "breast_mri_dataset",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=512,
        help="Use a smaller subset for a CPU-friendly local model.",
    )
    parser.add_argument("--max-val-samples", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("mri_model.pt"))
    args = parser.parse_args()

    if not (args.data_dir / "train").is_dir():
        raise FileNotFoundError(f"Training folder not found: {args.data_dir / 'train'}")

    train_transform = T.Compose([
        T.Resize((128, 128)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_train_data = ImageFolder(args.data_dir / "train", transform=train_transform)
    full_val_data = ImageFolder(args.data_dir / "val", transform=eval_transform)
    train_data = Subset(full_train_data, range(min(args.max_train_samples, len(full_train_data))))
    val_data = Subset(full_val_data, range(min(args.max_val_samples, len(full_val_data))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MRIClassifier(len(full_train_data.classes)).to(device)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.features.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    best_accuracy = -1.0

    print(
        f"Using {device}; classes: {full_train_data.classes}; "
        f"training {len(train_data)} images and validating {len(val_data)} images"
    )
    for epoch in range(args.epochs):
        model.train()
        model.features.eval()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                predictions = model(images.to(device)).argmax(dim=1)
                correct += int((predictions == labels.to(device)).sum())
                total += labels.size(0)
        accuracy = correct / max(total, 1)
        print(f"Epoch {epoch + 1}/{args.epochs}: validation accuracy {accuracy:.3f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {"state_dict": model.cpu().state_dict(), "class_names": full_train_data.classes},
                args.output,
            )
            model.to(device)

    print(f"Saved MRI model to {args.output.resolve()}")


if __name__ == "__main__":
    main()