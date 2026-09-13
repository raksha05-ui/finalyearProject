"""
Breast MRI model training — run this INSIDE a Kaggle Notebook.

Setup on Kaggle:
  1. New Notebook -> Add Data -> search "breast-mri-tumor-classification-dataset"
     (abenjelloun) -> Add. Mounted at /kaggle/input/... with NO download to
     your own computer.
  2. Turn on GPU.
  3. Paste this file into a cell and run.
  4. Download the two output files from the Output/Files panel when done:
         mri_model.pt     <- copy next to your main.py
         mri_report.txt   <- evaluation summary

Ported from your existing breast-mri-tumor-classification.ipynb (same
DenseNet201 transfer-learning idea, same two-stage frozen-then-fine-tune
training, same class-weighting for imbalance) but rewritten in PyTorch so
the saved weights load via torch.load()/load_state_dict() like the rest of
your app, instead of a Keras .h5.

Uses the dataset's own real folder-name labels (requirement: don't invent
labels) — whatever class subfolders actually exist under train/ are used
as-is, in sorted order.

Pipeline:
  MRI Dataset -> Preprocessing -> Train/Val/Test (dataset's own split) ->
  DenseNet201 transfer learning -> Evaluation -> Save trained model
"""

import os
import glob
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models import densenet201, DenseNet201_Weights
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ---------------------------------------------------------------------------
# 0) CONFIG
# ---------------------------------------------------------------------------

IMG_SIZE = 224
BATCH_SIZE = 32
FROZEN_EPOCHS = 30
FINE_TUNE_EPOCHS = 15
FROZEN_LR = 1e-4
FINE_TUNE_LR = 1e-5
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def find_mri_dataset_root() -> str:
    """Locate the breast MRI dataset root under /kaggle/input regardless of
    the exact mounted slug — looks for a folder containing train/test/val
    subfolders with class subfolders inside."""
    candidates = glob.glob("/kaggle/input/**/train", recursive=True)
    for c in candidates:
        root = os.path.dirname(c)
        if os.path.isdir(os.path.join(root, "test")) and os.path.isdir(os.path.join(root, "val")):
            return root
    raise FileNotFoundError(
        "Could not find a folder with train/test/val subfolders under "
        "/kaggle/input. Check the exact path in the Input panel and set "
        "DATA_ROOT manually, e.g.:\n"
        "  DATA_ROOT = '/kaggle/input/<slug>/breast_mri_dataset'"
    )


DATA_ROOT = find_mri_dataset_root()
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")
print(f"Detected dataset root: {DATA_ROOT}")

# ---------------------------------------------------------------------------
# 1) DATA — use the dataset's own class-folder names as the real labels
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomAffine(degrees=30, translate=(0.2, 0.2), shear=0.2, scale=(0.8, 1.2)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = ImageFolder(VAL_DIR, transform=eval_transform)
test_dataset = ImageFolder(TEST_DIR, transform=eval_transform)

CLASS_NAMES = train_dataset.classes  # real dataset labels, sorted folder names
print(f"Detected classes (from dataset folders): {CLASS_NAMES}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_dataset.targets),
    y=train_dataset.targets,
)
class_weights = torch.tensor(class_weights_arr, dtype=torch.float32).to(DEVICE)
print(f"Class weights: {dict(zip(CLASS_NAMES, class_weights_arr))}")

# ---------------------------------------------------------------------------
# 2) MODEL — DenseNet201 transfer learning (matches your existing notebook's
#    architecture/head, ported to torchvision/PyTorch)
# ---------------------------------------------------------------------------


class MRIClassifier(nn.Module):
    """Kept in its own class, separate from the ultrasound Classifier and
    the mammography MammographyClassifier — never cross-used between
    modalities per project requirements."""

    def __init__(self, num_classes: int):
        super().__init__()
        weights = DenseNet201_Weights.IMAGENET1K_V1
        backbone = densenet201(weights=weights)
        self.features = backbone.features
        for p in self.features.parameters():
            p.requires_grad = False
        num_backbone_features = backbone.classifier.in_features  # 1920 for densenet201
        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_backbone_features, 1024),
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

    def forward(self, x):
        return self.head(self.features(x))

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True


model = MRIClassifier(num_classes=len(CLASS_NAMES)).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)


def run_training_phase(model, optimizer, epochs, patience, phase_name):
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss_total, val_preds, val_true = 0.0, [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item() * imgs.size(0)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        val_loss = val_loss_total / len(val_loader.dataset)
        val_acc = accuracy_score(val_true, val_preds)
        print(f"[{phase_name}] epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[{phase_name}] Early stopping.")
                break
    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# 3) STAGE 1 — train the head with the backbone frozen
# ---------------------------------------------------------------------------

optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=FROZEN_LR)
model = run_training_phase(model, optimizer, FROZEN_EPOCHS, PATIENCE, "frozen-backbone")

# ---------------------------------------------------------------------------
# 4) STAGE 2 — unfreeze and fine-tune end-to-end at a lower LR
# ---------------------------------------------------------------------------

model.unfreeze_backbone()
optimizer = torch.optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
model = run_training_phase(model, optimizer, FINE_TUNE_EPOCHS, PATIENCE, "fine-tune")

# ---------------------------------------------------------------------------
# 5) EVALUATION on held-out test set
# ---------------------------------------------------------------------------

model.eval()
test_preds, test_true = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        test_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        test_true.extend(labels.numpy())

acc = accuracy_score(test_true, test_preds)
prec = precision_score(test_true, test_preds, average="weighted", zero_division=1)
rec = recall_score(test_true, test_preds, average="weighted")
f1 = f1_score(test_true, test_preds, average="weighted", zero_division=1)
cm = confusion_matrix(test_true, test_preds)
report_text = classification_report(test_true, test_preds, target_names=CLASS_NAMES, zero_division=1)

report = f"""Breast MRI Model Evaluation (held-out test set, n={len(test_true)})
Backbone: DenseNet201 (ImageNet pretrained, transfer learning, 2-stage fine-tune)
Classes (from dataset folder names): {CLASS_NAMES}

Accuracy:  {acc:.4f}
Precision: {prec:.4f}
Recall:    {rec:.4f}
F1 score:  {f1:.4f}
Confusion matrix (rows=true, cols=pred):
{cm}

{report_text}
These numbers reflect this specific model's measured performance on this
test split only — not a validated clinical accuracy claim.
"""
print(report)
with open("mri_report.txt", "w") as f:
    f.write(report)

# ---------------------------------------------------------------------------
# 6) SAVE — state_dict + class names, loads via the same torch.load() +
#    load_state_dict() pattern the rest of your app already uses.
# ---------------------------------------------------------------------------

torch.save({"state_dict": model.state_dict(), "class_names": CLASS_NAMES}, "mri_model.pt")
print("Saved mri_model.pt — download this from the Kaggle Output panel")
print("and place it next to main.py in your project.")
