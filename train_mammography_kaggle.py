"""
Mammography model training — run this INSIDE a Kaggle Notebook.

Setup on Kaggle:
  1. Create a new Notebook.
  2. Add Data -> search "cbis-ddsm-breast-cancer-image-dataset" (awsaf49) -> Add.
     This mounts the dataset at /kaggle/input/cbis-ddsm-breast-cancer-image-dataset
     with NO download to your own computer — Kaggle's own disk holds the images.
  3. Turn on GPU (Settings -> Accelerator -> GPU T4 x2 or P100).
  4. Paste this whole file into a cell (or upload it and %run it) and run.
  5. When finished, download the two output files from the Notebook's
     Output/Files panel:
         mammography_model.pt   <- copy next to your main.py
         mammography_report.txt <- evaluation summary, keep for your records

This mirrors the structure of your existing densenet169-cbis-ddsm.ipynb
(same CSVs, same path-fixing logic, same binary pathology mapping) but is
rewritten in PyTorch (torchvision) instead of TensorFlow/Keras, so the
saved weights load with the same torch.load()/load_state_dict() pattern
your existing main.py already uses for model_cnn.pt.

Pipeline (matches the required stages):
  Mammography Dataset -> Preprocessing -> Train/Val/Test split ->
  DenseNet169 transfer learning -> Evaluation -> Save trained model
"""

import os
import glob
import copy
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import densenet169, DenseNet169_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------------------------------------------------------------------
# 0) CONFIG
# ---------------------------------------------------------------------------

def find_dataset_root() -> str:
    """Locate the CBIS-DDSM dataset root under /kaggle/input regardless of
    the exact slug Kaggle mounted it under (this varies by how the dataset
    was added to the notebook, e.g. a version suffix or renamed folder).
    Looks for a 'csv' subfolder containing dicom_info.csv."""
    candidates = glob.glob("/kaggle/input/*/csv/dicom_info.csv")
    if not candidates:
        # Some mounts nest one level deeper, e.g. /kaggle/input/<slug>/<slug>/csv/...
        candidates = glob.glob("/kaggle/input/*/*/csv/dicom_info.csv")
    if not candidates:
        raise FileNotFoundError(
            "Could not find dicom_info.csv anywhere under /kaggle/input. "
            "Check the exact folder name in the Input panel on the right "
            "of your Kaggle Notebook (click into 'CBIS-DDSM...' -> 'csv') "
            "and set DATA_ROOT manually below if needed, e.g.:\n"
            "  DATA_ROOT = '/kaggle/input/<exact-folder-name-from-sidebar>'"
        )
    csv_dir = os.path.dirname(candidates[0])
    return os.path.dirname(csv_dir)


DATA_ROOT = find_dataset_root()
CSV_DIR = os.path.join(DATA_ROOT, "csv")
JPEG_DIR = os.path.join(DATA_ROOT, "jpeg")
print(f"Detected dataset root: {DATA_ROOT}")

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 12
LR = 1e-4
VAL_FRACTION_OF_TEMP = 0.5  # temp split -> half val, half test
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Real dataset labels only (CBIS-DDSM `pathology` column) — collapsed to
# binary per requirement #11 (no invented labels, just grouping the
# dataset's own callback/non-callback benign variants together).
CLASS_MAPPER = {"MALIGNANT": 1, "BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0}
CLASS_NAMES = {0: "BENIGN", 1: "MALIGNANT"}

print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# 1) LOAD CSVs + FIX IMAGE PATHS (same logic as your existing notebook)
# ---------------------------------------------------------------------------

dicom_data = pd.read_csv(os.path.join(CSV_DIR, "dicom_info.csv"))

full_mammogram_images = dicom_data[dicom_data.SeriesDescription == "full mammogram images"].image_path


def _rewrite_to_jpeg_dir(path: str) -> str:
    # Original DICOM-derived paths look like '.../CBIS-DDSM/jpeg/<series_uid>/<file>'.
    # Keep everything from 'jpeg/' onward and re-root it under our detected JPEG_DIR.
    marker = "jpeg/"
    idx = path.find(marker)
    tail = path[idx + len(marker):] if idx != -1 else os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)
    return os.path.join(JPEG_DIR, tail)


def _series_uid_from_rewritten_path(path: str) -> str:
    # After _rewrite_to_jpeg_dir, path = JPEG_DIR/<series_uid>/<file>.
    # The series UID folder name is the dict key we match against, regardless
    # of how deep JPEG_DIR itself is nested.
    return os.path.basename(os.path.dirname(path))


full_mammogram_images = full_mammogram_images.apply(_rewrite_to_jpeg_dir)

full_mammogram_dict = {}
for p in full_mammogram_images:
    key = _series_uid_from_rewritten_path(p)
    full_mammogram_dict[key] = p

print(f"Indexed {len(full_mammogram_dict)} full-mammogram series folders.")
if full_mammogram_dict:
    sample_key = next(iter(full_mammogram_dict))
    print(f"Example: {sample_key} -> {full_mammogram_dict[sample_key]}  "
          f"(exists={os.path.exists(full_mammogram_dict[sample_key])})")


def fix_full_mammogram_path(df: pd.DataFrame, col: str = "image file path") -> pd.DataFrame:
    df = df.copy()
    matched, unmatched = 0, 0
    for i, img in enumerate(df[col].values):
        # CSV path segments look like: 'Mass-Training_.../<series_uid>/1-1.dcm'
        # The series UID is the second-to-last path segment.
        parts = [p for p in img.split("/") if p]
        img_name = parts[-2] if len(parts) >= 2 else parts[-1]
        if img_name in full_mammogram_dict:
            df.iloc[i, df.columns.get_loc(col)] = full_mammogram_dict[img_name]
            matched += 1
        else:
            unmatched += 1
    print(f"  {col}: matched {matched}, unmatched {unmatched}")
    return df


mass_train = fix_full_mammogram_path(pd.read_csv(os.path.join(CSV_DIR, "mass_case_description_train_set.csv")))
mass_test = fix_full_mammogram_path(pd.read_csv(os.path.join(CSV_DIR, "mass_case_description_test_set.csv")))
calc_train = fix_full_mammogram_path(pd.read_csv(os.path.join(CSV_DIR, "calc_case_description_train_set.csv")))
calc_test = fix_full_mammogram_path(pd.read_csv(os.path.join(CSV_DIR, "calc_case_description_test_set.csv")))

mass_calc = pd.concat([mass_train, mass_test, calc_train, calc_test], axis=0, ignore_index=True)
mass_calc["labels"] = mass_calc["pathology"].map(CLASS_MAPPER)
before_dropna = len(mass_calc)
mass_calc = mass_calc.dropna(subset=["labels", "image file path"])
after_dropna = len(mass_calc)
mass_calc = mass_calc[mass_calc["image file path"].apply(os.path.exists)]
after_exists = len(mass_calc)
mass_calc["labels"] = mass_calc["labels"].astype(int)

print(f"Rows before filtering: {before_dropna}")
print(f"Rows after dropping missing label/path: {after_dropna}")
print(f"Rows after checking file exists on disk: {after_exists}")
if after_exists == 0:
    sample_paths = pd.concat([mass_train, mass_test, calc_train, calc_test])["image file path"].head(3).tolist()
    raise RuntimeError(
        "No mammogram files were found on disk after path-fixing — 0 usable rows.\n"
        f"Sample resolved paths that were checked (should point at real files):\n"
        + "\n".join(sample_paths)
        + f"\n\nActual JPEG_DIR being used: {JPEG_DIR}\n"
        "Open one of these paths' parent folder in the Kaggle file browser "
        "and compare the folder-naming pattern against what's printed above "
        "under 'Indexed N full-mammogram series folders' / 'Example:'."
    )

print(f"Total usable full-mammogram images: {len(mass_calc)}")
print(mass_calc["labels"].value_counts())

# ---------------------------------------------------------------------------
# 2) TRAIN / VAL / TEST SPLIT (stratified, uses real labels only)
# ---------------------------------------------------------------------------

train_df, temp_df = train_test_split(
    mass_calc, test_size=0.3, random_state=RANDOM_STATE, stratify=mass_calc["labels"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=VAL_FRACTION_OF_TEMP, random_state=RANDOM_STATE, stratify=temp_df["labels"]
)
print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

# ---------------------------------------------------------------------------
# 3) DATASET + PREPROCESSING
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class MammogramDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["image file path"].tolist()
        self.labels = df["labels"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


train_loader = DataLoader(MammogramDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(MammogramDataset(val_df, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(MammogramDataset(test_df, eval_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ---------------------------------------------------------------------------
# 4) MODEL — DenseNet169 transfer learning (matches your existing notebook's
#    choice of backbone, ported to torchvision/PyTorch)
# ---------------------------------------------------------------------------


class MammographyClassifier(nn.Module):
    """DenseNet169 backbone (ImageNet-pretrained) + a small classification
    head, fine-tuned for binary benign/malignant mammography classification.

    Kept in its own class, separate from the ultrasound `Classifier` in
    main.py, per requirement #6 (never used to predict ultrasound images
    and vice versa).
    """

    def __init__(self, num_classes: int = 2, freeze_backbone: bool = True):
        super().__init__()
        weights = DenseNet169_Weights.IMAGENET1K_V1
        backbone = densenet169(weights=weights)
        self.features = backbone.features
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False
        num_backbone_features = backbone.classifier.in_features  # 1664 for densenet169
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = MammographyClassifier(num_classes=2, freeze_backbone=True).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=LR
)

# ---------------------------------------------------------------------------
# 5) TRAIN LOOP with early stopping on val accuracy
# ---------------------------------------------------------------------------

best_val_acc = 0.0
best_state = copy.deepcopy(model.state_dict())
patience, bad_epochs = 4, 0

for epoch in range(EPOCHS):
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
    val_preds, val_true = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(labels.numpy())
    val_acc = accuracy_score(val_true, val_preds)
    print(f"Epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("Early stopping.")
            break

model.load_state_dict(best_state)

# ---------------------------------------------------------------------------
# 6) EVALUATION on held-out test set
# ---------------------------------------------------------------------------

model.eval()
test_preds, test_true, test_probs = [], [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_true.extend(labels.numpy())
        test_probs.extend(probs)

acc = accuracy_score(test_true, test_preds)
prec = precision_score(test_true, test_preds)
rec = recall_score(test_true, test_preds)
f1 = f1_score(test_true, test_preds)
cm = confusion_matrix(test_true, test_preds)

report = f"""Mammography Model Evaluation (held-out test set, n={len(test_true)})
Backbone: DenseNet169 (ImageNet pretrained, transfer learning)
Task: binary classification using CBIS-DDSM 'pathology' labels
Classes: 0=BENIGN (incl. BENIGN_WITHOUT_CALLBACK), 1=MALIGNANT

Accuracy:  {acc:.4f}
Precision: {prec:.4f}
Recall:    {rec:.4f}
F1 score:  {f1:.4f}
Confusion matrix (rows=true, cols=pred):
{cm}

These numbers reflect this specific model's measured performance on this
test split only — not a validated clinical accuracy claim.
"""
print(report)
with open("mammography_report.txt", "w") as f:
    f.write(report)

# ---------------------------------------------------------------------------
# 7) SAVE — state_dict only, so it loads with the same torch.load() +
#    load_state_dict() pattern your existing app already uses.
# ---------------------------------------------------------------------------

torch.save(model.state_dict(), "mammography_model.pt")
print("Saved mammography_model.pt — download this from the Kaggle Output panel")
print("and place it next to main.py in your project.")
