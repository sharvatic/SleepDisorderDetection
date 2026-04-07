"""
Sleep Disorder Classification — Stratified Split + 3D CNN Training
===================================================================
Uses the dataset built by build_dataset.py

SPLIT STRATEGY: 70% train / 15% validation / 15% test
    Stratified by disorder_label — every class keeps its proportions
    across all three splits. With 609 epochs and 5 classes, preserving
    class ratios prevents any split from having zero samples of a class.

    Why NOT 80/10/10:
        With 609 epochs, 10% = ~60 samples total for test.
        Some classes have only 13 epochs — 10% of that = 1 sample.
        15% gives more reliable evaluation metrics per class.

MODEL: Custom 3D CNN
    Input shape: (batch, 3, 30, 32, 32)
        PyTorch convention: (B, C, T, H, W)
        C=3  RGB channels (R=beta, G=alpha, B=delta)
        T=30 temporal slices (one per second)
        H=W=32 spatial grid

    Architecture:
        Block 1: Conv3D(3→32)   → BN → ReLU → MaxPool3D(2,2,2)
        Block 2: Conv3D(32→64)  → BN → ReLU → MaxPool3D(2,2,2)
        Block 3: Conv3D(64→128) → BN → ReLU → AdaptiveAvgPool3D(1)
        Classifier: Linear(128→64) → Dropout(0.5) → Linear(64→n_classes)

    WHY THIS ARCHITECTURE:
        Conv3D slides a 3D kernel across (T, H, W) simultaneously.
        Block 1 learns local spatiotemporal primitives:
            "red patch at central location" = beta burst
            "blue sustained region" = delta wave
        Block 2 combines those into event-level features:
            "red burst at central location lasting 3 slices" = micro arousal
            "green flash bilateral central" = sleep spindle
        Block 3 learns the disorder-level pattern:
            "recurring red bursts every 20s" = PLM
            "sudden red burst during blue background" = NFLE seizure
        AdaptiveAvgPool collapses spatial+temporal dims → fixed size
        regardless of input length — robust to variable epoch counts.
        Dropout(0.5) prevents overfitting on the small dataset.

    WHY NOT RESNET/EFFICIENTNET:
        ResNet3D-18 has 33M parameters — needs 10,000+ samples to train.
        EfficientNet pretrained weights don't transfer from natural images
        to EEG topomaps. This custom model has ~500K parameters which is
        appropriate for 609 training samples.

    ALL OPEN SOURCE:
        PyTorch (BSD license) — free forever
        scikit-learn (BSD license) — free forever
        No API keys, no paid services, no cloud required.

TRAINING DETAILS:
    Loss: CrossEntropyLoss with class weights (fixes imbalance)
    Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
    Scheduler: ReduceLROnPlateau — halves LR when val loss stagnates
    Early stopping: stops if val loss does not improve for 15 epochs
    Batch size: 16 (small dataset — larger batches see too few classes)
    Max epochs: 100

DEPENDENCIES:
    pip install torch torchvision scikit-learn pandas numpy matplotlib
    (no GPU required — CPU training takes ~5 minutes for 100 epochs)
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION — set these to match your dataset paths
# ─────────────────────────────────────────────────────────────

MANIFEST_PATH    = "dataset/metadata/manifest.csv"
LABEL_DIR        = "dataset/labels"
CLASS_WEIGHT_PATH= "dataset/metadata/class_weights.npy"
OUTPUT_DIR       = "training_output"

# split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15   # must sum to 1.0 with above two

# training hyperparameters
BATCH_SIZE      = 16
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
MAX_EPOCHS      = 100
EARLY_STOP_PAT  = 15    # stop if val loss doesn't improve for this many epochs
RANDOM_SEED     = 42

# device — uses GPU if available, falls back to CPU automatically
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# disorder label names — must match build_dataset.py
DISORDER_NAMES = {
    0: "normal",
    1: "nfle",
    2: "rbd",
    3: "plm",
    4: "insomnia",
    5: "narcolepsy",
    6: "sdb",
    7: "bruxism",
}

STAGE_NAMES = {
    0: "Wake", 1: "S1/N1", 2: "S2/N2",
    3: "S3/N3", 4: "S4/N3", 5: "REM",
}


# ─────────────────────────────────────────────────────────────
# STEP 1 — PYTORCH DATASET
# ─────────────────────────────────────────────────────────────

class SleepTensorDataset(Dataset):
    """
    PyTorch Dataset wrapping the .npy tensor files.

    Loads tensors on demand during training — does not load all 609
    tensors into RAM at once. Each __getitem__ call loads one .npy file.

    Tensor transformation:
        Loaded shape:   (30, 32, 32, 3)  — (T, H, W, C)
        Returned shape: (3, 30, 32, 32)  — (C, T, H, W)
        PyTorch Conv3D expects (batch, channels, depth, height, width)
        so we transpose from numpy convention to PyTorch convention.

    Labels returned:
        disorder_label : int  — what the model predicts
        stage_label    : int  — kept for analysis but not used in loss
    """

    def __init__(self, manifest_path, label_dir=None,
                 stage_filter=None, disorder_filter=None):
        """
        Parameters
        ----------
        manifest_path   : path to manifest.csv from build_dataset.py
        label_dir       : path to labels/ directory for slice labels
        stage_filter    : list of stage ints to keep, e.g. [1,2,3,4,5]
                          None = keep all stages including wake
        disorder_filter : list of disorder ints to keep
                          None = keep all disorders
        """
        self.manifest  = pd.read_csv(manifest_path)
        self.label_dir = label_dir

        # apply filters
        if stage_filter is not None:
            self.manifest = self.manifest[
                self.manifest["stage_label"].isin(stage_filter)
            ]
        if disorder_filter is not None:
            self.manifest = self.manifest[
                self.manifest["disorder_label"].isin(disorder_filter)
            ]

        self.manifest = self.manifest.reset_index(drop=True)

        # remap disorder labels to consecutive integers 0..n_classes-1
        # necessary because you may not have all 8 disorders in your subset
        # e.g. if you only have nfle(1), ins(4), nar(5), brux(7), n(0)
        # the model output would be size 8 but only 5 classes are present
        unique_labels   = sorted(self.manifest["disorder_label"].unique())
        self.label_remap = {orig: new for new, orig in enumerate(unique_labels)}
        self.n_classes   = len(unique_labels)
        self.class_names = [DISORDER_NAMES.get(l, str(l)) for l in unique_labels]

        # cache slice labels per patient
        self._slice_cache = {}

        print(f"Dataset: {len(self.manifest)} epochs  |  "
              f"{self.n_classes} classes: {self.class_names}")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        # load tensor: (30, 32, 32, 3) → transpose → (3, 30, 32, 32)
        tensor = np.load(row["tensor_path"]).astype(np.float32)
        tensor = tensor.transpose(3, 0, 1, 2)   # (C, T, H, W)
        tensor = torch.from_numpy(tensor)

        # remap disorder label to consecutive index
        disorder_label = self.label_remap[int(row["disorder_label"])]
        stage_label    = int(row["stage_label"])

        return tensor, disorder_label, stage_label

    def get_labels(self):
        """Return array of all disorder labels — needed for stratified split."""
        return np.array([
            self.label_remap[int(l)]
            for l in self.manifest["disorder_label"]
        ])


# ─────────────────────────────────────────────────────────────
# STEP 2 — STRATIFIED SPLIT
# ─────────────────────────────────────────────────────────────

def stratified_split(dataset, train_ratio=0.70, val_ratio=0.15,
                     random_seed=42):
    """
    Split dataset indices into train / val / test sets.

    STRATIFIED means each split has the same class proportions as the
    full dataset. Without stratification, a random split of 609 samples
    might put all 13 normal epochs into training and none into test,
    making test evaluation meaningless for the normal class.

    Method:
        1. Split full → train+val (70%) and test (15%) stratified by label
        2. Split train+val → train (70% of original = 82% of subset)
           and val (15% of original = 18% of subset) stratified by label

    Returns
    -------
    train_indices, val_indices, test_indices : lists of integer indices
    """
    labels = dataset.get_labels()
    n      = len(labels)

    all_indices = np.arange(n)

    # split off test first
    test_size = TEST_RATIO
    trainval_idx, test_idx = train_test_split(
        all_indices,
        test_size    = test_size,
        stratify     = labels,
        random_state = random_seed,
    )

    # split remaining into train and val
    # val should be val_ratio of the TOTAL, so relative to trainval:
    val_relative = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size    = val_relative,
        stratify     = labels[trainval_idx],
        random_state = random_seed,
    )

    print(f"\nDataset split (stratified by disorder label):")
    print(f"  Train      : {len(train_idx):4d} epochs  "
          f"({100*len(train_idx)/n:.1f}%)")
    print(f"  Validation : {len(val_idx):4d} epochs  "
          f"({100*len(val_idx)/n:.1f}%)")
    print(f"  Test       : {len(test_idx):4d} epochs  "
          f"({100*len(test_idx)/n:.1f}%)")

    # verify class distribution is preserved
    print("\n  Class distribution across splits:")
    print(f"  {'Class':<12} {'Total':>6} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("  " + "─"*42)
    for cls in range(dataset.n_classes):
        name   = dataset.class_names[cls]
        total  = (labels == cls).sum()
        tr     = (labels[train_idx] == cls).sum()
        va     = (labels[val_idx]   == cls).sum()
        te     = (labels[test_idx]  == cls).sum()
        print(f"  {name:<12} {total:>6} {tr:>6} {va:>6} {te:>6}")

    return list(train_idx), list(val_idx), list(test_idx)


# ─────────────────────────────────────────────────────────────
# STEP 3 — 3D CNN MODEL
# ─────────────────────────────────────────────────────────────

class SleepDisorderCNN(nn.Module):
    """
    3D Convolutional Neural Network for sleep disorder classification.

    Input:  (batch, 3, 30, 32, 32)
              batch = number of epochs per batch
              3     = RGB channels (R=beta, G=alpha, B=delta)
              30    = time slices (one per second)
              32×32 = spatial scalp grid

    Architecture detail:

    Block 1: Conv3D(3, 32, kernel=(3,3,3), padding=1)
        Each of the 32 filters learns a basic spatiotemporal primitive.
        kernel_size=3 means it looks at a 3-second × 3×3 pixel region.
        After MaxPool3D(2,2,2): output shape (32, 15, 16, 16)

    Block 2: Conv3D(32, 64, kernel=(3,3,3), padding=1)
        64 filters learn combinations of Block 1 features.
        e.g. "red blob (beta burst) at central location"
        After MaxPool3D(2,2,2): output shape (64, 7, 8, 8)

    Block 3: Conv3D(64, 128, kernel=(3,3,3), padding=1)
        128 filters learn disorder-level temporal patterns.
        e.g. "recurring red bursts with 20s periodicity" = PLM
        After AdaptiveAvgPool3D(1): output shape (128, 1, 1, 1)

    Why AdaptiveAvgPool instead of MaxPool in Block 3:
        MaxPool would reduce (7,8,8) to (3,4,4) — still spatial info left.
        AdaptiveAvgPool(1) collapses everything to a single vector of 128
        values regardless of input size — forces the network to summarise
        the ENTIRE epoch into 128 numbers before classification.

    Classifier:
        Linear(128 → 64) → ReLU → Dropout(0.5) → Linear(64 → n_classes)
        Dropout(0.5) randomly zeros half the neurons during training —
        prevents the network from memorising specific training samples.
    """

    def __init__(self, n_classes):
        super().__init__()

        # ── Block 1 ───────────────────────────────────────────────────────
        self.block1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),   # (32, 15, 16, 16)
        )

        # ── Block 2 ───────────────────────────────────────────────────────
        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),   # (64, 7, 8, 8)
        )

        # ── Block 3 ───────────────────────────────────────────────────────
        self.block3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),                  # (128, 3, 4, 4)
        )

        # ── Block 4 (The New High-Capacity Layer) ─────────────────────────
        self.block4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),                     # Spatial dropout is great for EEG
            nn.AdaptiveAvgPool3d(1),                 # FINALLY collapse to 1x1x1 here
        )

        # ── Classifier ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                            # (256,)
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
            # no softmax here — CrossEntropyLoss applies it internally which calculates probablilty distribution across all clas
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, gamma=2.0):
#         super().__init__()
#         self.weight = weight
#         self.gamma  = gamma

#     def forward(self, inputs, targets):
#         ce   = nn.functional.cross_entropy(
#                    inputs, targets,
#                    weight=self.weight,
#                    reduction="none")
#         pt   = torch.exp(-ce)
#         loss = ((1 - pt) ** self.gamma) * ce
#         return loss.mean()
    
# ─────────────────────────────────────────────────────────────
# STEP 4 — TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Run one full pass through the training data.
    Returns average loss and accuracy for this epoch.
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for tensors, disorder_labels, _ in loader:
        tensors         = tensors.to(device)
        disorder_labels = disorder_labels.to(device)

        optimizer.zero_grad()
        outputs = model(tensors)                          # (B, n_classes)
        loss    = criterion(outputs, disorder_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * tensors.size(0)
        predicted   = outputs.argmax(dim=1)
        correct    += (predicted == disorder_labels).sum().item()
        total      += tensors.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """
    Evaluate model on a data loader without updating weights.
    Returns loss, accuracy, all predictions, all true labels.
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for tensors, disorder_labels, _ in loader:
            tensors         = tensors.to(device)
            disorder_labels = disorder_labels.to(device)

            outputs   = model(tensors)
            loss      = criterion(outputs, disorder_labels)

            total_loss += loss.item() * tensors.size(0)
            predicted   = outputs.argmax(dim=1)
            correct    += (predicted == disorder_labels).sum().item()
            total      += tensors.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(disorder_labels.cpu().numpy())

    return (total_loss / total,
            correct / total,
            np.array(all_preds),
            np.array(all_labels))


# ─────────────────────────────────────────────────────────────
# STEP 5 — RESULTS VISUALISATION
# ─────────────────────────────────────────────────────────────

def plot_training_curves(history, output_dir):
    """Plot train/val loss and accuracy curves across epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train loss",      color="steelblue")
    ax1.plot(epochs, history["val_loss"],   label="Val loss",        color="orange")
    ax1.axvline(history["best_epoch"], color="red", linestyle="--",
                alpha=0.5, label=f"Best epoch ({history['best_epoch']})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train acc", color="steelblue")
    ax2.plot(epochs, history["val_acc"],   label="Val acc",   color="orange")
    ax2.axvline(history["best_epoch"], color="red", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "training_curves.png",
                dpi=150, bbox_inches="tight")
    print(f"  Saved training_curves.png")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, title, output_dir):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()

    fname = title.lower().replace(" ", "_") + ".png"
    plt.savefig(Path(output_dir) / fname, dpi=150, bbox_inches="tight")
    print(f"  Saved {fname}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")

    # ── load dataset ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 — Loading dataset")
    print("="*60)

    full_dataset = SleepTensorDataset(
        manifest_path = MANIFEST_PATH,
        label_dir     = LABEL_DIR,
        stage_filter  = [1, 2, 3, 4, 5],   # exclude wake epochs
    )

    n_classes    = full_dataset.n_classes
    class_names  = full_dataset.class_names

    # ── stratified split ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2 — Stratified split (70/15/15)")
    print("="*60)

    train_idx, val_idx, test_idx = stratified_split(
        full_dataset,
        train_ratio = TRAIN_RATIO,
        val_ratio   = VAL_RATIO,
        random_seed = RANDOM_SEED,
    )

    train_set = Subset(full_dataset, train_idx)
    val_set   = Subset(full_dataset, val_idx)
    test_set  = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0)

    # ── class weights for imbalanced dataset ─────────────────────────────
    # recompute weights from training split only
    # (not from full dataset — test labels should not influence training)
    train_labels = full_dataset.get_labels()[train_idx]
    counts       = np.bincount(train_labels, minlength=n_classes).astype(float)
    counts       = np.maximum(counts, 1)
    weights      = len(train_labels) / (n_classes * counts)
    class_weights_tensor = torch.FloatTensor(weights).to(DEVICE)

    print(f"\n  Class weights (from training split):")
    for i, name in enumerate(class_names):
        print(f"    {name:<12}  count={int(counts[i]):4d}  "
              f"weight={weights[i]:.3f}")

    # ── build model ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3 — Model")
    print("="*60)

    model = SleepDisorderCNN(n_classes=n_classes).to(DEVICE)
    print(f"  Architecture  : SleepDisorderCNN")
    print(f"  Parameters    : {model.count_parameters():,}")
    print(f"  Classes       : {n_classes}  ({class_names})")
    print(f"  Input shape   : (batch, 3, 30, 32, 32)")

    # ── loss, optimiser, scheduler ────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(),
                            lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── training loop ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4 — Training")
    print("="*60)
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Max epochs    : {MAX_EPOCHS}")
    print(f"  Early stop    : {EARLY_STOP_PAT} epochs patience")
    print(f"  Learning rate : {LEARNING_RATE}")
    print()

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "best_epoch": 1,
    }

    best_val_loss  = float("inf")
    best_model_path = Path(OUTPUT_DIR) / "best_model.pt"
    patience_count  = 0
    t_start         = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, DEVICE
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # print progress every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS}  |  "
                  f"train loss={train_loss:.4f}  acc={train_acc:.3f}  |  "
                  f"val loss={val_loss:.4f}  acc={val_acc:.3f}  |  "
                  f"{elapsed:.0f}s elapsed")

        # save best model
        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            history["best_epoch"] = epoch
            torch.save({
                "epoch"        : epoch,
                "model_state"  : model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss"     : val_loss,
                "val_acc"      : val_acc,
                "n_classes"    : n_classes,
                "class_names"  : class_names,
            }, best_model_path)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= EARLY_STOP_PAT:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no val improvement for {EARLY_STOP_PAT} epochs)")
                break

    total_train_time = time.time() - t_start
    print(f"\n  Training complete in {total_train_time/60:.1f} minutes")
    print(f"  Best epoch: {history['best_epoch']}  "
          f"val loss={best_val_loss:.4f}")

    # ── plot training curves ───────────────────────────────────────────────
    plot_training_curves(history, OUTPUT_DIR)

    # ── load best model for evaluation ────────────────────────────────────
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print(f"\n  Loaded best model from epoch {checkpoint['epoch']}")

    # ── validation evaluation ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5 — Validation Evaluation")
    print("="*60)

    _, val_acc_final, val_preds, val_labels = evaluate(
        model, val_loader, criterion, DEVICE
    )
    print(f"\n  Validation accuracy: {val_acc_final:.4f}  "
          f"({val_acc_final*100:.1f}%)")
    print("\n  Classification report (validation):")
    print(classification_report(
        val_labels, val_preds,
        target_names=class_names,
        zero_division=0,
    ))
    plot_confusion_matrix(
        val_labels, val_preds, class_names,
        title="Confusion Matrix — Validation",
        output_dir=OUTPUT_DIR,
    )

    # ── test evaluation ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 6 — Test Evaluation  (final, unbiased performance)")
    print("="*60)
    print("  NOTE: Run this evaluation only ONCE after all training")
    print("        decisions are finalised. Do not use test results")
    print("        to tune hyperparameters — that would invalidate them.")

    _, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, DEVICE
    )
    print(f"\n  Test accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print("\n  Classification report (test):")
    print(classification_report(
        test_labels, test_preds,
        target_names=class_names,
        zero_division=0,
    ))
    plot_confusion_matrix(
        test_labels, test_preds, class_names,
        title="Confusion Matrix — Test",
        output_dir=OUTPUT_DIR,
    )

    # ── save full results ─────────────────────────────────────────────────
    results = {
        "train_epochs"    : len(history["train_loss"]),
        "best_epoch"      : history["best_epoch"],
        "best_val_loss"   : float(best_val_loss),
        "val_accuracy"    : float(val_acc_final),
        "test_accuracy"   : float(test_acc),
        "train_time_min"  : total_train_time / 60,
        "n_classes"       : n_classes,
        "class_names"     : class_names,
        "n_train"         : len(train_idx),
        "n_val"           : len(val_idx),
        "n_test"          : len(test_idx),
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv(Path(OUTPUT_DIR) / "results_summary.csv", index=False)

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"  Model parameters  : {model.count_parameters():,}")
    print(f"  Training epochs   : {results['train_epochs']}")
    print(f"  Best val epoch    : {results['best_epoch']}")
    print(f"  Validation acc    : {results['val_accuracy']:.4f}  "
          f"({results['val_accuracy']*100:.1f}%)")
    print(f"  Test acc          : {results['test_accuracy']:.4f}  "
          f"({results['test_accuracy']*100:.1f}%)")
    print(f"\n  Outputs saved to  : {OUTPUT_DIR}/")
    print(f"    best_model.pt              ← trained model weights")
    print(f"    training_curves.png        ← loss and accuracy curves")
    print(f"    confusion_matrix_*.png     ← per-class performance")
    print(f"    results_summary.csv        ← all metrics in one row")

    return model, results


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, results = main()
