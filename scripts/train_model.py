#!/usr/bin/env python3
"""
Sleep Disorder Classification — Modular Training Script
========================================================
Orchestrates model training, validation, and evaluation using 
cross-entropy loss with class weighting.

Usage:
    python scripts/train_model.py
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from sklearn.metrics import classification_report

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import constants as cfg
from src.models.cnn3d import SleepDisorderCNN
from src.training.dataset import SleepTensorDataset, stratified_split
from src.training.engine import train_one_epoch, evaluate
from src.utils.visualization import plot_training_curves, plot_confusion_matrix
from src.utils.logger import setup_logger

def main():
    logger = setup_logger("training", cfg.TRAIN_OUTPUT_DIR)

    # 1. Setup
    torch.manual_seed(cfg.RANDOM_SEED)
    os.makedirs(cfg.TRAIN_OUTPUT_DIR, exist_ok=True)
    
    # 1.5 NVIDIA Optimizations
    if cfg.DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("[nvidia a100] Enabled cuDNN benchmark for optimized convolutions.")

    logger.info("="*60)
    logger.info("SLEEP DISORDER DETECTION — MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"Device: {cfg.DEVICE}")

    # 2. Load Dataset
    logger.info("[step 1] Loading Dataset...")
    full_dataset = SleepTensorDataset(
        manifest_path = cfg.MANIFEST_PATH,
        stage_filter  = [1, 2, 3, 4, 5]  # Standard: Exclude wake epochs for classification
    )
    
    n_classes = full_dataset.n_classes
    class_names = full_dataset.class_names

    # 3. Stratified Split
    logger.info("[step 2] Splitting Data (Stratified)...")
    train_idx, val_idx, test_idx = stratified_split(
        full_dataset, 
        train_ratio=cfg.TRAIN_RATIO, 
        val_ratio=cfg.VAL_RATIO, 
        random_seed=cfg.RANDOM_SEED
    )

    train_set = Subset(full_dataset, train_idx)
    val_set   = Subset(full_dataset, val_idx)
    test_set  = Subset(full_dataset, test_idx)

    dl_kwargs = {
        "num_workers": cfg.NUM_WORKERS, 
        "pin_memory": cfg.PIN_MEMORY and cfg.DEVICE.type == "cuda"
    }

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, **dl_kwargs)
    val_loader   = DataLoader(val_set,   batch_size=cfg.BATCH_SIZE, shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(test_set,  batch_size=cfg.BATCH_SIZE, shuffle=False, **dl_kwargs)

    # 4. Class Weights (Handling Imbalance)
    # We load computed weights from builder or recompute from train split
    if os.path.exists(cfg.CLASS_WEIGHT_PATH):
        weights_np = np.load(cfg.CLASS_WEIGHT_PATH)
        weights_np = np.nan_to_num(weights_np, posinf=1.0)
        
        # Pull only weights for classes that exist in this run (handles subset edgecases)
        active_weights = [weights_np[orig] for orig in full_dataset.label_remap.keys()]
        
        weights = torch.tensor(active_weights, dtype=torch.float32).to(cfg.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
        logger.info(f"[info] Using balanced class weights.")
    else:
        # Fallback to simple inverse frequency
        labels_train = np.array(full_dataset.get_labels())[train_idx]
        counts = np.bincount(labels_train, minlength=n_classes)
        weights_arr = len(labels_train) / (n_classes * np.maximum(counts, 1))
        weights = torch.tensor(weights_arr, dtype=torch.float32).to(cfg.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)

    # 5. Initialize Model
    logger.info("[step 3] Initializing Model...")
    model = SleepDisorderCNN(n_classes=n_classes).to(cfg.DEVICE)
    logger.info(f"  Architecture: 3D CNN")
    logger.info(f"  Parameters  : {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    # Mixed Precision Scaler for Tensor Cores
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.USE_AMP and cfg.DEVICE.type == "cuda")

    # 6. Training Loop
    logger.info("[step 4] Starting Training...")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "best_epoch": 0}
    best_val_loss = float("inf")
    patience_count = 0
    t_start = time.time()

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE, scaler=scaler)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, cfg.DEVICE, use_amp=cfg.USE_AMP)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:3d}/{cfg.MAX_EPOCHS} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | Val Loss: {va_loss:.4f} Acc: {va_acc:.3f}")

        # Save Best Model
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            history["best_epoch"] = epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": va_acc,
                "class_names": class_names
            }, cfg.BEST_MODEL_PATH)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.EARLY_STOP_PAT:
                logger.info(f"[early stop] No improvement for {cfg.EARLY_STOP_PAT} epochs.")
                break

    logger.info(f"[success] Training complete. Total time: {(time.time()-t_start)/60:.1f}m")

    # 7. Final Evaluation
    logger.info("[step 5] Final Evaluation (Best Model)...")
    checkpoint = torch.load(cfg.BEST_MODEL_PATH, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    # Test Set Performance
    _, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, cfg.DEVICE, use_amp=cfg.USE_AMP)
    
    logger.info("="*60)
    logger.info(f"TEST ACCURACY: {test_acc*100:.2f}%")
    logger.info("="*60)
    logger.info("\nClassification Report:\n" + classification_report(test_labels, test_preds, target_names=class_names, zero_division=0))

    # 8. Visualization
    plot_training_curves(history, cfg.TRAIN_OUTPUT_DIR)
    plot_confusion_matrix(test_labels, test_preds, class_names, "Confusion Matrix - Test", cfg.TRAIN_OUTPUT_DIR)
    
    logger.info(f"[finish] All results saved to {cfg.TRAIN_OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
