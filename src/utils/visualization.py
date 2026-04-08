import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_training_curves(history, output_dir):
    """
    Generate and save Loss and Accuracy curves for training and validation.

    Args:
        history (dict): Dictionary containing lists of metrics.
        output_dir (str): Directory to save the PNG file.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Loss Curves
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="orange", lw=2)
    if "best_epoch" in history:
        ax1.axvline(history["best_epoch"], color="red", linestyle="--", alpha=0.6, 
                    label=f"Best Model (Ep {history['best_epoch']})")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Accuracy Curves
    ax2.plot(epochs, history["train_acc"], label="Train Acc", color="steelblue", lw=2)
    ax2.plot(epochs, history["val_acc"], label="Val Acc", color="orange", lw=2)
    if "best_epoch" in history:
        ax2.axvline(history["best_epoch"], color="red", linestyle="--", alpha=0.6)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training vs Validation Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [viz] Saved training curves to {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, title, output_dir):
    """
    Generate and save a confusion matrix plot.

    Args:
        y_true (ndarray): True labels.
        y_pred (ndarray): Predicted labels.
        class_names (list): List of strings representing the classes.
        title (str): Title of the plot.
        output_dir (str): Directory to save the PNG file.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    # Normalize filename
    filename = title.lower().replace(" ", "_").replace("—", "-") + ".png"
    save_path = os.path.join(output_dir, filename)
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [viz] Saved confusion matrix to {save_path}")
