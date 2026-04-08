import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from config.constants import DISORDER_NAMES, MANIFEST_PATH, TEST_RATIO

class SleepTensorDataset(Dataset):
    """
    On-demand data loader for sleep disorder tensors.
    Reads .npy files from disk during training to conserve memory.
    """

    def __init__(self, manifest_path=MANIFEST_PATH, stage_filter=None, disorder_filter=None):
        """
        Args:
            manifest_path (str): Path to the CSV manifest.
            stage_filter (list, optional): List of stage IDs to keep. E.g., [1,2,3,4,5] to exclude wake.
            disorder_filter (list, optional): List of disorder IDs to keep.
        """
        self.manifest = pd.read_csv(manifest_path)

        # Apply data filtering
        if stage_filter is not None:
            self.manifest = self.manifest[self.manifest["stage_label"].isin(stage_filter)]
        if disorder_filter is not None:
            self.manifest = self.manifest[self.manifest["disorder_label"].isin(disorder_filter)]

        self.manifest = self.manifest.reset_index(drop=True)

        # Map non-consecutive labels to 0..n-1 indices
        unique_labels = sorted(self.manifest["disorder_label"].unique())
        self.label_remap = {orig: new for new, orig in enumerate(unique_labels)}
        self.n_classes   = len(unique_labels)
        self.class_names = [DISORDER_NAMES.get(l, str(l)) for l in unique_labels]

        print(f"  [dataset] Initialized with {len(self.manifest)} epochs and {self.n_classes} classes.")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        # Load Tensor: Disk shape (30, 32, 32, 3) [T, H, W, C]
        # PyTorch Conv3D expects: (Channel, Depth, Height, Width) [C, T, H, W]
        tensor = np.load(row["tensor_path"]).astype(np.float32)
        tensor = tensor.transpose(3, 0, 1, 2) 
        tensor = torch.from_numpy(tensor)

        # Label processing
        disorder_label = self.label_remap[int(row["disorder_label"])]
        stage_label    = int(row["stage_label"])

        return tensor, disorder_label, stage_label

    def get_labels(self):
        """Returns remapped disorder labels for all samples in the dataset."""
        return np.array([self.label_remap[int(l)] for l in self.manifest["disorder_label"]])


def stratified_split(dataset, train_ratio=0.70, val_ratio=0.15, random_seed=42):
    """
    Split dataset indices into Train, Val, and Test sets while preserving class distribution.

    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """
    labels = dataset.get_labels()
    indices = np.arange(len(labels))

    try:
        # 1. Try Stratified Split
        trainval_idx, test_idx = train_test_split(
            indices,
            test_size    = TEST_RATIO,
            stratify     = labels,
            random_state = random_seed
        )

        val_relative = val_ratio / (train_ratio + val_ratio)
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size    = val_relative,
            stratify     = labels[trainval_idx],
            random_state = random_seed
        )
    except ValueError as e:
        print(f"  [split] ⚠️ Stratification failed (likely rare classes in small sample set). Falling back to random split.")
        
        # 2. Fallback to Random Split
        trainval_idx, test_idx = train_test_split(
            indices,
            test_size    = TEST_RATIO,
            random_state = random_seed
        )

        val_relative = val_ratio / (train_ratio + val_ratio)
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size    = val_relative,
            random_state = random_seed
        )

    print(f"\n  [split] Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    return list(train_idx), list(val_idx), list(test_idx)
