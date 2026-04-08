import os
import torch

# ═════════════════════════════════════════════════════════════
# DATA PROCESSING CONSTANTS
# ═════════════════════════════════════════════════════════════

# Sampling and Windowing
RESAMPLE_HZ = 100.0  # Normalized frequency for all EEG data
EPOCH_SEC   = 30.0   # Matches standard sleep stage annotation granularity
SLICE_SEC   = 1.0    # One-second slices to catch micro-arousals
GRID_SIZE   = 32     # Spatial resolution for topomap interpolation

# STFT Settings
STFT_WINDOW_SEC = 4.0  # 4s window = 0.25 Hz resolution (valid for delta)
STFT_HOP_SEC    = 1.0  # 1s hop = matching our slice resolution

# ═════════════════════════════════════════════════════════════
# SIGNAL / ELECTRODE CONFIG
# ═════════════════════════════════════════════════════════════

# Normalized (x, y) coordinates for 10-20 system
ELECTRODE_10_20 = {
    "FP1": (-0.18,  0.85), "FP2": ( 0.18,  0.85),
    "F7":  (-0.72,  0.45), "F8":  ( 0.72,  0.45),
    "F3":  (-0.35,  0.50), "F4":  ( 0.35,  0.50),
    "F1":  (-0.18,  0.52), "F2":  ( 0.18,  0.52),
    "FZ":  ( 0.00,  0.50),
    "T3":  (-0.85,  0.00), "T4":  ( 0.85,  0.00),
    "C3":  (-0.50,  0.00), "C4":  ( 0.50,  0.00),
    "CZ":  ( 0.00,  0.00),
    "P3":  (-0.35, -0.50), "P4":  ( 0.35, -0.50),
    "PZ":  ( 0.00, -0.50),
    "O1":  (-0.30, -0.85), "O2":  ( 0.30, -0.85),
    "A1":  (-1.00,  0.00), "A2":  ( 1.00,  0.00),
    "ROC": ( 0.90,  0.30), "LOC": (-0.90,  0.30),
}

# Standard EEG channels used in research dataset
CAP_EEG_CHANNELS = [
    "F1-F3", "F2-F4", "F3-C3", "F4-C4", "C3-P3", "C4-P4", "P3-O1", "P4-O2", "C4-A1",
]

# EEG Frequency Bands
# Format: (fmin, fmax, rgb_channel_index, label)
# rgb_channel_index: 0=R(beta), 1=G(alpha), 2=B(delta)
BANDS = [
    (0.5,   4.0,  2, "delta"),   # Blue: deep sleep / slow wave
    (8.0,  13.0,  1, "alpha"),   # Green: spindles / relaxed alertness
    (13.0, 30.0,  0, "beta"),    # Red: arousals / logic / tension
]

# ═════════════════════════════════════════════════════════════
# LABEL MAPPINGS
# ═════════════════════════════════════════════════════════════

# Disorder label mapping based on filename prefix
DISORDER_MAP = {
    "n"    : 0,   # Normal / healthy control
    "nfle" : 1,   # Nocturnal Frontal Lobe Epilepsy
    "rbd"  : 2,   # REM Behaviour Disorder
    "plm"  : 3,   # Periodic Leg Movement
    "ins"  : 4,   # Insomnia
    "nar"  : 5,   # Narcolepsy
    "sdb"  : 6,   # Sleep Disordered Breathing
    "brux" : 7,   # Bruxism
}

DISORDER_NAMES = {v: k for k, v in DISORDER_MAP.items()}

STAGE_NAMES = {
    0: "Wake", 1: "S1/N1", 2: "S2/N2", 3: "S3/N3", 4: "S4/N3", 5: "REM",
    7: "MT",  # Movement Artifact
}

CAP_NAMES = {
    0: "none",
    1: "A1",  # synchronized, low arousal
    2: "A2",  # mixed, intermediate arousal
    3: "A3",  # desynchronized, heavy arousal
}

# ═════════════════════════════════════════════════════════════
# PATHS & TRAINING CONFIG
# ═════════════════════════════════════════════════════════════

PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Built Dataset Structure
DATASET_ROOT      = os.path.join(PROJECT_ROOT, "dataset")
TENSOR_DIR        = os.path.join(DATASET_ROOT, "tensors")
LABEL_DIR         = os.path.join(DATASET_ROOT, "labels")
METADATA_DIR      = os.path.join(DATASET_ROOT, "metadata")
MANIFEST_PATH     = os.path.join(METADATA_DIR, "manifest.csv")
GLOBAL_NORMS_PATH = os.path.join(METADATA_DIR, "global_norms.npy")
CLASS_WEIGHT_PATH = os.path.join(METADATA_DIR, "class_weights.npy")

TRAIN_OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "training_output")
BEST_MODEL_PATH   = os.path.join(TRAIN_OUTPUT_DIR, "best_model.pt")

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Training Hyperparameters
RANDOM_SEED     = 42
BATCH_SIZE      = 256            # Massively increased for A100 40GB VRAM
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
MAX_EPOCHS      = 100
EARLY_STOP_PAT  = 15

# NVMe & DataLoader Optimizations
NUM_WORKERS     = 8              # Concurrent background dataloader workers
PIN_MEMORY      = True           # Speeds up host-to-device transfers
USE_AMP         = True           # Use Automatic Mixed Precision for Tensor Cores

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ═════════════════════════════════════════════════════════════
# DATASET DOWNLOAD URLS
# ═════════════════════════════════════════════════════════════

SAMPLE_DATA_URL = "https://drive.google.com/file/d/1Zs3iS1kzqPL6bBr1sIxtJ7QKesipeDRf/view?usp=sharing"
DATA_URL = "https://drive.google.com/file/d/1FUGi2Vd4nGsjK9_-Vk0d9Yl6BoIAXJ6t/view?usp=sharing"