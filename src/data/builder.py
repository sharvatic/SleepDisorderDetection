import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from config.constants import (
    EPOCH_SEC, TENSOR_DIR, LABEL_DIR, METADATA_DIR,
    MANIFEST_PATH, GLOBAL_NORMS_PATH, CLASS_WEIGHT_PATH
)
from src.core.signal import load_edf, slice_epochs
from src.core.spatial import epoch_to_tensor
from src.data.parser import parse_st_annotations, assign_epoch_labels, get_disorder_label

def compute_global_norms(edf_paths, sample_patients=30):
    """
    Calculate global amplitude percentiles (1st and 99th) across a subset of patients.
    This ensures all tensors are scaled consistently regardless of individual recording levels.

    Args:
        edf_paths (list): List of paths to EDF files.
        sample_patients (int, optional): Number of patients to sample for normalization. Defaults to 30.

    Returns:
        dict: Mapping of {band_idx: (p1, p99)}.
    """
    print(f"\n[builder] Computing global norms (sampling {sample_patients} patients)...")
    
    # Shuffle and pick samples
    indices = np.random.choice(len(edf_paths), min(sample_patients, len(edf_paths)), replace=False)
    all_psd = []

    for i in indices:
        path = edf_paths[i]
        try:
            data, sfreq, ch_names = load_edf(path)
            # Use 60s epochs just for norm calculation speed
            epochs = slice_epochs(data, sfreq, epoch_sec=60.0)
            
            # Sample a few epochs from this patient
            ep_idx = np.random.choice(len(epochs), min(5, len(epochs)), replace=False)
            for ei in ep_idx:
                # Import here to avoid circular dependencies
                from src.core.signal import epoch_to_band_slices
                psd = epoch_to_band_slices(epochs[ei], sfreq)
                all_psd.append(psd.reshape(-1, 3))
        except Exception as e:
            print(f"  [builder] Skipping {path} for norms: {e}")

    # Stack all sampled PSD values
    X = np.concatenate(all_psd, axis=0) # (Total-Slices, 3)
    
    norms = {}
    for b in range(3):
        p1  = np.percentile(X[:, b], 1)
        p99 = np.percentile(X[:, b], 99)
        norms[b] = (p1, p99)
        print(f"  Band {b}: p1={p1:.2f}, p99={p99:.2f}")

    np.save(GLOBAL_NORMS_PATH, norms)
    return norms

def process_patient(edf_path, global_norms):
    """
    Full pipeline for a single patient: load EDF, parse labels, build tensors, and save files.

    Args:
        edf_path (str): Path to the source .edf file.
        global_norms (dict): Normalization bounds for RBF interpolation.

    Returns:
        list: List of dictionaries, each describing a processed epoch (for manifest).
    """
    path_obj = Path(edf_path)
    patient_id = path_obj.stem
    disorder_label = get_disorder_label(path_obj.name)
    
    if disorder_label == -1:
        return []

    print(f"\n[builder] Processing Patient: {patient_id} (Disorder: {disorder_label})")
    
    # 1. Load Signal and Annotations
    try:
        data, sfreq, ch_names = load_edf(edf_path)
        hypnogram, cap_events = parse_st_annotations(edf_path, default_sfreq=sfreq)
    except Exception as e:
        print(f"  [builder] Error loading {patient_id}: {e}")
        return []

    # 2. Slice into 30s epochs
    epochs = slice_epochs(data, sfreq, epoch_sec=EPOCH_SEC)
    n_epochs = len(epochs)
    
    manifest_rows = []
    all_stage_labels = []
    all_slice_labels = []

    # 3. Process each epoch
    for i in range(n_epochs):
        # Get labels
        stage_label, slice_labels = assign_epoch_labels(i, EPOCH_SEC, hypnogram, cap_events)
        
        # We skip unannotated regions (usually the ends of files)
        if stage_label == -1:
            continue

        # Build spatiotemporal tensor
        tensor, _ = epoch_to_tensor(
            epochs[i], sfreq, ch_names, global_norms=global_norms
        )

        # Save tensor to disk
        tensor_filename = f"{patient_id}_ep{i:04d}.npy"
        tensor_path = os.path.join(TENSOR_DIR, tensor_filename)
        np.save(tensor_path, tensor)

        # Record for manifest
        manifest_rows.append({
            "patient_id": patient_id,
            "epoch_idx": i,
            "tensor_path": tensor_path,
            "disorder_label": disorder_label,
            "stage_label": stage_label
        })
        
        all_stage_labels.append(stage_label)
        all_slice_labels.append(slice_labels)

    # 4. Save combined labels for this patient
    label_path = os.path.join(LABEL_DIR, f"{patient_id}_labels.npz")
    np.savez(label_path, 
             stage_labels=np.array(all_stage_labels), 
             slice_labels=np.array(all_slice_labels))

    return manifest_rows

def build_dataset(data_dir, output_dir=DATASET_ROOT, sample_norms=30):
    """
    Master function to build the entire dataset.

    Args:
        data_dir (str): Directory containing raw EDF and .edf.st files.
        output_dir (str, optional): Where to save processed data. Defaults to DATASET_ROOT.
        sample_norms (int, optional): Number of files to use for normalization. Defaults to 30.
    """
    # Initialize directory structure
    for d in [TENSOR_DIR, LABEL_DIR, METADATA_DIR]:
        os.makedirs(d, exist_ok=True)

    edf_paths = sorted(list(Path(data_dir).glob("*.edf")))
    if not edf_paths:
        raise ValueError(f"No .edf files found in {data_dir}")

    print(f"[builder] Found {len(edf_paths)} patients.")

    # Step 1: Compute or Load Norms
    if os.path.exists(GLOBAL_NORMS_PATH):
        print(f"[builder] Loading existing norms from {GLOBAL_NORMS_PATH}")
        global_norms = np.load(GLOBAL_NORMS_PATH, allow_pickle=True).item()
    else:
        global_norms = compute_global_norms(edf_paths, sample_patients=sample_norms)

    # Step 2: Process all patients
    all_manifest_rows = []
    start_time = time.time()

    for idx, path in enumerate(edf_paths):
        print(f"\n[{idx+1}/{len(edf_paths)}] Progress...")
        rows = process_patient(path, global_norms)
        all_manifest_rows.extend(rows)

    # Step 3: Save Manifest and Compute Weights
    df = pd.DataFrame(all_manifest_rows)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"\n[builder] Saved manifest to {MANIFEST_PATH}")
    
    # Compute inverse frequency class weights
    compute_class_weights(df)

    total_time = (time.time() - start_time) / 3600
    print(f"\n[builder] Completed in {total_time:.2f} hours. Total Epochs: {len(df)}")

def compute_class_weights(df):
    """Calculate balanced class weights to handle dataset imbalance during training."""
    labels = df["disorder_label"].values
    counts = np.bincount(labels)
    n_classes = len(counts)
    total = len(labels)
    
    # Standard balanced weight formula: n_samples / (n_classes * np.bincount(y))
    weights = total / (n_classes * counts.astype(float))
    np.save(CLASS_WEIGHT_PATH, weights)
    print(f"[builder] Saved class weights to {CLASS_WEIGHT_PATH}")
    for i, w in enumerate(weights):
        print(f"  Class {i}: count={counts[i]}, weight={w:.3f}")
