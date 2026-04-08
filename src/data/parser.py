import re
import warnings
from pathlib import Path
import numpy as np
import mne
import wfdb
from config.constants import DISORDER_MAP, SLICE_SEC

# Suppress expected MNE warnings regarding EDF header discrepancies
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")

def parse_st_annotations(edf_path, default_sfreq=512.0):
    """
    Parse a CAP Sleep Dataset .edf.st binary annotation file.

    This file contains sleep stage labels (every 30s) and CAP event labels (variable duration).

    Args:
        edf_path (str): Path to the .edf file. The .edf.st file must be in the same directory.
        default_sfreq (float, optional): Fallback sampling frequency if EDF header is unreadable. 
                                         Defaults to 512.0.

    Returns:
        tuple: (hypnogram, cap_events)
            - hypnogram: list of (start_sec, stage_int)
            - cap_events: list of (start_sec, duration_sec, cap_type_int)
    """
    edf_path    = str(edf_path)
    record_name = edf_path.replace(".edf", "") 

    hypnogram  = []
    cap_events = []

    # Attempt to get native sfreq from EDF to convert sample indices to time
    try:
        raw_info     = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        native_sfreq = raw_info.info["sfreq"]
    except Exception:
        native_sfreq = default_sfreq
        print(f"  [parser] Warning: Using default sfreq {default_sfreq} Hz for {edf_path}")

    # Read binary annotations
    try:
        ann = wfdb.rdann(record_name, "edf.st")
    except Exception as e:
        print(f"  [parser] Warning: Failed to read .edf.st for {record_name}: {e}")
        return [], []

    for i, aux in enumerate(ann.aux_note):
        if not aux or not aux.strip():
            continue

        aux     = aux.strip()
        rel_sec = float(ann.sample[i]) / native_sfreq

        # Case 1: Sleep Stage Label
        if "SLEEP-" in aux:
            m = re.search(r"SLEEP-([A-Z0-9]+)", aux)
            if m:
                s = m.group(1)
                if s == "W":                  stage = 0
                elif s in ("REM", "R"):       stage = 5
                elif s == "MT":               stage = 7
                else:
                    try:                      stage = int(s.replace("S", ""))
                    except ValueError:        continue
                hypnogram.append((rel_sec, stage))

        # Case 2: CAP Event Label
        elif "MCAP-" in aux:
            tm = re.search(r"A(\d)", aux)
            dm = re.search(r"\((\d+)s\)", aux)
            if tm:
                cap_type = int(tm.group(1))
                duration = int(dm.group(1)) if dm else 5
                cap_events.append((rel_sec, duration, cap_type))

    print(f"  [parser] Parsed {len(hypnogram)} stage labels and {len(cap_events)} CAP events.")
    return hypnogram, cap_events

def assign_epoch_labels(epoch_idx, epoch_sec, hypnogram, cap_events):
    """
    Map global annotations to a specific epoch window.

    Args:
        epoch_idx (int): 0-indexed index of the epoch.
        epoch_sec (float): Duration of each epoch in seconds.
        hypnogram (list): List of (start_sec, stage_int).
        cap_events (list): List of (start_sec, duration, type).

    Returns:
        tuple: (stage_label, slice_labels)
            - stage_label (int): Consensus sleep stage for the 30s window.
            - slice_labels (ndarray): Second-by-second CAP events (shape 30).
    """
    epoch_start = epoch_idx * epoch_sec
    epoch_end   = epoch_start + epoch_sec
    n_slices    = int(epoch_sec / SLICE_SEC)

    # 1. Determine Sleep Stage
    # Look for exact start time match (standard for 30s epochs)
    stage_label = -1
    for (h_start, h_stage) in hypnogram:
        if abs(h_start - epoch_start) < 1.0: 
            stage_label = h_stage
            break

    # Fallback to nearest neighbor if no exact match
    if stage_label == -1 and len(hypnogram) > 0:
        closest = min(hypnogram, key=lambda x: abs(x[0] - epoch_start))
        if abs(closest[0] - epoch_start) < epoch_sec:
            stage_label = closest[1]

    # 2. Determine Per-Second CAP labels
    slice_labels = np.zeros(n_slices, dtype=np.int8)

    for (cap_start, cap_dur, cap_type) in cap_events:
        cap_end = cap_start + cap_dur

        # Skip if event is entirely outside this epoch
        if cap_end <= epoch_start or cap_start >= epoch_end:
            continue

        # Map event overlap to indices
        overlap_start = max(cap_start, epoch_start)
        overlap_end   = min(cap_end,   epoch_end)

        idx_start = int((overlap_start - epoch_start) / SLICE_SEC)
        idx_end   = int(np.ceil((overlap_end - epoch_start) / SLICE_SEC))
        
        idx_start = max(0, idx_start)
        idx_end   = min(n_slices, idx_end)

        slice_labels[idx_start:idx_end] = cap_type

    return stage_label, slice_labels

def get_disorder_label(filename):
    """
    Extract the disorder class index from the EDF filename.
    
    Example: 'nfle1.edf' starts with 'nfle' -> returns 1.

    Args:
        filename (str): The filename or path.

    Returns:
        int: Class index from DISORDER_MAP. Returns -1 if unknown.
    """
    stem = Path(filename).stem.lower()
    
    # Check prefixes in descending order of length to handle overlapping prefixes (e.g., 'n' vs 'nfle')
    sorted_prefixes = sorted(DISORDER_MAP.keys(), key=len, reverse=True)
    for prefix in sorted_prefixes:
        if stem.startswith(prefix):
            return DISORDER_MAP[prefix]

    print(f"  [parser] Warning: Unknown disorder prefix for '{filename}'")
    return -1
