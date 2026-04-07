"""
CAP Sleep Dataset — Complete Build Pipeline
============================================
Self-contained script. No external imports from other project files.

Contains:
    PART A — Signal Processing (from eeg_4d_pipeline)
        constants, electrode positions, bandpass filter,
        STFT power, RBF interpolation, tensor builder

    PART B — Dataset Builder
        annotation parser, label assignment, global norm computation,
        patient processor, manifest builder, dataset loader

Usage:
    1. Set DATA_DIR and OUTPUT_DIR at the bottom
    2. Run:  python build_dataset.py
    3. Entire dataset built to OUTPUT_DIR in ~2 hours for 108 patients
    4. After that, train from manifest.csv + .npy tensors forever

Dependencies:
    pip install mne scipy numpy pandas matplotlib
"""

import numpy as np
import mne
from scipy.signal import butter, sosfiltfilt, stft as scipy_stft
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401


# ═════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════

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

CAP_EEG_CHANNELS = [
    "F1-F3", "F2-F4",
    "F3-C3", "F4-C4",
    "C3-P3", "C4-P4",
    "P3-O1", "P4-O2",
    "C4-A1",
]

# Band definitions — used for both bandpass filter cutoffs and naming
# Each entry: (fmin, fmax, rgb_channel_index, label)
#   rgb_channel_index:  0=R(beta)  1=G(alpha)  2=B(delta)
BANDS = [
    (0.5,   4.0,  2, "delta"),   # B channel — deep sleep
    (8.0,  13.0,  1, "alpha"),   # G channel — spindles
    (13.0, 30.0,  0, "beta"),    # R channel — arousals
]

# STFT window length in seconds
# 4 seconds gives bin spacing of 100/400 = 0.25 Hz
# sufficient to resolve 0.5 Hz delta waves accurately
STFT_WINDOW_SEC = 4.0

# Hop between consecutive STFT windows in seconds
# 1 second hop → one power value per second → matches our slice resolution
STFT_HOP_SEC = 1.0


# ═════════════════════════════════════════════════════════════
# HELPER — electrode position resolver
# UNCHANGED from CWT version
# ═════════════════════════════════════════════════════════════

def resolve_position(ch_name):
    """
    Return normalised (x, y) scalp position.
    Bipolar 'F3-C3' → midpoint of F3 and C3.
    UNCHANGED from CWT version.
    """
    name = ch_name.upper().strip()
    if name in ELECTRODE_10_20:
        return ELECTRODE_10_20[name]
    if "-" in name:
        parts = name.split("-", 1)
        pts = [np.array(ELECTRODE_10_20[p.strip()])
               for p in parts if p.strip() in ELECTRODE_10_20]
        if len(pts) == 2:
            return tuple(((pts[0] + pts[1]) / 2).tolist())
        if len(pts) == 1:
            return tuple(pts[0].tolist())
    print(f"  [warn] Unknown channel '{ch_name}' → (0, 0)")
    return (0.0, 0.0)


# ═════════════════════════════════════════════════════════════
# STEP 1 — LOAD EDF
# ═════════════════════════════════════════════════════════════

def load_edf(edf_path, target_channels=None, resample_hz=100.0):
    """
    Load EDF, pick EEG channels by name, convert V → µV,
    bandpass 0.5–45 Hz, resample.
    UNCHANGED from CWT version.

    Returns
    -------
    data     : (n_channels, n_samples)  float64  µV
    sfreq    : float
    ch_names : list of str
    """
    if target_channels is None:
        target_channels = CAP_EEG_CHANNELS

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    raw_lower = {c.lower(): c for c in raw.ch_names}
    picked = []
    for ch in target_channels:
        key = ch.lower()
        if key in raw_lower:
            picked.append(raw_lower[key])
        else:
            match = next((v for k, v in raw_lower.items() if key in k), None)
            if match:
                picked.append(match)

    if not picked:
        raise ValueError(f"None of {target_channels} found.\n"
                         f"Available: {raw.ch_names}")

    raw.pick_channels(picked)
    raw.filter(0.5, 45.0, fir_design="firwin", verbose=False)
    if abs(raw.info["sfreq"] - resample_hz) > 1.0:
        raw.resample(resample_hz, verbose=False)

    data = raw.get_data() * 1e6   # V → µV

    print(f"  Channels  : {raw.ch_names}")
    print(f"  sfreq     : {raw.info['sfreq']} Hz")
    print(f"  Duration  : {raw.n_times / raw.info['sfreq']:.1f} s")
    print(f"  Amplitude : {data.min():.1f} – {data.max():.1f} µV")

    return data, raw.info["sfreq"], raw.ch_names


# ═════════════════════════════════════════════════════════════
# STEP 2 — SLICE INTO EPOCHS
# UNCHANGED from CWT version
# ═════════════════════════════════════════════════════════════

def slice_epochs(data, sfreq, epoch_sec=60.0):
    """
    Cut continuous signal into non-overlapping epochs.
    UNCHANGED from CWT version.

    Returns
    -------
    epochs : (n_epochs, n_channels, epoch_samples)
    """
    n       = int(epoch_sec * sfreq)
    n_ep    = data.shape[1] // n
    trimmed = data[:, :n_ep * n]
    return trimmed.reshape(data.shape[0], n_ep, n).transpose(1, 0, 2)


# ═════════════════════════════════════════════════════════════
# STEP 3 — BANDPASS + STFT POWER  (REPLACES CWT)
# THIS IS THE ONLY STEP THAT CHANGED
# ═════════════════════════════════════════════════════════════

def make_bandpass_filter(fmin, fmax, sfreq, order=4):
    """
    NEW — did not exist in CWT version.

    Design a Butterworth bandpass filter for one frequency band.

    What this does:
        Creates a filter that passes frequencies between fmin and fmax
        and blocks everything outside that range.
        After applying this filter to the raw EEG signal, the filtered
        signal contains ONLY the frequencies in [fmin, fmax].
        When STFT is then run on this filtered signal, there is no
        power from outside the band — leakage between bands is impossible.

    Why Butterworth:
        Maximally flat frequency response in the passband.
        No ripples — power within the band is measured uniformly.
        Standard choice for EEG preprocessing in published literature.

    Why order=4:
        Higher order = steeper rolloff at the band edges = less leakage
        Order 4 gives -80 dB/decade rolloff — sufficient for EEG bands
        Higher orders cause phase distortion — we use sosfiltfilt
        (zero-phase filtering) to eliminate phase distortion entirely

    Parameters
    ----------
    fmin, fmax : band cutoff frequencies in Hz
    sfreq      : sampling frequency in Hz
    order      : filter order (4 is standard)

    Returns
    -------
    sos : second-order sections array for use with sosfiltfilt
    """
    nyquist = sfreq / 2.0
    low     = fmin / nyquist
    high    = fmax / nyquist

    # clamp to valid range — fmax must be strictly below nyquist
    high    = min(high, 0.999)

    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sos


def stft_band_power(signal, sfreq,
                    window_sec=STFT_WINDOW_SEC,
                    hop_sec=STFT_HOP_SEC):
    """

    Compute per-second power of a signal that has ALREADY been
    bandpass filtered. Because the signal is already band-limited,
    we just need the total power in each time window — we average
    across ALL frequency bins of the STFT output.

    How STFT works step by step:
    ------------------------------------------------------------
    1. Take window_sec seconds of signal (e.g. 4s = 400 samples)
    2. Multiply by a Hann window function
       (tapers edges to zero — reduces spectral leakage further)
    3. Run FFT on the 400 samples
       → 200 complex coefficients (one per frequency bin)
       → |coefficient|² = power at that bin
    4. Slide window forward by hop_sec (1 second = 100 samples)
    5. Repeat from step 1 until end of signal
    6. Stack all FFT outputs → 2D matrix (n_bins × n_windows)
    7. Average all rows (all frequency bins) → 1 scalar per window
       (valid because signal is already bandpass filtered —
        all bins contain only the target band's energy)

    Why Hann window in step 2:
        A rectangular window (no tapering) causes the FFT to see
        an abrupt start and end to the signal chunk. This creates
        ringing in the frequency domain — artificial power at
        frequencies not present in the signal.
        The Hann window smoothly tapers to zero at both ends,
        eliminating this edge artifact.

    Parameters
    ----------
    signal     : (n_samples,)  already bandpass filtered, in µV
    sfreq      : sampling frequency in Hz
    window_sec : STFT window length in seconds (4.0 recommended)
    hop_sec    : hop between windows in seconds (1.0 = one per second)

    Returns
    -------
    power_per_sec : (n_windows,)  mean power in µV² per time window
    """
    n_window = int(window_sec * sfreq)   # e.g. 400 samples
    n_hop    = int(hop_sec    * sfreq)   # e.g. 100 samples

    # scipy.signal.stft returns:
    #   freqs  : frequency axis (not needed — signal already filtered)
    #   times  : time axis (centre of each window)
    #   Zxx    : complex STFT coefficients  shape (n_freqs, n_windows)
    _, _, Zxx = scipy_stft(
        signal,
        fs          = sfreq,
        window      = "hann",       # Hann window — reduces edge artifacts
        nperseg     = n_window,     # window length in samples
        noverlap    = n_window - n_hop,  # overlap = window - hop
        boundary    = None,         # no padding at edges
        padded      = False,
    )

    # |Zxx|² = power at each frequency bin at each time window
    # shape: (n_freqs, n_windows)
    power = np.abs(Zxx) ** 2

    # average across all frequency bins
    # valid because signal is already bandpass filtered
    # every bin contains only the target band's energy
    power_per_sec = power.mean(axis=0)   # (n_windows,)

    return power_per_sec.astype(np.float32)


def epoch_to_band_slices(epoch, sfreq,
                          slice_sec=1.0,
                          window_sec=STFT_WINDOW_SEC):
    """

    PREVIOUS (CWT):
        for each channel:
            run CWT (40 separate wavelet convolutions)
            slice rows by band
            average rows per band per slice
        cost: ~70 seconds per epoch

    NOW (Bandpass-STFT):
        for each band:
            bandpass filter the raw signal → band-specific signal
            run STFT on filtered signal → power per second
        for each channel:
            extract the per-second power from each band's STFT output
        cost: ~2 seconds per epoch

    The output shape is identical to the CWT version:
        (n_slices, n_channels, 3)
        axis-2:  0=delta  1=alpha  2=beta

    So everything downstream (RBF interpolation, tensor building,
    visualisation) is completely unchanged.

    Parameters
    ----------
    epoch      : (n_channels, epoch_samples)  in µV
    sfreq      : sampling frequency in Hz
    slice_sec  : duration of each output slice in seconds
    window_sec : STFT window length (4s recommended for delta accuracy)

    Returns
    -------
    band_psd : (n_slices, n_channels, 3)  float32
               axis-2:  0=delta  1=alpha  2=beta
    """
    n_ch, n_samples = epoch.shape
    n_slices        = int(n_samples / (slice_sec * sfreq))

    band_psd = np.zeros((n_slices, n_ch, 3), dtype=np.float32)

    for band_idx, (fmin, fmax, rgb_ch, label) in enumerate(BANDS):

        # design bandpass filter for this band
        # this creates the filter ONCE per band, reused for all channels
        sos = make_bandpass_filter(fmin, fmax, sfreq)

        for ch in range(n_ch):
            # Step 1: bandpass filter the raw channel signal
            # sosfiltfilt applies the filter forwards AND backwards
            # → zero phase distortion (no time shift in the output)
            filtered = sosfiltfilt(sos, epoch[ch])

            # Step 2: STFT on the filtered signal
            # returns one power value per second (matching slice_sec=1.0)
            power_per_sec = stft_band_power(
                filtered, sfreq,
                window_sec=window_sec,
                hop_sec=slice_sec,
            )

            # trim or pad to exactly n_slices
            # (STFT may return slightly more or fewer windows
            #  depending on signal length and window centering)
            n_out = min(len(power_per_sec), n_slices)
            band_psd[:n_out, ch, band_idx] = power_per_sec[:n_out]

    return band_psd   # (n_slices, n_channels, 3)


# ═════════════════════════════════════════════════════════════
# STEP 4 — RBF INTERPOLATION → RGB FRAME
# ═════════════════════════════════════════════════════════════

def band_psd_to_rgb(band_row, ch_names, grid_size,
                    vmin_delta, vmax_delta,
                    vmin_alpha, vmax_alpha,
                    vmin_beta,  vmax_beta):
    """
    Interpolate one time slice onto (H, W, 3) RGB frame.

    B = RBF interpolation of delta power → normalised → blue channel
    G = RBF interpolation of alpha power → normalised → green channel
    R = RBF interpolation of beta  power → normalised → red channel

    Each band independently interpolated and normalised.
    Electrode-range clipping applied per band.
    """
    xy = np.array([resolve_position(ch) for ch in ch_names])

    lin      = np.linspace(-1.0, 1.0, grid_size)
    gx, gy   = np.meshgrid(lin, lin)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    scalp    = (gx**2 + gy**2) <= 1.0

    rgb = np.zeros((grid_size, grid_size, 3), dtype=np.float32)

    for band_idx, (vmin, vmax) in enumerate([
            (vmin_delta, vmax_delta),
            (vmin_alpha, vmax_alpha),
            (vmin_beta,  vmax_beta),
    ]):
        psd_vals = band_row[:, band_idx]

        interp = RBFInterpolator(xy, psd_vals,
                                 kernel="thin_plate_spline",
                                 smoothing=1e-3)
        values = interp(grid_pts).reshape(grid_size, grid_size)
        values = np.clip(values, psd_vals.min(), psd_vals.max())
        normed = np.clip((values - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)

        rgb_channel      = 2 - band_idx   # 0→B  1→G  2→R
        rgb[:, :, rgb_channel] = normed

    rgb[~scalp, :] = 0.0
    return rgb


# ═════════════════════════════════════════════════════════════
# STEP 5 — BUILD 4D TENSOR FOR ONE EPOCH
# ═════════════════════════════════════════════════════════════

def epoch_to_tensor(epoch, sfreq, ch_names,
                    slice_sec=1.0, grid_size=32,
                    vmin_delta=None, vmax_delta=None,
                    vmin_alpha=None, vmax_alpha=None,
                    vmin_beta=None,  vmax_beta=None):
    """
    UNCHANGED from CWT version except it now calls the STFT-based
    epoch_to_band_slices() instead of the CWT-based one.

    Returns
    -------
    tensor   : (n_slices, grid_size, grid_size, 3)  float32
    band_psd : (n_slices, n_channels, 3)  raw band power
    """
    band_psd = epoch_to_band_slices(epoch, sfreq, slice_sec)

    _vmin_d = band_psd[:, :, 0].min() if vmin_delta is None else vmin_delta
    _vmax_d = band_psd[:, :, 0].max() if vmax_delta is None else vmax_delta
    _vmin_a = band_psd[:, :, 1].min() if vmin_alpha is None else vmin_alpha
    _vmax_a = band_psd[:, :, 1].max() if vmax_alpha is None else vmax_alpha
    _vmin_b = band_psd[:, :, 2].min() if vmin_beta  is None else vmin_beta
    _vmax_b = band_psd[:, :, 2].max() if vmax_beta  is None else vmax_beta

    n_slices = band_psd.shape[0]
    tensor   = np.zeros((n_slices, grid_size, grid_size, 3), dtype=np.float32)

    for s in range(n_slices):
        tensor[s] = band_psd_to_rgb(
            band_psd[s], ch_names, grid_size,
            _vmin_d, _vmax_d,
            _vmin_a, _vmax_a,
            _vmin_b, _vmax_b,
        )

    return tensor, band_psd


"""
CAP Sleep Dataset — Build Training Dataset
==========================================
Reads all EDF + TXT files, builds tensors, parses labels,
saves everything to disk ready for model training.

DIRECTORY STRUCTURE EXPECTED:
    cap_data/
        n1.edf          normal subject 1
        n1.txt          annotations for n1
        n2.edf
        n2.txt
        ...
        nfle1.edf       NFLE patient 1
        nfle1.edf.st
        ...
        rbd1.edf
        rbd1.edf.st
        ...

OUTPUT STRUCTURE:
    dataset/
        tensors/
            n1_epoch_000.npy        shape (30, 32, 32, 3)
            n1_epoch_001.npy
            ...
            nfle1_epoch_000.npy
            ...
        labels/
            n1_labels.npz           contains all label arrays for this patient
            nfle1_labels.npz
            ...
        metadata/
            global_norms.npy        global percentile bounds for R,G,B channels
            manifest.csv            one row per epoch: path, all labels
            class_weights.npy       inverse frequency weights for training

HOW IT IS USED FOR TRAINING:
    1. Run this script ONCE — builds entire dataset overnight
    2. Training script loads manifest.csv
    3. For each batch: load tensor .npy + labels from manifest
    4. Feed to 3D CNN — no reprocessing ever needed

THREE LABEL LEVELS:
    disorder_label  : int 0–7   patient diagnosis (from filename)
    stage_label     : int 0–5   sleep stage per epoch (from .edf.st, every 30s)
    slice_labels    : int array shape (30,)  CAP event per second (0=none,
                                             1=A1, 2=A2, 3=A3)

EPOCH CHOICE — 30 SECONDS:
    Labels in the .edf.st file are given every 30 seconds.
    Using 30-second epochs gives a clean one-to-one mapping:
        one epoch → one sleep stage label → no ambiguity
    Each 30-second epoch produces 30 one-second slices.
    Tensor shape: (30, 32, 32, 3)
    Micro arousals (3–15s AASM) are still fully captured.
"""

import os
import re
import time
import numpy as np
import pandas as pd
from pathlib import Path

# all processing functions are defined below


# ═════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════

# epoch length — 30s matches .txt label granularity exactly
EPOCH_SEC  = 30.0

# one slice per second — catches 3–15s micro arousals
SLICE_SEC  = 1.0

# spatial grid resolution
GRID_SIZE  = 32

# resampling frequency
RESAMPLE_HZ = 100.0

# disorder label mapping — derived from filename prefix
# n1.edf → "n" → 0 (normal)
# nfle3.edf → "nfle" → 1
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

# sleep stage label names for reference
STAGE_NAMES = {
    0: "Wake",
    1: "S1/N1",
    2: "S2/N2",
    3: "S3/N3",
    4: "S4/N3",
    5: "REM",
    7: "MT",    # movement artifact — excluded from training
}

# CAP event label names
CAP_NAMES = {
    0: "none",
    1: "A1",    # synchronized, low arousal impact
    2: "A2",    # mixed, intermediate arousal
    3: "A3",    # desynchronized, heavy arousal — most clinically significant
}


# ═════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════
# STEP 1 — PARSE ANNOTATION FILE  (.edf.st format)
# ═════════════════════════════════════════════════════════════

def parse_st_annotations(edf_path, default_sfreq=512.0):
    """
    Parse a CAP .edf.st WFDB annotation file.

    WHY .edf.st INSTEAD OF .txt:
        The CAP dataset ships two annotation formats for the same data:
            .edf.st  WFDB binary annotation file  ← what we use
            .txt     human-readable REMlogic export
        Both contain identical data: sleep stages + CAP A phase events.
        You do NOT need .txt files.

    WHAT THE .edf.st FILE CONTAINS:
        Each annotation has a sample number and an aux_note string:
            "SLEEP-S2 (30s) C3-A2"   stage 2 sleep
            "MCAP-A3 (5s) C3-A2"     CAP A3 phase, duration 5 seconds
            "SLEEP-REM (30s) C3-A2"  REM sleep
            "SLEEP-W (30s) C3-A2"    wake

    Parameters
    ----------
    edf_path      : path to .edf file — .edf.st must be in same directory
    default_sfreq : fallback native sfreq if header unreadable (512 Hz for CAP)

    Returns
    -------
    hypnogram  : list of (start_sec, stage_int)
    cap_events : list of (start_sec, duration_sec, cap_type_int)
    """
    import wfdb

    edf_path    = str(edf_path)
    record_name = edf_path.replace(".edf", "")   # wfdb record = path minus .edf

    hypnogram  = []
    cap_events = []

    # get native sfreq from EDF header to convert sample numbers to seconds
    try:
        raw_info     = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        native_sfreq = raw_info.info["sfreq"]
    except Exception:
        native_sfreq = default_sfreq
        print(f"  [warn] Using default sfreq {default_sfreq} Hz")

    # read the .edf.st annotation file
    try:
        ann = wfdb.rdann(record_name, "edf.st")
    except FileNotFoundError:
        print(f"  [warn] No .edf.st file found at {record_name}.edf.st")
        return [], []
    except Exception as e:
        print(f"  [warn] Failed to read {record_name}.edf.st: {e}")
        return [], []

    for i, aux in enumerate(ann.aux_note):
        if not aux or not aux.strip():
            continue

        aux     = aux.strip()
        rel_sec = float(ann.sample[i]) / native_sfreq

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

        elif "MCAP-" in aux:
            tm = re.search(r"A(\d)", aux)
            dm = re.search(r"\((\d+)s\)", aux)
            if tm:
                cap_type = int(tm.group(1))
                duration = int(dm.group(1)) if dm else 5
                cap_events.append((rel_sec, duration, cap_type))

    print(f"  Annotations: {len(hypnogram)} stage labels, "
          f"{len(cap_events)} CAP events")
    return hypnogram, cap_events

# ═════════════════════════════════════════════════════════════
# STEP 2 — ASSIGN LABELS TO EPOCHS
# ═════════════════════════════════════════════════════════════

def assign_epoch_labels(epoch_idx, epoch_sec, hypnogram, cap_events):
    """
    Given an epoch index, return all three labels for that epoch.

    STAGE LABEL:
        The hypnogram gives one label per 30 seconds.
        With 30-second epochs, this is a clean one-to-one mapping:
            epoch 0 covers seconds 0–30   → hypnogram[0]
            epoch 1 covers seconds 30–60  → hypnogram[1]
        The stage label is the single integer for this 30-second window.

    SLICE LABELS:
        Each of the 30 one-second slices within the epoch gets its own label.
        Default is 0 (no CAP event).
        If a CAP A phase event overlaps with a slice, that slice gets the
        CAP type (1, 2, or 3).
        This gives second-level resolution for arousal detection.

    Parameters
    ----------
    epoch_idx  : int    which epoch (0-indexed)
    epoch_sec  : float  epoch length in seconds (30.0)
    hypnogram  : list of (start_sec, stage_int)
    cap_events : list of (start_sec, duration_sec, cap_type_int)

    Returns
    -------
    stage_label  : int       sleep stage for this epoch (0–5, 7)
    slice_labels : ndarray   shape (n_slices,)  CAP event per second
    """
    epoch_start = epoch_idx * epoch_sec
    epoch_end   = epoch_start + epoch_sec
    n_slices    = int(epoch_sec / SLICE_SEC)

    # ── stage label ───────────────────────────────────────────────────────
    # find the hypnogram entry whose start time matches this epoch
    # with 30-second epochs this should be exact
    stage_label = -1   # -1 = no annotation found
    for (h_start, h_stage) in hypnogram:
        if abs(h_start - epoch_start) < 1.0:   # within 1 second tolerance
            stage_label = h_stage
            break

    # fallback: use closest hypnogram entry
    if stage_label == -1 and len(hypnogram) > 0:
        closest = min(hypnogram, key=lambda x: abs(x[0] - epoch_start))
        if abs(closest[0] - epoch_start) < epoch_sec:
            stage_label = closest[1]

    # ── slice labels ──────────────────────────────────────────────────────
    # initialise all slices as 0 (no CAP event)
    slice_labels = np.zeros(n_slices, dtype=np.int8)

    for (cap_start, cap_dur, cap_type) in cap_events:
        cap_end = cap_start + cap_dur

        # does this CAP event overlap with our epoch at all?
        if cap_end <= epoch_start or cap_start >= epoch_end:
            continue

        # find which slices this event covers
        # clamp to epoch boundaries
        overlap_start = max(cap_start, epoch_start)
        overlap_end   = min(cap_end,   epoch_end)

        # convert to slice indices within this epoch
        slice_start = int((overlap_start - epoch_start) / SLICE_SEC)
        slice_end   = int(np.ceil((overlap_end - epoch_start) / SLICE_SEC))
        slice_start = max(0, slice_start)
        slice_end   = min(n_slices, slice_end)

        # mark those slices with the CAP type
        slice_labels[slice_start:slice_end] = cap_type

    return stage_label, slice_labels


# ═════════════════════════════════════════════════════════════
# STEP 3 — GET DISORDER LABEL FROM FILENAME
# ═════════════════════════════════════════════════════════════

def get_disorder_label(filename):
    """
    Extract disorder label from EDF filename.

    Filename convention:
        n1.edf     → normal       → 0
        n2.edf     → normal       → 0
        nfle1.edf  → NFLE         → 1
        nfle12.edf → NFLE         → 1
        rbd3.edf   → RBD          → 2
        plm2.edf   → PLM          → 3
        ins1.edf   → Insomnia     → 4
        nar2.edf   → Narcolepsy   → 5
        sdb1.edf   → SDB          → 6
        brux1.edf  → Bruxism      → 7

    The prefix is matched longest-first to avoid "n" matching "nfle".
    """
    stem = Path(filename).stem.lower()   # e.g. "nfle12"

    # sort by length descending so "nfle" is checked before "n"
    for prefix in sorted(DISORDER_MAP.keys(), key=len, reverse=True):
        if stem.startswith(prefix):
            return DISORDER_MAP[prefix]

    print(f"  [warn] Unknown disorder prefix in '{filename}' → label=-1")
    return -1


# ═════════════════════════════════════════════════════════════
# STEP 4 — COMPUTE GLOBAL NORMS ACROSS ALL PATIENTS
# ═════════════════════════════════════════════════════════════

def compute_global_norms(edf_paths, sample_patients=30):
    """
    Compute global percentile normalisation bounds across all patients.

    WHY GLOBAL NORMS:
        Each patient has different absolute EEG amplitude.
        If you normalise per-patient, a quiet patient and a loud patient
        produce tensors on different scales — the model cannot compare them.
        Global norms ensure that the same colour (e.g. red) means the
        same level of arousal activity regardless of which patient it is.

    HOW IT WORKS:
        1. Sample signal variance from a subset of patients
           (computing full STFT on all 108 patients is expensive)
        2. Pool all variance values together
        3. Compute p1 and p99 across the pool
        4. Use a reference epoch to convert variance → real µV² band power
        5. Store three pairs (vmin, vmax) — one per band

    WHY VARIANCE AS PROXY:
        Signal variance is proportional to total broadband power.
        It is computed in microseconds — no STFT needed.
        The scale factor converts variance units to µV² band power units.

    Parameters
    ----------
    edf_paths       : list of all EDF file paths
    sample_patients : how many patients to sample for norm estimation
                      30 out of 108 gives stable percentile estimates
                      more = more accurate but slower

    Returns
    -------
    norms : dict with keys:
            vmin_delta, vmax_delta,
            vmin_alpha, vmax_alpha,
            vmin_beta,  vmax_beta
    """
    print(f"\nComputing global norms from {sample_patients} sampled patients ...")

    # sample evenly across the patient list
    # step     = max(1, len(edf_paths) // sample_patients)
    # sampled  = edf_paths[::step][:sample_patients]

    # 1. Group patients by disorder prefix to ensure diverse representation
    disorder_groups = {}
    for path in edf_paths:
        label = get_disorder_label(path)
        if label not in disorder_groups: 
            disorder_groups[label] = []
        disorder_groups[label].append(path)
    
    # 2. Pick at least 3-4 patients from every disorder group
    # This ensures your 'Global Norm' accounts for the highest arousal power 
    # across ALL pathologies, not just the healthy ones.
    sampled = []
    for label in sorted(disorder_groups.keys()):
        paths = disorder_groups[label]
        # Take up to 4 patients per category to get a balanced max-power estimate
        sampled.extend(paths[:min(len(paths), 4)]) 
    
    # 3. Limit the total sampled to keep computation time reasonable (~30-40 min)
    sampled = sampled[:sample_patients]
        
    all_vars = []   # collect variance values from all sampled epochs

    ref_patient_data = None   # store one patient's full data as STFT reference

    for i, edf_path in enumerate(sampled):
        print(f"  Sampling {i+1}/{len(sampled)}: {Path(edf_path).name}")
        try:
            data, sfreq, ch_names = load_edf(edf_path,
                                              resample_hz=RESAMPLE_HZ)
            epochs = slice_epochs(data, sfreq, EPOCH_SEC)

            # collect per-epoch variance for all epochs of this patient
            patient_vars = np.var(epochs, axis=2).mean(axis=1)
            all_vars.append(patient_vars)

            # store middle patient as STFT reference
            if i == len(sampled) // 2:
                ref_patient_data = (epochs, sfreq, ch_names)

        except Exception as e:
            print(f"  [warn] Failed to load {edf_path}: {e}")
            continue

    if len(all_vars) == 0:
        raise RuntimeError("No patients successfully loaded for norm computation")

    all_vars = np.concatenate(all_vars)   # pool all variance values

    # compute p1 and p99 across all sampled patients and epochs
    p1  = float(np.percentile(all_vars, 1))
    p99 = float(np.percentile(all_vars, 99))

    print(f"  Variance p1  = {p1:.4e}")
    print(f"  Variance p99 = {p99:.4e}")
    print(f"  Variance max = {all_vars.max():.4e}")

    # compute scale factors using STFT on a reference epoch
    # this converts variance units to real µV² band power for each band
    print("  Computing STFT scale factors from reference patient ...")
    epochs_ref, sfreq_ref, ch_names_ref = ref_patient_data
    ref_idx = len(epochs_ref) // 2   # use middle epoch as reference

    _, ref_bp = epoch_to_tensor(
        epochs_ref[ref_idx], sfreq_ref, ch_names_ref,
        slice_sec=SLICE_SEC, grid_size=GRID_SIZE
    )
    ref_var = np.var(epochs_ref[ref_idx])

    # scale factor per band: µV² band power / variance
    scale_d = ref_bp[:, :, 0].mean() / (ref_var + 1e-12)
    scale_a = ref_bp[:, :, 1].mean() / (ref_var + 1e-12)
    scale_b = ref_bp[:, :, 2].mean() / (ref_var + 1e-12)

    norms = {
        "vmin_delta": p1  * scale_d,
        "vmax_delta": p99 * scale_d,
        "vmin_alpha": p1  * scale_a,
        "vmax_alpha": p99 * scale_a,
        "vmin_beta" : p1  * scale_b,
        "vmax_beta" : p99 * scale_b,
    }

    print(f"  Delta norm: {norms['vmin_delta']:.3e} – {norms['vmax_delta']:.3e} µV²")
    print(f"  Alpha norm: {norms['vmin_alpha']:.3e} – {norms['vmax_alpha']:.3e} µV²")
    print(f"  Beta  norm: {norms['vmin_beta']:.3e}  – {norms['vmax_beta']:.3e}  µV²")

    return norms


# ═════════════════════════════════════════════════════════════
# STEP 5 — PROCESS ONE PATIENT
# ═════════════════════════════════════════════════════════════

def process_patient(edf_path, norms, output_dir):
    """
    Process one patient file completely.

    LOAD ONCE PATTERN:
        Load EDF → resample → slice epochs → store in memory
        Then loop through all epochs using already-loaded data.
        The 67-second EDF loading cost is paid ONCE per patient.
        Each epoch then takes ~0.3 seconds.

    For each epoch:
        1. Build tensor (30, 32, 32, 3) using global norms
        2. Get stage label from hypnogram
        3. Get slice labels from CAP events
        4. Get disorder label from filename
        5. Save tensor as .npy
        6. Collect all labels into manifest row

    Parameters
    ----------
    edf_path   : path to .edf file
    edf_path   : path to .edf file (.edf.st must be alongside it)
    norms      : dict of global normalisation bounds (computed once for all patients)
    output_dir : root output directory

    Returns
    -------
    manifest_rows : list of dicts, one per epoch
                    each dict has all labels + tensor path
    """
    patient_id      = Path(edf_path).stem         # e.g. "nfle3"
    disorder_label  = get_disorder_label(edf_path)
    disorder_name   = DISORDER_NAMES.get(disorder_label, "unknown")

    tensor_dir = Path(output_dir) / "tensors"
    label_dir  = Path(output_dir) / "labels"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Patient : {patient_id}  ({disorder_name})")
    print(f"{'='*60}")

    # ── load once ─────────────────────────────────────────────────────────
    t_load = time.time()
    try:
        data, sfreq, ch_names = load_edf(edf_path, resample_hz=RESAMPLE_HZ)
    except Exception as e:
        print(f"  [ERROR] Failed to load EDF: {e}")
        return []

    epochs   = slice_epochs(data, sfreq, EPOCH_SEC)
    n_epochs = epochs.shape[0]
    print(f"  Loaded in {time.time()-t_load:.1f}s  |  {n_epochs} epochs")

    # ── parse annotations from .edf.st ───────────────────────────────────
    hypnogram, cap_events = parse_st_annotations(edf_path)

    # ── process each epoch ────────────────────────────────────────────────
    manifest_rows  = []
    stage_array    = np.full(n_epochs, -1,  dtype=np.int8)
    slice_label_matrix = np.zeros(
        (n_epochs, int(EPOCH_SEC / SLICE_SEC)), dtype=np.int8
    )

    skipped = 0
    t_proc  = time.time()

    for ep_idx in range(n_epochs):

        # get labels for this epoch
        stage_label, slice_labels = assign_epoch_labels(
            ep_idx, EPOCH_SEC, hypnogram, cap_events
        )

        # skip epochs with no annotation (stage_label == -1)
        # and epochs marked as movement artifact (stage_label == 7)
        if stage_label in (-1, 7):
            skipped += 1
            continue

        # build tensor using global norms
        tensor, _ = epoch_to_tensor(
            epochs[ep_idx], sfreq, ch_names,
            slice_sec  = SLICE_SEC,
            grid_size  = GRID_SIZE,
            vmin_delta = norms["vmin_delta"],
            vmax_delta = norms["vmax_delta"],
            vmin_alpha = norms["vmin_alpha"],
            vmax_alpha = norms["vmax_alpha"],
            vmin_beta  = norms["vmin_beta"],
            vmax_beta  = norms["vmax_beta"],
        )

        # save tensor
        tensor_filename = f"{patient_id}_epoch_{ep_idx:04d}.npy"
        tensor_path     = tensor_dir / tensor_filename
        np.save(tensor_path, tensor)

        # store labels
        stage_array[ep_idx]            = stage_label
        slice_label_matrix[ep_idx]     = slice_labels

        # build manifest row — one row per epoch in the CSV
        # this is what the training script loads to find everything
        manifest_rows.append({
            "patient_id"     : patient_id,
            "disorder_label" : disorder_label,
            "disorder_name"  : disorder_name,
            "epoch_idx"      : ep_idx,
            "epoch_start_sec": ep_idx * EPOCH_SEC,
            "stage_label"    : stage_label,
            "stage_name"     : STAGE_NAMES.get(stage_label, "unknown"),
            "has_A1"         : int(1 in slice_labels),
            "has_A2"         : int(2 in slice_labels),
            "has_A3"         : int(3 in slice_labels),
            "n_arousal_sec"  : int((slice_labels > 0).sum()),
            "tensor_path"    : str(tensor_path),
        })

        # progress every 50 epochs
        if (ep_idx + 1) % 50 == 0:
            elapsed  = time.time() - t_proc
            per_ep   = elapsed / (ep_idx + 1 - skipped)
            remaining = per_ep * (n_epochs - ep_idx - 1)
            print(f"  epoch {ep_idx+1:4d}/{n_epochs}  "
                  f"{per_ep:.2f}s/epoch  "
                  f"~{remaining/60:.1f} min remaining")

    # save all labels for this patient in one compressed file
    np.savez(
        label_dir / f"{patient_id}_labels.npz",
        stage_labels       = stage_array,
        slice_label_matrix = slice_label_matrix,
        disorder_label     = np.array([disorder_label]),
    )

    total_time = time.time() - t_proc
    print(f"\n  Done: {len(manifest_rows)} epochs saved  "
          f"({skipped} skipped)  "
          f"total {total_time/60:.1f} min")

    return manifest_rows


# ═════════════════════════════════════════════════════════════
# STEP 6 — COMPUTE CLASS WEIGHTS
# ═════════════════════════════════════════════════════════════

def compute_class_weights(manifest_df):
    """
    Compute inverse frequency weights for disorder classification.

    WHY CLASS WEIGHTS:
        NFLE has ~23,000 epochs. Bruxism has ~1,200.
        Without correction the model predicts NFLE for everything.
        Inverse frequency weighting gives rare classes proportionally
        higher loss so the model is forced to learn them.

    FORMULA:
        weight[c] = total_epochs / (n_classes × count[c])

        This ensures the weighted sum of losses is the same as if
        all classes had equal representation.

    Returns
    -------
    class_weights : ndarray shape (n_classes,)
                    weight_for_class[disorder_label]
    """
    n_classes    = len(DISORDER_MAP)
    counts       = np.zeros(n_classes, dtype=np.float32)
    total        = len(manifest_df)

    for label in range(n_classes):
        counts[label] = (manifest_df["disorder_label"] == label).sum()

    # avoid division by zero for missing classes
    counts = np.maximum(counts, 1)

    weights = total / (n_classes * counts)
    weights = weights / weights.sum() * n_classes   # normalise to sum = n_classes

    print("\nClass weights:")
    for label, name in DISORDER_NAMES.items():
        if label < n_classes:
            print(f"  {name:<8}  count={int(counts[label]):6d}  "
                  f"weight={weights[label]:.3f}")

    return weights


# ═════════════════════════════════════════════════════════════
# MAIN — BUILD ENTIRE DATASET
# ═════════════════════════════════════════════════════════════

def build_dataset(data_dir, output_dir, sample_patients_for_norms=20):
    """
    Full dataset build pipeline.

    WORKFLOW:
        1. Find all EDF+EDF.ST file pairs in data_dir
        2. Compute global norms ONCE from a sample of patients
        3. For each patient:
               load EDF once
               process all epochs
               save tensors + labels
        4. Build manifest CSV — one row per epoch
        5. Save class weights
        6. Print dataset summary

    Parameters
    ----------
    data_dir   : directory containing all .edf and .edf.st files
    output_dir : where to write tensors, labels, manifest, norms
    sample_patients_for_norms : how many patients to use for norm estimation
    """
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)

    t_total = time.time()

    # ── find all EDF+.edf.st pairs ────────────────────────────────────────
    # .edf.st is the annotation file shipped with the CAP dataset
    edf_files = sorted(data_dir.glob("*.edf"))
    pairs     = []
    for edf in edf_files:
        st_file = Path(str(edf) + ".st")
        if st_file.exists():
            pairs.append(str(edf))
        else:
            print(f"  [warn] No .edf.st for {edf.name} — skipping")

    print(f"Found {len(pairs)} EDF files with .edf.st annotations")

    if len(pairs) == 0:
        raise RuntimeError(
            f"No .edf.st files found in {data_dir}\n"
            f"Ensure cap_data contains both *.edf and *.edf.st files."
        )

    # ── compute global norms ONCE ─────────────────────────────────────────
    # paid once, used for every epoch of every patient
    norms_path = output_dir / "metadata" / "global_norms.npy"

    if norms_path.exists():
        # load cached norms if already computed
        norms = np.load(norms_path, allow_pickle=True).item()
        print(f"\nLoaded cached global norms from {norms_path}")
    else:
        norms = compute_global_norms(pairs, sample_patients_for_norms)
        np.save(norms_path, norms)
        print(f"  Saved global norms → {norms_path}")

    # ── process each patient ──────────────────────────────────────────────
    all_manifest_rows = []

    for i, edf_path in enumerate(pairs):
        print(f"\nPatient {i+1}/{len(pairs)}")
        try:
            rows = process_patient(edf_path, norms, output_dir)
            all_manifest_rows.extend(rows)
        except Exception as e:
            print(f"  [ERROR] Failed to process {edf_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ── build manifest CSV ────────────────────────────────────────────────
    manifest_df   = pd.DataFrame(all_manifest_rows)
    manifest_path = output_dir / "metadata" / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nManifest saved → {manifest_path}")
    print(f"Total epochs in dataset: {len(manifest_df)}")

    # ── compute and save class weights ────────────────────────────────────
    class_weights      = compute_class_weights(manifest_df)
    class_weight_path  = output_dir / "metadata" / "class_weights.npy"
    np.save(class_weight_path, class_weights)
    print(f"Class weights saved → {class_weight_path}")

    # ── dataset summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total patients    : {len(pairs)} (with .edf.st annotations)")
    print(f"Total epochs      : {len(manifest_df)}")
    print(f"Tensor shape      : ({int(EPOCH_SEC/SLICE_SEC)}, "
          f"{GRID_SIZE}, {GRID_SIZE}, 3)")
    print(f"Epoch duration    : {EPOCH_SEC}s")
    print(f"Slice resolution  : {SLICE_SEC}s per slice")
    print(f"Total time        : {(time.time()-t_total)/3600:.2f} hours")

    print("\nEpochs per disorder:")
    for label, name in DISORDER_NAMES.items():
        count = (manifest_df["disorder_label"] == label).sum()
        if count > 0:
            print(f"  {name:<8}  {count:6d} epochs")

    print("\nEpochs per sleep stage:")
    for stage, name in STAGE_NAMES.items():
        count = (manifest_df["stage_label"] == stage).sum()
        if count > 0:
            print(f"  {name:<8}  {count:6d} epochs")

    print("\nCAP event distribution:")
    print(f"  Epochs with A1: {manifest_df['has_A1'].sum():6d}")
    print(f"  Epochs with A2: {manifest_df['has_A2'].sum():6d}")
    print(f"  Epochs with A3: {manifest_df['has_A3'].sum():6d}")

    print(f"\nDataset ready at: {output_dir}")
    print("To train: load manifest.csv, use tensor_path to load each sample")

    return manifest_df, norms, class_weights


# ═════════════════════════════════════════════════════════════
# TRAINING LOADER — how the training script uses this dataset
# ═════════════════════════════════════════════════════════════

class CAPDataset:
    """
    Dataset loader for the built dataset.

    Loads tensors from disk on demand during training.
    Each sample returns:
        tensor          : (30, 32, 32, 3)  float32
        disorder_label  : int  0–7
        stage_label     : int  0–5
        slice_labels    : (30,) int array  0–3
                            0 = nothing happening    — background sleep, no arousal event
                            1 = CAP Phase A1         — synchronized EEG event, low arousal impact
                            2 = CAP Phase A2         — mixed event, intermediate arousal
                            3 = CAP Phase A3         — desynchronized event, heavy arousal impact

    Example usage in training:
        dataset    = CAPDataset("dataset/metadata/manifest.csv")
        sample, labels = dataset[0]

    Example usage with PyTorch DataLoader:
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for tensors, disorder_labels, stage_labels, slice_labels in loader:
            loss = model(tensors, disorder_labels)
            loss.backward()
    """

    def __init__(self, manifest_path, label_dir=None,
                 disorder_filter=None, stage_filter=None):
        """
        Parameters
        ----------
        manifest_path   : path to manifest.csv
        label_dir       : path to labels directory (for slice labels)
                          if None, slice labels not loaded
        disorder_filter : list of disorder labels to include
                          e.g. [0, 1] for normal + NFLE only
                          None = include all
        stage_filter    : list of stage labels to include
                          e.g. [2, 3, 4, 5] for sleep-only epochs
                          None = include all
        """
        self.manifest = pd.read_csv(manifest_path)
        self.label_dir = label_dir

        # apply filters
        if disorder_filter is not None:
            self.manifest = self.manifest[
                self.manifest["disorder_label"].isin(disorder_filter)
            ]
        if stage_filter is not None:
            self.manifest = self.manifest[
                self.manifest["stage_label"].isin(stage_filter)
            ]

        self.manifest = self.manifest.reset_index(drop=True)

        # cache slice label matrices per patient for fast access
        self._slice_cache = {}

        print(f"Dataset loaded: {len(self.manifest)} epochs")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        # load tensor from disk
        tensor = np.load(row["tensor_path"]).astype(np.float32)

        # disorder and stage labels — directly from manifest
        disorder_label = int(row["disorder_label"])
        stage_label    = int(row["stage_label"])

        # slice labels — from cached npz file
        slice_labels = np.zeros(int(EPOCH_SEC / SLICE_SEC), dtype=np.int8)
        if self.label_dir is not None:
            patient_id = row["patient_id"]
            if patient_id not in self._slice_cache:
                npz_path = Path(self.label_dir) / f"{patient_id}_labels.npz"
                if npz_path.exists():
                    self._slice_cache[patient_id] = np.load(npz_path)[
                        "slice_label_matrix"
                    ]
            if patient_id in self._slice_cache:
                ep_idx = int(row["epoch_idx"])
                if ep_idx < len(self._slice_cache[patient_id]):
                    slice_labels = self._slice_cache[patient_id][ep_idx]

        return tensor, disorder_label, stage_label, slice_labels

    def get_class_weights(self, weight_path):
        """Load pre-computed class weights for weighted loss."""
        return np.load(weight_path)

    def summary(self):
        """Print dataset statistics."""
        print(f"Total epochs    : {len(self.manifest)}")
        print(f"Disorder distribution:")
        for label, name in DISORDER_NAMES.items():
            count = (self.manifest["disorder_label"] == label).sum()
            if count > 0:
                pct = 100 * count / len(self.manifest)
                print(f"  {name:<8}  {count:6d}  ({pct:.1f}%)")


# ═════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── configure these paths ─────────────────────────────────────────────
    DATA_DIR   = "./cap_data_v1"    # folder containing all .edf and .edf.st files
    OUTPUT_DIR = "./dataset"     # where to write everything

    # ── build the full dataset ────────────────────────────────────────────
    manifest, norms, class_weights = build_dataset(
        data_dir   = DATA_DIR,
        output_dir = OUTPUT_DIR,
        sample_patients_for_norms = 20,  # use 20 patients to estimate norms
    )

    # ── verify the dataset can be loaded ─────────────────────────────────
    print("\nVerifying dataset loader")
    dataset = CAPDataset(
        manifest_path   = f"{OUTPUT_DIR}/metadata/manifest.csv",
        label_dir       = f"{OUTPUT_DIR}/labels",
        stage_filter    = [1, 2, 3, 4, 5],  # exclude wake epochs
    )
    dataset.summary()

    # load one sample to verify
    tensor, disorder, stage, slices = dataset[0]
    print(f"\nSample 0:")
    print(f"  tensor shape    : {tensor.shape}")
    print(f"  disorder_label  : {disorder}  ({DISORDER_NAMES.get(disorder)})")
    print(f"  stage_label     : {stage}  ({STAGE_NAMES.get(stage)})")
    print(f"  slice_labels    : {slices}")
    print(f"  arousal seconds : {(slices > 0).sum()} out of {len(slices)}")
