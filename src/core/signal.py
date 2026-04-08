import numpy as np
import mne
from scipy.signal import butter, sosfiltfilt, stft as scipy_stft
from config.constants import (
    CAP_EEG_CHANNELS, 
    BANDS, 
    STFT_WINDOW_SEC, 
    STFT_HOP_SEC
)

def load_edf(edf_path, target_channels=None, resample_hz=100.0):
    """
    Load an EDF file, pick target EEG channels, apply a bandpass filter, and resample.

    Args:
        edf_path (str): Path to the .edf file.
        target_channels (list, optional): List of channel names to keep. Defaults to CAP_EEG_CHANNELS.
        resample_hz (float, optional): Target sampling frequency in Hz. Defaults to 100.0.

    Returns:
        tuple: (data, sfreq, ch_names)
            - data: numpy array of shape (n_channels, n_samples) in µV.
            - sfreq: Actual sampling frequency after resampling.
            - ch_names: List of strings containing selected channel names.
    """
    if target_channels is None:
        target_channels = CAP_EEG_CHANNELS

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Normalize channel names to lowercase for robust matching
    raw_lower = {c.lower(): c for c in raw.ch_names}
    picked = []
    for ch in target_channels:
        key = ch.lower()
        if key in raw_lower:
            picked.append(raw_lower[key])
        else:
            # Flexible matching if "C3-A2" is named "EEG C3-A2"
            match = next((v for k, v in raw_lower.items() if key in k), None)
            if match:
                picked.append(match)

    if not picked:
        raise ValueError(f"None of {target_channels} found in {edf_path}.\n"
                         f"Available channels: {raw.ch_names}")

    raw.pick_channels(picked)
    # Standard EEG bandpass to remove DC drift and high-freq noise before STFT
    raw.filter(0.5, 45.0, fir_design="firwin", verbose=False)
    
    if abs(raw.info["sfreq"] - resample_hz) > 1.0:
        raw.resample(resample_hz, verbose=False)

    data = raw.get_data() * 1e6   # Convert Volts to Microvolts (µV)

    print(f"  [signal] Loaded {len(picked)} channels. Duration: {raw.n_times / raw.info['sfreq']:.1f}s")
    return data, raw.info["sfreq"], raw.ch_names

def slice_epochs(data, sfreq, epoch_sec=60.0):
    """
    Split continuous EEG signal into non-overlapping segments (epochs).

    Args:
        data (ndarray): Input signal of shape (n_channels, n_samples).
        sfreq (float): Sampling frequency.
        epoch_sec (float, optional): Duration of each epoch in seconds. Defaults to 60.0.

    Returns:
        ndarray: Epoched data of shape (n_epochs, n_channels, samples_per_epoch).
    """
    n_samples_per_epoch = int(epoch_sec * sfreq)
    n_epochs = data.shape[1] // n_samples_per_epoch
    
    # Trim excess samples at the end
    trimmed = data[:, :n_epochs * n_samples_per_epoch]
    
    # Reshape and transpose to (n_epochs, n_channels, samples)
    return trimmed.reshape(data.shape[0], n_epochs, n_samples_per_epoch).transpose(1, 0, 2)

def make_bandpass_filter(fmin, fmax, sfreq, order=4):
    """
    Design a Butterworth bandpass filter in Second-Order Sections (SOS) format for stability.

    Args:
        fmin (float): Low cutoff frequency in Hz.
        fmax (float): High cutoff frequency in Hz.
        sfreq (float): Sampling frequency in Hz.
        order (int, optional): Order of the filter. Defaults to 4.

    Returns:
        ndarray: SOS coefficients for use with sosfiltfilt.
    """
    nyquist = sfreq / 2.0
    low     = fmin / nyquist
    high    = fmax / nyquist

    # Clamp high frequency to slightly below Nyquist to avoid errors
    high    = min(high, 0.999)

    return butter(order, [low, high], btype="bandpass", output="sos")

def stft_band_power(signal, sfreq, window_sec=STFT_WINDOW_SEC, hop_sec=STFT_HOP_SEC):
    """
    Compute the average power in a specific frequency band using STFT.
    Expects the signal to have been already filtered to the target band.

    Args:
        signal (ndarray): 1D signal of shape (n_samples,).
        sfreq (float): Sampling frequency.
        window_sec (float, optional): Length of the STFT window. Defaults to 4.0.
        hop_sec (float, optional): Hop between windows. Defaults to 1.0.

    Returns:
        ndarray: 1D array of power values per second.
    """
    n_window = int(window_sec * sfreq)
    n_hop    = int(hop_sec    * sfreq)

    # Compute Short-Time Fourier Transform
    _, _, Zxx = scipy_stft(
        signal,
        fs          = sfreq,
        window      = "hann",         # Taper boundaries to reduce leakage
        nperseg     = n_window,
        noverlap    = n_window - n_hop,
        boundary    = None,
        padded      = False,
    )

    # Average power across all frequency bins for the filtered band
    power_per_sec = (np.abs(Zxx) ** 2).mean(axis=0)
    return power_per_sec.astype(np.float32)

def epoch_to_band_slices(epoch, sfreq, slice_sec=1.0, window_sec=STFT_WINDOW_SEC):
    """
    Transform a raw EEG epoch into a 3D volume of band power (Time x Channels x Bands).

    Args:
        epoch (ndarray): Signal of shape (n_channels, n_samples).
        sfreq (float): Sampling frequency.
        slice_sec (float, optional): Resolution of the output slices. Defaults to 1.0.
        window_sec (float, optional): Resolution of the STFT window. Defaults to 4.0.

    Returns:
        ndarray: Band power feature volume of shape (n_slices, n_channels, 3).
                 Axis 2 corresponds to [Delta, Alpha, Beta] respectively.
    """
    n_ch, n_samples = epoch.shape
    n_slices        = int(n_samples / (slice_sec * sfreq))

    # Initialize output: (Time, Channels, 3 Bands)
    band_psd = np.zeros((n_slices, n_ch, 3), dtype=np.float32)

    for band_idx, (fmin, fmax, rgb_ch, label) in enumerate(BANDS):
        # Design filter once per band
        sos = make_bandpass_filter(fmin, fmax, sfreq)

        for ch in range(n_ch):
            # Apply zero-phase forward-backward filter
            filtered = sosfiltfilt(sos, epoch[ch])

            # Extract power per second
            power_per_sec = stft_band_power(
                filtered, sfreq,
                window_sec=window_sec,
                hop_sec=slice_sec,
            )

            # Ensure output fits exact target slice count (trim/pad slightly)
            n_out = min(len(power_per_sec), n_slices)
            band_psd[:n_out, ch, band_idx] = power_per_sec[:n_out]

    return band_psd
