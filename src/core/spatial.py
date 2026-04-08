import numpy as np
from scipy.interpolate import RBFInterpolator
from config.constants import ELECTRODE_10_20, GRID_SIZE
from src.core.signal import epoch_to_band_slices

def resolve_position(ch_name):
    """
    Resolve the 2D scalp position (normalized x, y) for a given channel name.
    Supports bipolar names like 'F3-C3' by calculating the midpoint.

    Args:
        ch_name (str): The name of the channel.

    Returns:
        tuple: (x, y) coordinates. Defaults to (0, 0) if unknown.
    """
    name = ch_name.upper().strip()
    
    # Direct match in 10-20 system
    if name in ELECTRODE_10_20:
        return ELECTRODE_10_20[name]
    
    # Check for bipolar referenced channels (e.g., F3-C3)
    if "-" in name:
        parts = name.split("-", 1)
        pts = [np.array(ELECTRODE_10_20[p.strip()])
               for p in parts if p.strip() in ELECTRODE_10_20]
        
        if len(pts) == 2:
            # Midpoint between the two electrodes
            return tuple(((pts[0] + pts[1]) / 2).tolist())
        if len(pts) == 1:
            # Use the single known electrode
            return tuple(pts[0].tolist())
            
    print(f"  [spatial] Warning: Unknown channel position for '{ch_name}' → defaulting to (0, 0)")
    return (0.0, 0.0)

def band_psd_to_rgb(band_row, ch_names, grid_size=GRID_SIZE,
                    norms=None):
    """
    Interpolate one time-slice of multi-channel PSD values into an RGB topomap frame.
    
    Args:
        band_row (ndarray): (n_channels, 3) matrix of PSD values for [Delta, Alpha, Beta].
        ch_names (list): Names of the channels corresponding to the rows.
        grid_size (int, optional): Spatial resolution of the output image. Defaults to GRID_SIZE.
        norms (dict, optional): Normalization bounds for each band. 
                               Format: {band_idx: (min, max)}. Defaults to local min/max.

    Returns:
        ndarray: RGB topomap frame of shape (grid_size, grid_size, 3).
    """
    xy = np.array([resolve_position(ch) for ch in ch_names])

    # Create interpolation grid
    lin      = np.linspace(-1.0, 1.0, grid_size)
    gx, gy   = np.meshgrid(lin, lin)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    
    # Create mask for pixels inside the scalp (unit circle)
    scalp_mask = (gx**2 + gy**2) <= 1.0

    rgb = np.zeros((grid_size, grid_size, 3), dtype=np.float32)

    # Interpolate each band separately
    for band_idx in range(3):
        psd_vals = band_row[:, band_idx]
        
        # Use Thin Plate Spline kernel for smooth EEG interpolation
        interp = RBFInterpolator(xy, psd_vals,
                                 kernel="thin_plate_spline",
                                 smoothing=1e-3)
        
        values = interp(grid_pts).reshape(grid_size, grid_size)
        
        # Determine normalization bounds
        if norms and band_idx in norms:
            vmin, vmax = norms[band_idx]
        else:
            vmin, vmax = psd_vals.min(), psd_vals.max()

        # Clip and Normalize to [0, 1]
        normed = np.clip((values - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)

        # Mapping: 0=Delta(B), 1=Alpha(G), 2=Beta(R)
        # Note: constants.py BANDS defines band_idx 0=delta, 1=alpha, 2=beta
        # and rgb_channel_index as 2, 1, 0 respectively.
        # We'll use the rgb_channel_index from constants.BANDS if available or hardcode.
        from config.constants import BANDS
        target_rgb_ch = BANDS[band_idx][2]
        rgb[:, :, target_rgb_ch] = normed

    # Apply scalp mask to black out regions outside the head
    rgb[~scalp_mask, :] = 0.0
    return rgb

def epoch_to_tensor(epoch, sfreq, ch_names,
                    slice_sec=1.0, grid_size=GRID_SIZE,
                    global_norms=None):
    """
    Convert a raw EEG epoch into a 4D spatiotemporal tensor.

    Args:
        epoch (ndarray): Raw signal of shape (n_channels, n_samples).
        sfreq (float): Sampling frequency.
        ch_names (list): Names of channels.
        slice_sec (float, optional): Temporal resolution. Defaults to 1.0.
        grid_size (int, optional): Spatial resolution. Defaults to GRID_SIZE.
        global_norms (dict, optional): Percentile-based normalization bounds.

    Returns:
        tuple: (tensor, band_psd)
            - tensor: (n_slices, grid_size, grid_size, 3) float32
            - band_psd: (n_slices, n_channels, 3) raw power values
    """
    # Step 1: Extract spectral power per slice
    band_psd = epoch_to_band_slices(epoch, sfreq, slice_sec)
    
    n_slices = band_psd.shape[0]
    tensor   = np.zeros((n_slices, grid_size, grid_size, 3), dtype=np.float32)

    # Step 2: Generate spatial topomap for each slice
    for s in range(n_slices):
        tensor[s] = band_psd_to_rgb(
            band_psd[s], ch_names, grid_size,
            norms=global_norms
        )

    return tensor, band_psd
