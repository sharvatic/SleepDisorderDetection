# Sleep Disorder Detection — 3D CNN Pipeline

This repository contains a modular, production-ready pipeline for detecting sleep disorders using EEG signal processing and 3D Convolutional Neural Networks.

## Project Structure

```text
├── config/
│   └── constants.py          # EEG bands, electrode maps, and shared config
├── src/
│   ├── core/                 # Core signal & spatial processing logic
│   ├── data/                 # Annotation parsing and dataset construction
│   ├── models/               # 3D CNN architecture definitions
│   ├── training/             # PyTorch dataset and training loops
│   └── utils/                # Visualization and plotting utilities
├── scripts/                  # Unified entry point scripts
│   ├── build_data.py         # Process raw EDFs into spatiotemporal tensors
│   └── train_model.py        # Train and evaluate the neural network
├── notebooks/                # Experimental Jupyter notebooks
└── requirements.txt          # Project dependencies
```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Raw Data**:
   Ensure you have the CAP Sleep Dataset files (`.edf` and `.edf.st`) in a directory (e.g., `raw_data/`).

## Usage

### 1. Build the Dataset
Process raw EEG recordings into 4D tensors (Batch x Channel x Time x Height x Width):
```bash
python scripts/build_data.py --data_dir /path/to/raw_data/
```
The script will:
- Resample signals to 100Hz.
- Apply bandpass filters for Delta, Alpha, and Beta bands.
- Compute power via STFT.
- Perform RBF interpolation to generate 32x32 topomaps.
- Save result to the `dataset/` folder.

### 2. Train the Model
Train the 3D CNN model using the built tensors:
```bash
python scripts/train_model.py
```
This will:
- Perform a stratified split (70/15/15).
- Train using Adam optimizer and Weighted CrossEntropy.
- Save the best model to `training_output/best_model.pt`.
- Generate training curves and confusion matrices.

## Features
- **Spatiotemporal Analysis**: Captures both frequency dynamics and scalp topography.
- **Robust Normalization**: Implements global percentile-based scaling across patients.
- **Bipolar Support**: Automatically resolves bipolar referenced channels (e.g., F3-C3).
- **Class Balancing**: Handles imbalanced datasets using automated weight computation.

---
*Developed for advanced EEG sleep disorder research.*