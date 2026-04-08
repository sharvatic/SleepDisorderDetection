# 🧠 Sleep Disorder Detection (3D CNN Pipeline)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![GPU Support](https://img.shields.io/badge/GPU-A100_Optimized-76B900.svg?logo=nvidia)

This repository contains a modular, production-ready pipeline for detecting sleep disorders. It translates classical multi-channel EEG readings into **4D Spatiotemporal Tensors** and trains a custom 3D Convolutional Neural Network to classify sleeping epochs into distinct clinical disorder categories.

---

## 🚀 Cloud / GPU Deployment (A100 Optimized)

This pipeline has been rewritten and aggressively optimized for execution on **NVIDIA A100 40GB GPUs**. The training script seamlessly leverages Tensor Cores via **Automatic Mixed Precision (AMP)** and saturates GPU throughput using highly threaded memory-pinned DataLoaders (`batch_size=256`, `num_workers=8`, `pin_memory=True`).

If you are cloning this repository into a fresh Cloud GPU instance (e.g. Google Colab, AWS EC2, or Lambda Labs), you can bypass manual setup entirely using our 1-click bootstrap script:

```bash
# Ensure the script is explicitly executable
chmod +x setup_server.sh

# Run the automated deployment script
./setup_server.sh
```

**What `setup_server.sh` automatically does:**
1. Upgrades `pip` and installs all PyTorch & signal processing dependencies dynamically.
2. Evaluates the Google Drive URLs securely using `gdown` to download the *Full Clinical CAP Dataset*.
3. Extracts and organizes the raw binary data into `raw_data/actual/` for immediate pipeline training.

---

## 💻 Local Testing & Setup

If you wish to test or debug the pipeline locally (such as on an Apple Silicon/MPS Mac) where an A100 isn't available, follow these steps to avoid out-of-memory errors:

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Sample Dataset**:
   We provide a tiny, curated 5-patient test dataset for fast local debugging.
   ```bash
   python scripts/download_data.py --type sample
   ```

---

## ⚙️ Running the Pipeline

You have two main options for running the end-to-end pipeline:

### Option 1: The Execution Hub (Jupyter Notebook) `[Recommended]`
Open `notebooks/pipeline_run/execution_hub.ipynb`. 
This interactive notebook acts as a master dashboard. Simply modify the `DATA_DIR` variable to point to your raw data path and **Run All** cells. It will safely stream execution outputs to the terminal, and inline matplotlib graphs natively inside your browser.

### Option 2: Headless Scripts (Terminal)
If you prefer running headless commands (e.g., inside an SSH `tmux` shell on your server):

1. **Build the Spatiotemporal Tensors**
   ```bash
   python scripts/build_data.py --data_dir raw_data/actual
   ```
   *This resamples signals to 100Hz, applies exact STFT band filters (Delta, Alpha, Beta), creates Topomaps via Spatial RBF interpolation, saves global scaling norms, and writes the chunked 30s `.npy` tensors to disk.*

2. **Train the 3D CNN**
   ```bash
   python scripts/train_model.py
   ```
   *Dynamically maps available classes, weights imbalanced disorders inverse to their frequency, performs an exact stratified `70/15/15` split, and runs CNN training via cross-entropy loss with early stopping. Time-stamped logs and visualizations are pushed to `training_output/`.*

---

## 📁 Repository Structure

```text
├── config/
│   └── constants.py          # Master config (GPU hardware flags, hyperparams, 10-20 system map)
├── notebooks/
│   ├── pipeline_run/         # The Execution Hub (Interactive Master Dashboard)
│   └── data_processing_run/  # Exploratory and experimental dataset parsing notebooks
├── scripts/
│   ├── download_data.py      # Automated Google Drive dataset scraper & unzipper
│   ├── build_data.py         # Extracts .edf to multi-dimensional tensors
│   └── train_model.py        # Scalable PyTorch training loop
├── src/
│   ├── core/                 # EEG filtering, signal processing, STFT spatial mapping
│   ├── data/                 # EDF+ CAP binary annotation parser, dataset manifestation
│   ├── models/               # PyTorch 3D CNN architect classes
│   ├── training/             # SKLearn Split strategies, Native Dataset loader, AMP tools
│   └── utils/                # Standardized logging mechanisms and Matplotlib plotters
├── setup_server.sh           # 1-Click Cloud GPU Environment bootstrapper
└── requirements.txt          # Production-ready package constraints
```