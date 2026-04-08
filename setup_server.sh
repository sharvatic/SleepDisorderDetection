#!/bin/bash
set -e

echo "=========================================================="
echo "      SLEEP DISORDER DETECTION — A100 SERVER SETUP        "
echo "=========================================================="

echo "[1/3] Upgrading pip..."
pip install --upgrade pip

echo "[2/3] Installing Python Dependencies..."
pip install -r requirements.txt

echo "[3/3] Downloading Full Clinical Dataset from Google Drive..."
# This will pull down the actual dataset and extract it into raw_data/actual/
python3 scripts/download_data.py --type actual

echo ""
echo "=========================================================="
echo "✅ SERVER SETUP COMPLETE!"
echo "=========================================================="
echo "Next Steps:"
echo " 1. Open notebooks/pipeline_run/execution_hub.ipynb"
echo " 2. Ensure DATA_DIR is pointed to 'raw_data/actual'"
echo " 3. Run all cells to process the dataset and train the model!"