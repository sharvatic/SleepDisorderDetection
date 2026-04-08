#!/usr/bin/env python3
"""
Dataset Download Utility
========================
Downloads the sample or full dataset directly from Google Drive and extracts it.

Usage:
    python scripts/download_data.py --type sample   # Downloads to raw_data/sample/
    python scripts/download_data.py --type actual   # Downloads to raw_data/actual/
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config.constants import SAMPLE_DATA_URL, DATA_URL

def download_and_extract(url, output_dir):
    try:
        import gdown
    except ImportError:
        print("Installing gdown to handle Google Drive links...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[download] Starting download into {output_dir}")
    print("[download] This might take a few minutes depending on file size...\n")
    
    # Download file using gdown (fuzzy=True handles standard 'view' urls)
    download_path = gdown.download(url, quiet=False, fuzzy=True, output=f"{output_dir}/")
    
    if not download_path:
        print("\n[error] Failed to download from Google Drive.")
        sys.exit(1)
        
    print(f"\n[download] Successfully downloaded to {download_path}")

    # Auto-extract if it's an archive
    if download_path.endswith(".zip"):
        print(f"[extract] Unzipping {download_path}...")
        import zipfile
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("[extract] Done! You can delete the .zip file to save space.")
    elif download_path.endswith(".tar.gz") or download_path.endswith(".tgz"):
        print(f"[extract] Untarring {download_path}...")
        import tarfile
        with tarfile.open(download_path, 'r:gz') as tar_ref:
            tar_ref.extractall(output_dir)
        print("[extract] Done! You can delete the archive to save space.")

def main():
    parser = argparse.ArgumentParser(description="Download Sleep Disorder Dataset")
    parser.add_argument("--type", choices=["sample", "actual"], default="sample", 
                        help="Type of dataset to download (sample or actual)")
    parser.add_argument("--output_dir", type=str, default="raw_data", 
                        help="Target base directory for downloaded data")
    args = parser.parse_args()

    url = SAMPLE_DATA_URL if args.type == "sample" else DATA_URL
    target = os.path.join(args.output_dir, args.type)
    
    download_and_extract(url, target)

if __name__ == "__main__":
    main()
