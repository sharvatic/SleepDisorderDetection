#!/usr/bin/env python3
"""
CAP Sleep Dataset — Dataset Build Entry Point
=============================================
This script orchestrates the full pipeline of loading raw EDF files, 
parsing annotations, extracting spectral features, generating 3D topomap 
tensors, and building the training manifest.

Usage:
    python scripts/build_data.py --data_dir /path/to/raw/edf --sample_norms 30
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path so we can import src/config
sys.path.append(str(Path(__file__).parent.parent))

from src.data.builder import build_dataset
from src.utils.logger import setup_logger
from config.constants import DATASET_ROOT

def main():
    parser = argparse.ArgumentParser(description="Build Sleep Disorder Dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to directory containing .edf and .edf.st files")
    parser.add_argument("--output_dir", type=str, default=DATASET_ROOT,
                        help=f"Directory to save processed results (default: {DATASET_ROOT})")
    parser.add_argument("--sample_norms", type=int, default=30,
                        help="Number of patients to sample for global normalization (default: 30)")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger = setup_logger("build_data", "training_output")

    logger.info("="*60)
    logger.info("SLEEP DISORDER DETECTION — DATASET BUILDER")
    logger.info("="*60)
    logger.info(f"Source Data : {args.data_dir}")
    logger.info(f"Output Path : {args.output_dir}")
    logger.info(f"Norm Samples: {args.sample_norms}")
    logger.info("="*60)

    try:
        build_dataset(
            data_dir=args.data_dir, 
            output_dir=args.output_dir, 
            sample_norms=args.sample_norms
        )
        logger.info("[success] Dataset build completed successfully.")
    except Exception as e:
        logger.error(f"[error] Dataset build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
