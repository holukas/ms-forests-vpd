"""
Parallel ERA5 batch manager for Google Earth Engine downloads.

PRIMARY ERA5 DOWNLOAD SCRIPT
This script orchestrates multiple parallel downloads of ERA5 climate data
for FLUXNET sites using Google Earth Engine (GEE) by dividing the full site
list into batches and spawning separate processes.

Purpose:
    - Parallelize GEE ERA5 downloads across multiple CPU cores
    - Enable resumable downloads (each batch tracks its own progress)
    - Provide progress monitoring and batch-level error reporting
    - Cloud-based processing (no massive local file downloads)

Features:
    - Configurable number of parallel batches (default: 4)
    - Automatic batch sizing based on total sites
    - Per-batch log files for troubleshooting
    - Centralized error/warning/no-data logs (shared across batches)
    - Progress tracking with success/failure reporting
    - Graceful handling of partial failures
    - 30-year validation (1991-2020 with no gaps)

Usage:
    Simply run this file in PyCharm. The script will:
    1. Calculate batch size based on NUM_BATCHES
    2. Spawn N parallel download processes
    3. Monitor each batch for completion
    4. Report success/failure status
    5. Display batch-specific and centralized log file locations

Configuration:
    - NUM_BATCHES: Change this value to adjust parallelism
      - 2 batches: ~105 sites each (light load, GEE quota friendly)
      - 4 batches: ~52 sites each (balanced, default)
      - 6 batches: ~35 sites each (aggressive parallelism)

WARNING:
    - Google Earth Engine has API rate limits per project
    - Running too many parallel batches may trigger rate limits
    - Start with 2-3 batches if experiencing GEE quota errors
    - Batch log files show which sites completed vs. failed

Output:
    - Batch logs: 16_download_era5_gee_log_YYYYMMDD_HHMMSS.txt
    - Centralized logs:
      - 16_download_era5_ERRORS.log
      - 16_download_era5_WARNINGS.log
      - 16_download_era5_NO_DATA_SITES.log
    - Data: ../../data/outputs/10_datasets/16_ERA5_climate_1991-2020/{SITE}/

Dependencies:
    - 16b_download_era5_mat_map_GoogleEarthEngine.py
    - ee (earthengine-api)
    - pandas
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

# ==================== CONFIGURATION ====================
NUM_BATCHES = 4  # Number of parallel processes (default: 4, GEE-friendly)
# ========================================================

# Load datasets info to get total site count
infile = Path('../../data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv')
datasets_df = pd.read_csv(infile)
total_sites = len(datasets_df)

sites_per_batch = (total_sites + NUM_BATCHES - 1) // NUM_BATCHES

print("=" * 80)
print("ERA5 GOOGLE EARTH ENGINE PARALLEL DOWNLOAD")
print("=" * 80)
print(f"Total sites: {total_sites}")
print(f"Number of parallel batches: {NUM_BATCHES}")
print(f"Sites per batch: {sites_per_batch}\n")

processes = []
for batch_num in range(NUM_BATCHES):
    batch_start = batch_num * sites_per_batch
    batch_end = min((batch_num + 1) * sites_per_batch, total_sites)

    if batch_start >= total_sites:
        break

    print(f"Batch {batch_num + 1}/{NUM_BATCHES}: sites {batch_start:3d}-{batch_end - 1:3d} ({batch_end - batch_start:2d} sites)")

    # Spawn subprocess
    cmd = [
        sys.executable,
        "16b_download_era5_mat_map_GoogleEarthEngine.py",
        str(batch_start),
        str(batch_end)
    ]

    try:
        process = subprocess.Popen(cmd)
        processes.append((batch_num + 1, process, batch_start, batch_end))
    except Exception as e:
        print(f"  ERROR: Failed to start batch {batch_num + 1}: {e}")

print(f"\n{len(processes)} batch processes started.\n")
print("=" * 80)
print("MONITORING PROGRESS")
print("=" * 80 + "\n")

# Wait for all processes to complete
failed = []
for batch_num, process, batch_start, batch_end in processes:
    try:
        returncode = process.wait()
        status = "[OK] SUCCESS" if returncode == 0 else f"[FAILED] (exit code {returncode})"
        print(f"Batch {batch_num} (sites {batch_start:3d}-{batch_end - 1:3d}): {status}")
        if returncode != 0:
            failed.append(batch_num)
    except Exception as e:
        print(f"Batch {batch_num} (sites {batch_start:3d}-{batch_end - 1:3d}): [ERROR] {e}")
        failed.append(batch_num)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
if failed:
    print(f"COMPLETED WITH ERRORS: {len(failed)} batch(es) failed: {failed}")
    print(f"Success rate: {len(processes) - len(failed)}/{len(processes)}")
else:
    print(f"ALL {len(processes)} BATCHES COMPLETED SUCCESSFULLY!")
print("=" * 80)

# Check log files
log_dir = Path('../../data/outputs/10_datasets/16_ERA5_climate_1991-2020')
batch_logs = sorted(log_dir.glob('16_download_era5_gee_log_*.txt'))
if batch_logs:
    print(f"\nBatch log files created:")
    for log_file in batch_logs:
        print(f"  - {log_file.name}")

# Check centralized logs
summary_logs = [
    log_dir / "16_download_era5_ERRORS.log",
    log_dir / "16_download_era5_WARNINGS.log",
    log_dir / "16_download_era5_NO_DATA_SITES.log"
]
print(f"\nCentralized summary logs:")
for log_file in summary_logs:
    if log_file.exists():
        print(f"  - {log_file.name}")
