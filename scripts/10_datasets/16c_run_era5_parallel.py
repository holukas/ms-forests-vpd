"""
Run script 16d in parallel batches to download ERA5-Land data from Copernicus CDS.

Splits the site list into NUM_BATCHES equal batches, starts one 16d process
per batch, waits for all of them and reports which batches failed.

Settings:
- NUM_BATCHES = 3

Run it from scripts/10_datasets, because 16d is called by its file name.

Reads: data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv
Writes (through 16d), in data/outputs/10_datasets/16_ERA5_climate_1991-2020_Copernicus/:
per-site folders, per-batch logs (16_download_era5_log_batch_*.txt) and the
shared ERRORS, WARNINGS and NO_DATA_SITES logs.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
from src.paths import data_path

# ==================== CONFIGURATION ====================
NUM_BATCHES = 3  # Number of parallel processes
# ========================================================

# Load datasets info to get total site count
infile = data_path("data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv")
datasets_df = pd.read_csv(infile)
total_sites = len(datasets_df)

sites_per_batch = (total_sites + NUM_BATCHES - 1) // NUM_BATCHES

print("=" * 80)
print("ERA5 PARALLEL DOWNLOAD")
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
        "16d_download_era5_mat_map.py",
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
log_dir = data_path("data/outputs/10_datasets/16_ERA5_climate_1991-2020_Copernicus")
batch_logs = sorted(log_dir.glob('16_download_era5_log_batch_*.txt'))
summary_logs = [
    log_dir / "16_download_era5_ERRORS.log",
    log_dir / "16_download_era5_WARNINGS.log",
    log_dir / "16_download_era5_NO_DATA_SITES.log"
]

if batch_logs:
    print(f"\nBatch log files created:")
    for log_file in batch_logs:
        print(f"  - {log_file.name}")

print(f"\nSummary log files:")
for log_file in summary_logs:
    if log_file.exists():
        print(f"  - {log_file.name}")
