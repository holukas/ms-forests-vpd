"""
Find overlapping folder names between AmeriFlux and FLUXNET-Shuttle data sources.

This script identifies datasets that exist in both the AmeriFlux and FLUXNET-Shuttle
directories, which could indicate duplicate or redundant data sources.

Optional: Delete overlapping folders from AmeriFlux directory (keeping FLUXNET-Shuttle as primary).
"""

from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime
from src.paths import data_path

# ===== SETTINGS =====
DELETE_OVERLAPPING_FOLDERS = True  # Set to False to skip deletion
# ====================

# Define directories
ameriflux_dir = data_path("data/00_raw/ameriflux")
shuttle_dir = data_path("data/00_raw/from-fluxnet-shuttle")

# Folders to exclude (metadata/info folders)
EXCLUDE_PATTERNS = {'0-info', '1-extended_info', '1-zips', '.cache'}

print(f"Scanning for overlapping folders between data sources...\n")
print(f"AmeriFlux directory:     {ameriflux_dir.resolve()}")
print(f"FLUXNET-Shuttle directory: {shuttle_dir.resolve()}\n")

# Get folder names from both directories
def get_folder_names(directory):
    """Get all folder names, excluding metadata folders."""
    if not directory.exists():
        print(f"WARNING: Directory not found: {directory}")
        return set()

    folders = set()
    for item in directory.iterdir():
        if item.is_dir():
            folder_name = item.name
            # Skip metadata folders
            if folder_name not in EXCLUDE_PATTERNS:
                folders.add(folder_name)
    return folders

ameriflux_folders = get_folder_names(ameriflux_dir)
shuttle_folders = get_folder_names(shuttle_dir)

print(f"Found {len(ameriflux_folders)} datasets in AmeriFlux")
print(f"Found {len(shuttle_folders)} datasets in FLUXNET-Shuttle\n")

# Find overlaps
overlapping_folders = ameriflux_folders.intersection(shuttle_folders)
only_in_ameriflux = ameriflux_folders - shuttle_folders
only_in_shuttle = shuttle_folders - ameriflux_folders

print("=" * 80)
print(f"OVERLAPPING DATASETS: {len(overlapping_folders)} found")
print("=" * 80)

if overlapping_folders:
    overlapping_list = sorted(list(overlapping_folders))
    for folder in overlapping_list:
        print(f"  {folder}")
else:
    print("  (No overlaps found)")

print(f"\n{'=' * 80}")
print(f"ONLY IN AMERIFLUX: {len(only_in_ameriflux)} datasets")
print("=" * 80)
if len(only_in_ameriflux) <= 20:
    for folder in sorted(list(only_in_ameriflux)):
        print(f"  {folder}")
else:
    print(f"  (Showing first 10 of {len(only_in_ameriflux)})")
    for folder in sorted(list(only_in_ameriflux))[:10]:
        print(f"  {folder}")
    print(f"  ... and {len(only_in_ameriflux) - 10} more")

print(f"\n{'=' * 80}")
print(f"ONLY IN FLUXNET-SHUTTLE: {len(only_in_shuttle)} datasets")
print("=" * 80)
if len(only_in_shuttle) <= 20:
    for folder in sorted(list(only_in_shuttle)):
        print(f"  {folder}")
else:
    print(f"  (Showing first 10 of {len(only_in_shuttle)})")
    for folder in sorted(list(only_in_shuttle))[:10]:
        print(f"  {folder}")
    print(f"  ... and {len(only_in_shuttle) - 10} more")

# Save results to CSV
output_dir = data_path("data/outputs/10_datasets")
output_dir.mkdir(parents=True, exist_ok=True)

# Create summary dataframe
summary_data = {
    'LOCATION': ['AmeriFlux', 'FLUXNET-Shuttle', 'Overlapping', 'Only in AmeriFlux', 'Only in FLUXNET-Shuttle'],
    'COUNT': [len(ameriflux_folders), len(shuttle_folders), len(overlapping_folders), len(only_in_ameriflux), len(only_in_shuttle)]
}
summary_df = pd.DataFrame(summary_data)

summary_file = output_dir / '00_overlap_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"\n[SAVED] Summary to: {summary_file}\n")

# Save detailed overlapping list
if overlapping_folders:
    overlapping_df = pd.DataFrame({
        'OVERLAPPING_FOLDER_NAME': sorted(list(overlapping_folders))
    })
    overlapping_file = output_dir / '00_overlapping_datasets.csv'
    overlapping_df.to_csv(overlapping_file, index=False)
    print(f"[SAVED] Overlapping datasets to: {overlapping_file}")

# Save detailed lists
ameriflux_only_df = pd.DataFrame({
    'FOLDER_NAME': sorted(list(only_in_ameriflux))
})
ameriflux_only_file = output_dir / '00_only_in_ameriflux.csv'
ameriflux_only_df.to_csv(ameriflux_only_file, index=False)
print(f"[SAVED] AmeriFlux-only datasets to: {ameriflux_only_file}")

shuttle_only_df = pd.DataFrame({
    'FOLDER_NAME': sorted(list(only_in_shuttle))
})
shuttle_only_file = output_dir / '00_only_in_shuttle.csv'
shuttle_only_df.to_csv(shuttle_only_file, index=False)
print(f"[SAVED] FLUXNET-Shuttle-only datasets to: {shuttle_only_file}\n")

print(f"{'=' * 80}")
print("RECOMMENDATION:")
print(f"{'=' * 80}")
if overlapping_folders:
    print(f"\nWARNING: Found {len(overlapping_folders)} overlapping datasets!")
    print("You may have duplicate downloads. Consider:")
    print("  1. Checking which source has better data quality")
    print("  2. Removing duplicate folders to save disk space")
    print("  3. Verifying timestamps and versions are consistent")
else:
    print("\nGood: No overlapping datasets found between sources.")
    print("Your data sources appear to be complementary.")

# --- DELETE OVERLAPPING FOLDERS FROM AMERIFLUX ---
if DELETE_OVERLAPPING_FOLDERS and overlapping_folders:
    print(f"\n{'=' * 80}")
    print("DELETING OVERLAPPING FOLDERS FROM AMERIFLUX")
    print(f"{'=' * 80}\n")

    print(f"CAUTION: About to delete {len(overlapping_folders)} folders from AmeriFlux!")
    print(f"Keeping FLUXNET-Shuttle as primary source.\n")

    # Confirm before deletion
    user_input = input("Type 'DELETE' to confirm deletion (or press Enter to skip): ").strip().upper()

    if user_input == 'DELETE':
        print("\nStarting deletion...\n")

        deleted_folders = []
        failed_folders = []
        deleted_size = 0

        for folder_name in sorted(list(overlapping_folders)):
            folder_path = ameriflux_dir / folder_name

            if folder_path.exists():
                try:
                    # Get folder size before deletion
                    def get_folder_size(path):
                        total = 0
                        for entry in path.rglob('*'):
                            if entry.is_file():
                                total += entry.stat().st_size
                        return total

                    folder_size = get_folder_size(folder_path)

                    # Delete folder
                    shutil.rmtree(folder_path)
                    deleted_folders.append(folder_name)
                    deleted_size += folder_size
                    print(f"  [DELETED] {folder_name} ({folder_size / (1024**3):.2f} GB)")

                except Exception as e:
                    failed_folders.append((folder_name, str(e)))
                    print(f"  [FAILED] {folder_name}: {e}")

        # Save deletion log
        log_file = output_dir / f'00_deletion_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        deletion_data = {
            'FOLDER_NAME': deleted_folders,
            'STATUS': ['DELETED'] * len(deleted_folders)
        }
        deletion_df = pd.DataFrame(deletion_data)
        deletion_df.to_csv(log_file, index=False)

        print(f"\n{'=' * 80}")
        print("DELETION SUMMARY")
        print(f"{'=' * 80}")
        print(f"Successfully deleted: {len(deleted_folders)} folders")
        print(f"Failed to delete:     {len(failed_folders)} folders")
        print(f"Space freed:          {deleted_size / (1024**3):.2f} GB")
        print(f"\nDeletion log saved to: {log_file}")

        if failed_folders:
            print(f"\nFailed deletions:")
            for folder_name, error in failed_folders:
                print(f"  {folder_name}: {error}")
    else:
        print("\nDeletion cancelled. No folders were removed.")
elif DELETE_OVERLAPPING_FOLDERS:
    print("\nNo overlapping folders found to delete.")
