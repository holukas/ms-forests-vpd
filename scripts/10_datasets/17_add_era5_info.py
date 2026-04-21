"""
Add ERA5 Climate Data to FLUXNET Site Information

This script aggregates 30-year average (1991-2020) Mean Annual Temperature (MAT)
and Mean Annual Precipitation (MAP) from ERA5 reanalysis data for each FLUXNET site.

Data Source Strategy:
    - Primary: Google Earth Engine (GEE) ERA5 (available for all sites)
    - Override: FLUXNET embedded ERA5 (when available for SHUTTLE-CLI & AMERIFLUX)
    - Preference: Configurable per variable (PREF_SOURCE_MAT, PREF_SOURCE_MAP)

Input:
    - 15_datasets_info_parquet_vars_stats_usedsites.csv
      Contains site metadata including SITE, DOWNLOADED_VIA, and _DIRPATH

Output:
    - 17_datasets_info_parquet_vars_stats_usedsites_era5.csv
      Original site info plus:
      * ERA5_MAT_1991_2020: Mean Annual Temperature (°C)
      * ERA5_MAP_1991_2020: Mean Annual Precipitation (mm/year)
      * ERA5_MAT_SOURCE: Data source for temperature ('GEE' or 'FLUXNET')
      * ERA5_MAP_SOURCE: Data source for precipitation ('GEE' or 'FLUXNET')
    - 17_add_era5_info_YYYYMMDD_HHMMSS.log
      Complete execution log with all processing details

ERA5 Data Sources:
    1. Google Earth Engine (GEE) - AVAILABLE FOR ALL SITES:
       - Path: data/outputs/10_datasets/16_ERA5_climate_1991-2020/{SITE}/
       - File: {SITE}_era5_1991-2020_yearly.csv
       - Columns: MAT_degC (temperature), PRECIP_TOT_mm (precipitation)
       - Data range: 1991-2020 (pre-filtered)

    2. FLUXNET Files (SHUTTLE-CLI & AMERIFLUX) - OPTIONAL OVERRIDE:
       - ERA5 data embedded in downloaded FLUXNET files
       - Pattern: *_{SITE}_FLUXNET_ERA5_YY_*.csv
       - Columns: TA_ERA (temperature), P_ERA (precipitation)
       - Data range: 1981-2024, filtered to 1991-2020
       - Used only if available AND preferred for that variable

Configuration:
    - PREF_SOURCE_MAT: Set to 'FLUXNET' to prefer FLUXNET temp data, 'GEE' to always use GEE
    - PREF_SOURCE_MAP: Set to 'FLUXNET' to prefer FLUXNET precip data, 'GEE' to always use GEE
    - Default: Both set to 'FLUXNET' (prefer FLUXNET when available)
    - Quality Check: If FLUXNET precipitation > 3000 mm/year, GEE is used instead (outlier detection)

Validation:
    - Each source must have exactly 30 years of data (1991-2020)
    - All sites must have ERA5 data (from either GEE or FLUXNET)
    - Source selection follows preferences (or falls back to GEE if not available)
    - Detailed error messages if data is missing or incomplete
    - All output logged to file for audit trail
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Setup logging
output_dir = Path('../../data/outputs/10_datasets')
output_dir.mkdir(parents=True, exist_ok=True)
log_file = output_dir / f"17_add_era5_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        self.closed = False

    def write(self, message):
        self.terminal.write(message)
        if not self.closed:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        if not self.closed:
            self.log.flush()

    def close(self):
        if not self.closed:
            self.log.close()
            self.closed = True


sys.stdout = Logger(log_file)
print(f"Log file: {log_file}\n")

# Load datasets info
infile = Path('../../data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv')
datasets_df = pd.read_csv(infile)

# # Output folder for ERA5 data for each site
# dir_era5 = Path('../../data/outputs/10_datasets/16_ERA5_climate_1991-2020')

# Initialize new columns to store the 30-year averages
datasets_df['ERA5_MAT_1991_2020'] = np.nan
datasets_df['ERA5_MAP_1991_2020'] = np.nan
datasets_df['ERA5_MAT_SOURCE'] = ''
datasets_df['ERA5_MAP_SOURCE'] = ''

# ====== CONFIGURATION: Data source preferences ======
# Set to 'FLUXNET' to prefer FLUXNET ERA5 data when available
# Set to 'GEE' to always use Google Earth Engine data
PREF_SOURCE_MAT = 'FLUXNET'  # Temperature source preference
PREF_SOURCE_MAP = 'FLUXNET'  # Precipitation source preference
# ===================================================

# Track sites where precipitation is switched to GEE due to outlier detection
precip_outlier_sites = []

# Track all FLUXNET vs GEE precipitation comparisons for comprehensive analysis
precip_comparisons = []  # List of (site_id, fluxnet_val, gee_val, ratio, diff_percent)

print("Calculating 30-year MAT and MAP averages for each site...")

# Cycle through the sites and read their corresponding ERA5 CSVs
for index, row in datasets_df.iterrows():
    site_id = row['SITE']
    downloaded_via = row['DOWNLOADED_VIA']

    if site_id == 'JP-Ynf':
        print("X")

    # Initialize variables for each site
    mean_mat_gee = None
    mean_map_gee = None
    mean_mat_fxn = None
    mean_map_fxn = None
    source_mat = None
    source_map = None

    # ------------------------------------
    # Step 1: ERA5 from Google Earth Engine (DEFAULT/FALLBACK)
    # ------------------------------------
    # Available for all sites
    # Files already have the correct range 1991-2020
    dir_era5 = rf"..\..\data\outputs\10_datasets\16_ERA5_climate_1991-2020"
    file_era5 = f"{site_id}_era5_1991-2020_yearly.csv"
    filepath_era5 = Path(dir_era5) / file_era5
    if not filepath_era5.is_file():
        raise FileNotFoundError(f"ERA5 file not found for {site_id}: {filepath_era5}")
    tacol = 'MAT_degC'
    precipcol = 'PRECIP_TOT_mm'
    # Calculate the 30-year mean for MAT and MAP, rounded to 3 decimal places
    df_site = pd.read_csv(filepath_era5)  # Read yearly data
    if not len(df_site) == 30:
        raise Exception(f"Expected 30 years of data for {site_id}, found {len(df_site)}")
    mean_mat_gee = round(df_site[tacol].mean(), 3)
    mean_map_gee = round(df_site[precipcol].mean(), 3)

    # ------------------------------------
    # Step 2: Try to get FLUXNET ERA5 data if available
    # ------------------------------------
    # Available for SHUTTLE-CLI and AMERIFLUX sites
    fluxnet_sources = ['SHUTTLE-CLI', 'AMERIFLUX']
    if downloaded_via in fluxnet_sources:
        dir_era5 = row['_DIRPATH']
        pattern_file_era5 = f"*_{site_id}_FLUXNET_ERA5_YY_*.csv"
        tacol = 'TA_ERA'
        precipcol = 'P_ERA'
        # Find matching file using glob pattern
        matching_files = list(Path(dir_era5).glob(pattern_file_era5))
        if len(matching_files) == 0:
            raise FileNotFoundError(
                f"No ERA5 file found for {site_id} in {dir_era5}. Expected pattern: {pattern_file_era5}")
        if len(matching_files) > 1:
            raise Exception(f"Multiple files found for {site_id}: {matching_files}")
        filepath_era5 = matching_files[0]
        # Calculate the 30-year mean for MAT and MAP, rounded to 3 decimal places
        df_site = pd.read_csv(filepath_era5)  # Read yearly data
        locs = (df_site['TIMESTAMP'] >= 1991) & (df_site['TIMESTAMP'] <= 2020)
        df_site_1991_2020 = df_site.loc[locs].copy()
        if not len(df_site_1991_2020) == 30:
            raise Exception(f"Expected 30 years of data for {site_id}, found {len(df_site_1991_2020)}")
        mean_mat_fxn = round(df_site_1991_2020[tacol].mean(), 3)
        mean_map_fxn = round(df_site_1991_2020[precipcol].mean(), 3)

        # Track FLUXNET vs GEE precipitation comparison
        if mean_map_fxn is not None and mean_map_gee is not None:
            ratio = mean_map_fxn / mean_map_gee if mean_map_gee != 0 else 0
            diff_percent = abs(mean_map_fxn - mean_map_gee) / mean_map_gee * 100 if mean_map_gee != 0 else 0
            precip_comparisons.append((site_id, mean_map_fxn, mean_map_gee, ratio, diff_percent))

    # ------------------------------------
    # Step 3: Apply source preferences and select output values
    # ------------------------------------
    if downloaded_via in fluxnet_sources and mean_mat_fxn is not None:
        # FLUXNET data available for this site
        # Apply temperature preference
        if PREF_SOURCE_MAT == 'FLUXNET':
            mean_mat_out = mean_mat_fxn
            source_mat = 'FLUXNET'
        else:
            mean_mat_out = mean_mat_gee
            source_mat = 'GEE'

        # Apply precipitation preference with quality check
        # Use GEE if FLUXNET precip > 3000 mm/year (outlier detection)
        # Exception: if GEE > FLUXNET, use FLUXNET (smaller value preferred for outliers)
        if PREF_SOURCE_MAP == 'FLUXNET' and mean_map_fxn <= 3000:
            mean_map_out = mean_map_fxn
            source_map = 'FLUXNET'
        elif PREF_SOURCE_MAP == 'FLUXNET' and mean_map_fxn > 3000:
            # Outlier detected: FLUXNET > 3000 mm/year
            # If GEE > FLUXNET, use FLUXNET (both are high, choose lower)
            # If GEE <= FLUXNET, use GEE (GEE is lower)
            if mean_map_gee > mean_map_fxn:
                mean_map_out = mean_map_fxn
                source_map = 'FLUXNET'
                precip_outlier_sites.append((site_id, mean_map_fxn, mean_map_gee, 'GEE_higher'))
            else:
                mean_map_out = mean_map_gee
                source_map = 'GEE'
                precip_outlier_sites.append((site_id, mean_map_fxn, mean_map_gee, 'GEE_lower'))
        else:
            mean_map_out = mean_map_gee
            source_map = 'GEE'
    else:
        # No FLUXNET data or other source (FLUXNET_ORG, etc.)
        # Always use GEE
        mean_mat_out = mean_mat_gee
        mean_map_out = mean_map_gee
        source_mat = 'GEE'
        source_map = 'GEE'

    # Assign the values back to the main dataframe
    datasets_df.at[index, 'ERA5_MAT_1991_2020'] = mean_mat_out
    datasets_df.at[index, 'ERA5_MAP_1991_2020'] = mean_map_out
    datasets_df.at[index, 'ERA5_MAT_SOURCE'] = source_mat
    datasets_df.at[index, 'ERA5_MAP_SOURCE'] = source_map
    print(f"  {site_id}: MAT={mean_mat_out}°C ({source_mat}), MAP={mean_map_out}mm ({source_map})")

# Validate that all sites have ERA5 data
print(f"\n{'-' * 80}")
print("ERA5 DATA VALIDATION")
print(f"{'-' * 80}")

missing_era5 = datasets_df[datasets_df['ERA5_MAT_1991_2020'].isna()]
if len(missing_era5) > 0:
    print(f"\nERROR: {len(missing_era5)} sites are missing ERA5 data:")
    for _, row in missing_era5.iterrows():
        print(f"  - {row['SITE']} ({row['DOWNLOADED_VIA']})")
    raise Exception(f"ERA5 data is missing for {len(missing_era5)} sites. Cannot proceed.")
else:
    print(f"\n[OK] SUCCESS: All {len(datasets_df)} sites have ERA5 data (MAT and MAP)")
    print(
        f"  MAT range: {datasets_df['ERA5_MAT_1991_2020'].min():.1f} to {datasets_df['ERA5_MAT_1991_2020'].max():.1f} degC")
    print(
        f"  MAP range: {datasets_df['ERA5_MAP_1991_2020'].min():.1f} to {datasets_df['ERA5_MAP_1991_2020'].max():.1f} mm/year")

# ====== PRECIPITATION OUTLIER DETECTION SUMMARY ======
print(f"\n{'-' * 80}")
print("PRECIPITATION OUTLIER DETECTION (> 3000 mm/year)")
print(f"{'-' * 80}")
if len(precip_outlier_sites) > 0:
    print(f"\n[WARNING] {len(precip_outlier_sites)} site(s) detected FLUXNET precipitation > 3000 mm/year (outlier):\n")
    for outlier_info in precip_outlier_sites:
        site_id, fluxnet_val, gee_val, case = outlier_info
        print(f"  {site_id}:")
        print(f"    FLUXNET: {fluxnet_val:.1f} mm/year (> 3000 threshold)")
        print(f"    GEE:     {gee_val:.1f} mm/year")
        if case == 'GEE_higher':
            print(f"    Decision: Use FLUXNET (GEE is higher, FLUXNET is smaller)")
        else:  # GEE_lower
            print(f"    Decision: Use GEE (GEE is lower)")
else:
    print("\n[OK] No precipitation outliers detected. All FLUXNET values are <= 3000 mm/year.")

# ====== PRECIPITATION COMPARISON SUMMARY ======
print(f"\n{'-' * 80}")
print("PRECIPITATION FLUXNET vs GEE COMPARISON")
print(f"{'-' * 80}")

if len(precip_comparisons) > 0:
    # Sort by difference percentage (largest differences first)
    precip_comparisons_sorted = sorted(precip_comparisons, key=lambda x: x[4], reverse=True)

    print(f"\nComparing FLUXNET vs GEE precipitation for {len(precip_comparisons)} site(s):\n")
    print(f"{'Site':<15} {'FLUXNET':<12} {'GEE':<12} {'Ratio':<8} {'Diff %':<10} {'Highlight':<15}")
    print(f"{'-' * 80}")

    # Threshold for highlighting large differences
    large_diff_threshold = 30  # 30% difference

    for site_id, fluxnet_val, gee_val, ratio, diff_percent in precip_comparisons_sorted:
        # Determine highlight status
        if diff_percent > large_diff_threshold:
            highlight = "[LARGE DIFF]"
        else:
            highlight = ""

        print(f"{site_id:<15} {fluxnet_val:>10.1f}mm {gee_val:>10.1f}mm {ratio:>7.2f}x {diff_percent:>8.1f}% {highlight:<15}")

    # Summary statistics
    print(f"\n{'-' * 80}")
    all_ratios = [x[3] for x in precip_comparisons]
    all_diffs = [x[4] for x in precip_comparisons]
    large_diff_count = sum(1 for d in all_diffs if d > large_diff_threshold)

    print(f"Summary Statistics:")
    print(f"  Mean ratio (FLUXNET/GEE): {np.mean(all_ratios):.2f}x")
    print(f"  Min ratio:  {min(all_ratios):.2f}x")
    print(f"  Max ratio:  {max(all_ratios):.2f}x")
    print(f"  Mean difference: {np.mean(all_diffs):.1f}%")
    print(f"  Max difference:  {max(all_diffs):.1f}%")
    print(f"  Sites with >30% difference: {large_diff_count}/{len(precip_comparisons)}")
else:
    print("\nNo FLUXNET precipitation data available for comparison.")

# Save to file
datasets_df = datasets_df.reset_index(drop=True)
datasets_df = datasets_df.sort_values(by=['SITE'], inplace=False)
outfile = Path('../../data/outputs/10_datasets/17_datasets_info_parquet_vars_stats_usedsites_era5.csv')

print(f"\n{'-' * 80}\nSaving info about {len(datasets_df)} datasets to file {outfile}.\n{'-' * 80}")
datasets_df.to_csv(outfile, index=False)

print(f"\n{'-' * 80}")
print("SCRIPT COMPLETED SUCCESSFULLY")
print(f"{'-' * 80}")
print(f"Output CSV: {outfile}")
print(f"Log file: {log_file}")

# Close log file
sys.stdout.close()
