"""
Add 1991-2020 ERA5 mean annual temperature (MAT) and precipitation (MAP) to the site table.

Each site gets 30-year means from three ERA5 sources: the GEE and Copernicus
yearly files from scripts 16a-16d (required for every site), and the ERA5
file shipped with the flux data for SHUTTLE-CLI and AMERIFLUX sites
(TA_ERA, P_ERA). Each source must cover all 30 years.

One source is chosen per site from precipitation and then used for both MAT
and MAP:
- FLUXNET data available, MAP <= 2000 mm/year: FLUXNET.
- FLUXNET MAP > 2000 mm/year: FLUXNET if Copernicus is within 300 mm or
  higher; otherwise Copernicus; if Copernicus is zero, the lower of GEE and
  FLUXNET.
- No FLUXNET data: Copernicus if nonzero, else GEE.

Reads: data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv
Writes, in data/outputs/10_datasets/:
- 17_datasets_info_parquet_vars_stats_usedsites_era5.csv, with the added
  columns ERA5_MAT_1991_2020, ERA5_MAP_1991_2020, ERA5_MAT_SOURCE and
  ERA5_MAP_SOURCE
- a time-stamped log 17_add_era5_info_*.log with source comparison tables
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from src.paths import data_path

# Setup logging
output_dir = data_path("data/outputs/10_datasets")
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
print("=" * 100)
print("ERA5 CLIMATE DATA AGGREGATION (30-year: 1991-2020)")
print("=" * 100)
print(f"\nLog file: {log_file}\n")

# Display data source locations
print("DATA SOURCES:")
print(f"  Input Site Info: <DATA_ROOT>/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv")
print(f"  GEE ERA5 Data: <DATA_ROOT>/outputs/10_datasets/16_ERA5_climate_1991-2020_GoogleEarthEngine/{{SITE}}/")
print(f"  Copernicus ERA5 Data: <DATA_ROOT>/outputs/10_datasets/16_ERA5_climate_1991-2020_Copernicus/{{SITE}}/")
print(f"  FLUXNET Embedded ERA5: In FLUXNET download files (SHUTTLE-CLI, AMERIFLUX)")
print(f"  Output: <DATA_ROOT>/outputs/10_datasets/17_datasets_info_parquet_vars_stats_usedsites_era5.csv")

print("\n" + "=" * 100)
print("PROCESSING LOGIC:")
print("=" * 100)
print("STEP 1: Precipitation Source Selection (Primary Decision)")
print("  - Analyze precipitation (MAP) with multi-tiered validation")
print("  - Default: Use FLUXNET if available")
print("  - Special validation for high-precipitation sites (>2000 mm/year):")
print("    * Compare FLUXNET against Copernicus/CDS and GEE")
print("    * Select source based on data agreement and quality")
print("  - Fallback: Copernicus (non-zero) then GEE")
print("\nSTEP 2: Temperature Source Selection (Unified)")
print("  - Use the SAME source selected for precipitation")
print("  - Extract temperature (MAT) from selected source")
print("  - Result: Both variables use consistent data source")
print("\nSTEP 3: Output & Validation")
print("  - Both ERA5_MAT_SOURCE and ERA5_MAP_SOURCE set to same source")
print("  - Comprehensive comparison tables show source agreement")
print("  - Audit trail shows why each source was selected")
print("=" * 100 + "\n")

# Load datasets info
infile = data_path("data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv")
datasets_df = pd.read_csv(infile)

# # Output folder for ERA5 data for each site
# dir_era5 = data_path("data/outputs/10_datasets/16_ERA5_climate_1991-2020")

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
precip_comparisons = []  # List of (site_id, fluxnet_val, cccs_val, gee_val, ratio, diff_percent, gee_diff)

# Track all three-source precipitation comparisons (GEE, Copernicus, FLUXNET)
all_three_source_comparisons = []  # List of (site_id, gee_val, cccs_val, fluxnet_val)

# Track temperature comparisons (similar to precipitation)
temp_comparisons = []  # List of (site_id, fluxnet_val, cccs_val, gee_val)
all_three_source_temp_comparisons = []  # List of (site_id, gee_val, cccs_val, fluxnet_val)

print("Calculating 30-year MAT and MAP averages for each site...")

# Cycle through the sites and read their corresponding ERA5 CSVs
for index, row in datasets_df.iterrows():
    site_id = row['SITE']
    downloaded_via = row['DOWNLOADED_VIA']

    # Initialize variables for each site
    mean_mat_gee = None
    mean_map_gee = None
    mean_mat_fxn = None
    mean_map_fxn = None
    source_mat = None
    source_map = None

    # ------------------------------------
    # Step 1: ERA5 from Google Earth Engine
    # ------------------------------------
    # Available for all sites
    # Files already have the correct range 1991-2020
    dir_era5 = rf"..\..\data\outputs\10_datasets\16_ERA5_climate_1991-2020_GoogleEarthEngine"
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
    # Step 2: ERA5 from Copernicus Climate Change Service
    # ------------------------------------
    # Available for all sites
    # Files already have the correct range 1991-2020
    dir_era5 = rf"..\..\data\outputs\10_datasets\16_ERA5_climate_1991-2020_Copernicus\{site_id}"
    file_era5 = f"{site_id}_era5_1991-2020_yearly.csv"
    filepath_era5 = Path(dir_era5) / file_era5
    if not filepath_era5.is_file():
        raise FileNotFoundError(f"ERA5 file not found for {site_id}: {filepath_era5}")
    tacol = 'TA_degC'
    precipcol = 'PRECIP_TOT_mm'
    # Calculate the 30-year mean for MAT and MAP, rounded to 3 decimal places
    df_site = pd.read_csv(filepath_era5)  # Read yearly data
    if not len(df_site) == 30:
        raise Exception(f"Expected 30 years of data for {site_id}, found {len(df_site)}")
    mean_mat_cccs = round(df_site[tacol].mean(), 3)
    mean_map_cccs = round(df_site[precipcol].mean(), 3)

    # ------------------------------------
    # Step 3: Try to get FLUXNET ERA5 data if available
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

        # Track FLUXNET vs Copernicus precipitation comparison (with GEE for cross-check)
        cccs_valid = mean_map_cccs is not None and not np.isnan(mean_map_cccs)
        if mean_map_fxn is not None and cccs_valid:
            ratio = mean_map_fxn / mean_map_cccs if mean_map_cccs != 0 else 0
            diff_percent = abs(mean_map_fxn - mean_map_cccs) / mean_map_cccs * 100 if mean_map_cccs != 0 else 0
            gee_diff = abs(mean_map_gee - mean_map_cccs)
            # Track: (site_id, fluxnet, copernicus, gee, ratio, diff_percent, gee_diff_abs)
            precip_comparisons.append((site_id, mean_map_fxn, mean_map_cccs, mean_map_gee, ratio, diff_percent, gee_diff))

    # Track all three sources for comparison (GEE, Copernicus, FLUXNET)
    all_three_source_comparisons.append((site_id, mean_map_gee, mean_map_cccs, mean_map_fxn if downloaded_via in fluxnet_sources else None))

    # Track temperature data for later comparison
    temp_comparisons.append((site_id, mean_mat_fxn, mean_mat_cccs, mean_mat_gee))
    all_three_source_temp_comparisons.append((site_id, mean_mat_gee, mean_mat_cccs, mean_mat_fxn if downloaded_via in fluxnet_sources else None))

    # ------------------------------------
    # Step 4: Determine precipitation source (which will also be used for temperature)
    # ------------------------------------
    source_map = None  # Will be determined by the logic below
    mean_map_out = None

    if downloaded_via in fluxnet_sources and mean_map_fxn is not None:
        # FLUXNET precipitation data available for this site
        # Apply precipitation preference with sophisticated validation
        # Default: Use FLUXNET if available
        # Special case: For precip > 2000 mm/year, validate against other sources

        if mean_map_fxn > 2000:
            # High precipitation: validate FLUXNET against Copernicus
            # Check if Copernicus is similar (within ±300mm) to FLUXNET
            cccs_valid = mean_map_cccs is not None and not np.isnan(mean_map_cccs)
            cccs_diff = abs(mean_map_cccs - mean_map_fxn) if cccs_valid else float('inf')
            gee_diff = abs(mean_map_gee - mean_map_fxn)

            if cccs_valid and cccs_diff <= 300:
                # Copernicus is similar to FLUXNET (±300mm) - use FLUXNET
                mean_map_out = mean_map_fxn
                source_map = 'FLUXNET'
                precip_outlier_sites.append((site_id, mean_map_fxn, mean_map_cccs, 'FLUXNET_used'))
            elif cccs_valid and mean_map_cccs > mean_map_fxn:
                # Copernicus is HIGHER than FLUXNET - prefer lower value (FLUXNET)
                mean_map_out = mean_map_fxn
                source_map = 'FLUXNET'
                precip_outlier_sites.append((site_id, mean_map_fxn, mean_map_cccs, 'FLUXNET_lower'))
            else:
                # Copernicus lower and dissimilar (or unavailable) - use Copernicus as fallback
                # But reject if Copernicus is exactly zero (unrealistic)
                if cccs_valid and mean_map_cccs != 0:
                    mean_map_out = mean_map_cccs
                    source_map = 'Copernicus'
                    precip_outlier_sites.append((site_id, mean_map_fxn, mean_map_cccs, 'Copernicus_fallback'))
                elif cccs_valid and mean_map_cccs == 0:
                    # Copernicus is zero (rejected) - compare GEE with FLUXNET
                    if mean_map_gee > mean_map_fxn:
                        # GEE is higher - use FLUXNET (prefer lower value)
                        mean_map_out = mean_map_fxn
                        source_map = 'FLUXNET'
                        precip_outlier_sites.append((site_id, mean_map_fxn, mean_map_gee, 'GEE_higher_use_FLUXNET'))
                    else:
                        # GEE is lower/similar - use GEE
                        mean_map_out = mean_map_gee
                        source_map = 'GEE'
                        precip_outlier_sites.append((site_id, mean_map_fxn, mean_map_gee, 'Copernicus_zero_rejected'))
                else:
                    mean_map_out = mean_map_gee
                    source_map = 'GEE'
                    precip_outlier_sites.append((site_id, mean_map_fxn, mean_map_gee, 'GEE_fallback'))
        else:
            # Normal precipitation (≤2000 mm/year) - use FLUXNET directly
            mean_map_out = mean_map_fxn
            source_map = 'FLUXNET'
    else:
        # No FLUXNET data (FLUXNET_ORG or other sources)
        # Fallback: Copernicus if available and non-zero, else GEE
        # For precipitation, prefer Copernicus as fallback (but reject if zero)
        if mean_map_cccs is not None and not np.isnan(mean_map_cccs) and mean_map_cccs != 0:
            mean_map_out = mean_map_cccs
            source_map = 'Copernicus'
        else:
            mean_map_out = mean_map_gee
            source_map = 'GEE'

    # ------------------------------------
    # Step 5: Apply selected source to BOTH temperature and precipitation
    # The source is determined by precipitation validation, then applied to both
    # ------------------------------------
    source_mat = source_map  # Use same source for temperature as precipitation

    # Select temperature value from the appropriate source
    if source_mat == 'FLUXNET':
        mean_mat_out = mean_mat_fxn
    elif source_mat == 'Copernicus':
        mean_mat_out = mean_mat_cccs
    else:  # GEE
        mean_mat_out = mean_mat_gee

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

# ====== HIGH PRECIPITATION SUMMARY (> 2000 mm/year) ======
print(f"\n{'-' * 80}")
print("HIGH PRECIPITATION VALIDATION (FLUXNET > 2000 mm/year)")
print(f"{'-' * 80}")
if len(precip_outlier_sites) > 0:
    print(f"\n[INFO] {len(precip_outlier_sites)} site(s) with FLUXNET precipitation > 2000 mm/year:\n")

    fluxnet_used_count = 0
    fluxnet_lower_count = 0
    cccs_fallback_count = 0
    cccs_zero_rejected_count = 0
    gee_higher_use_fluxnet_count = 0
    gee_fallback_count = 0

    for outlier_info in precip_outlier_sites:
        site_id, fluxnet_val, compare_val, case = outlier_info
        print(f"  {site_id}:")
        print(f"    FLUXNET: {fluxnet_val:.1f} mm/year")

        if case == 'FLUXNET_used':
            print(f"    Copernicus: {compare_val:.1f} mm/year [SIMILAR: within ±300mm]")
            print(f"    Decision: Use FLUXNET (Copernicus validates)")
            fluxnet_used_count += 1
        elif case == 'FLUXNET_lower':
            diff = abs(compare_val - fluxnet_val)
            print(f"    Copernicus: {compare_val:.1f} mm/year [HIGHER by {diff:.1f}mm]")
            print(f"    Decision: Use FLUXNET (lower value preferred)")
            fluxnet_lower_count += 1
        elif case == 'Copernicus_fallback':
            diff = abs(compare_val - fluxnet_val)
            print(f"    Copernicus: {compare_val:.1f} mm/year [DISSIMILAR: {diff:.1f}mm difference, lower]")
            # Cross-check with GEE
            gee_info = next((x for x in all_three_source_comparisons if x[0] == site_id), None)
            if gee_info:
                gee_val = gee_info[1]  # GEE is second element
                gee_cccs_diff = abs(gee_val - compare_val)
                if gee_cccs_diff > 150:
                    print(f"    GEE: {gee_val:.1f} mm/year [DISSIMILAR from Copernicus: {gee_cccs_diff:.1f}mm]")
                else:
                    print(f"    GEE: {gee_val:.1f} mm/year [similar to Copernicus]")
            print(f"    Decision: Use Copernicus (FLUXNET validation failed)")
            cccs_fallback_count += 1
        elif case == 'Copernicus_zero_rejected':
            print(f"    Copernicus: 0.0 mm/year [REJECTED: unrealistic value]")
            print(f"    GEE: {compare_val:.1f} mm/year (fallback)")
            print(f"    Decision: Use GEE (Copernicus is zero)")
            cccs_zero_rejected_count += 1
        elif case == 'GEE_higher_use_FLUXNET':
            print(f"    Copernicus: 0.0 mm/year [REJECTED: unrealistic value]")
            print(f"    GEE: {compare_val:.1f} mm/year [HIGHER than FLUXNET by {compare_val - fluxnet_val:.1f}mm]")
            print(f"    Decision: Use FLUXNET (GEE is anomalously high)")
            gee_higher_use_fluxnet_count += 1
        else:  # GEE_fallback
            print(f"    GEE: {compare_val:.1f} mm/year (fallback)")
            print(f"    Decision: Use GEE (Copernicus not available)")
            gee_fallback_count += 1

    print(f"\n{'-' * 80}")
    print(f"Summary of {len(precip_outlier_sites)} high-precipitation sites:")
    print(f"  FLUXNET used (Copernicus within ±300mm): {fluxnet_used_count}")
    print(f"  FLUXNET used (Copernicus is higher): {fluxnet_lower_count}")
    print(f"  Copernicus fallback (lower & dissimilar): {cccs_fallback_count}")
    if cccs_zero_rejected_count > 0:
        print(f"  GEE fallback (Copernicus is zero, GEE lower/similar): {cccs_zero_rejected_count}")
    if gee_higher_use_fluxnet_count > 0:
        print(f"  FLUXNET used (Copernicus zero, GEE anomalously high): {gee_higher_use_fluxnet_count}")
    if gee_fallback_count > 0:
        print(f"  GEE fallback (no Copernicus): {gee_fallback_count}")
else:
    print("\n[OK] No high-precipitation sites (> 2000 mm/year).")

# ====== PRECIPITATION COMPARISON SUMMARY ======
print(f"\n{'-' * 100}")
print("PRECIPITATION FLUXNET vs COPERNICUS COMPARISON (with GEE cross-check)")
print(f"{'-' * 100}")

if len(precip_comparisons) > 0:
    # Sort by difference percentage (largest differences first)
    precip_comparisons_sorted = sorted(precip_comparisons, key=lambda x: x[5], reverse=True)

    print(f"\nComparing FLUXNET vs Copernicus for {len(precip_comparisons)} site(s):\n")
    print(f"{'Site':<12} {'FLUXNET':<12} {'Copernicus':<12} {'Ratio':<8} {'Diff %':<10} {'GEE Check':<15}")
    print(f"{'-' * 100}")

    # Threshold for highlighting large differences
    large_diff_threshold = 30  # 30% difference

    for site_id, fluxnet_val, cccs_val, gee_val, ratio, diff_percent, gee_diff in precip_comparisons_sorted:
        # Determine GEE cross-check status
        if gee_diff > 150:
            gee_check = f"[DISSIMILAR:{gee_diff:.0f}mm]"
        else:
            gee_check = "[Similar]"

        # Determine FLUXNET vs Copernicus difference highlighting
        if diff_percent > large_diff_threshold:
            highlight = "LARGE"
        else:
            highlight = ""

        print(f"{site_id:<12} {fluxnet_val:>10.1f}mm {cccs_val:>10.1f}mm {ratio:>7.2f}x {diff_percent:>8.1f}% {gee_check:<15}")

    # Summary statistics
    print(f"\n{'-' * 100}")
    all_ratios = [x[4] for x in precip_comparisons]
    all_diffs = [x[5] for x in precip_comparisons]
    all_gee_diffs = [x[6] for x in precip_comparisons]
    large_diff_count = sum(1 for d in all_diffs if d > large_diff_threshold)
    gee_dissimilar_count = sum(1 for d in all_gee_diffs if d > 150)

    print(f"FLUXNET vs Copernicus Statistics:")
    print(f"  Mean ratio (FLUXNET/Copernicus): {np.mean(all_ratios):.2f}x")
    print(f"  Min ratio:  {min(all_ratios):.2f}x")
    print(f"  Max ratio:  {max(all_ratios):.2f}x")
    print(f"  Mean difference: {np.mean(all_diffs):.1f}%")
    print(f"  Max difference:  {max(all_diffs):.1f}%")
    print(f"  Sites with >30% difference: {large_diff_count}/{len(precip_comparisons)}")

    print(f"\nGEE Cross-Check (vs Copernicus):")
    print(f"  Sites with dissimilar GEE (>150mm diff): {gee_dissimilar_count}/{len(precip_comparisons)}")
    if gee_dissimilar_count > 0:
        print(f"  Mean GEE-Copernicus difference: {np.mean(all_gee_diffs):.1f}mm")
        print(f"  Max GEE-Copernicus difference: {max(all_gee_diffs):.1f}mm")
else:
    print("\nNo FLUXNET precipitation data available for comparison.")

# ====== THREE-SOURCE PRECIPITATION COMPARISON ======
print(f"\n{'-' * 100}")
print("ALL THREE SOURCES PRECIPITATION COMPARISON (GEE, Copernicus, FLUXNET)")
print(f"{'-' * 100}")

if len(all_three_source_comparisons) > 0:
    print(f"\nComparing all three precipitation sources for {len(all_three_source_comparisons)} site(s):\n")
    print(f"{'Site':<12} {'GEE':<12} {'Copernicus':<12} {'FLUXNET':<12} {'Selected':<12} {'Reason':<50} {'Status':<25}")
    print(f"{'-' * 185}")

    # Create mapping of site_id to outlier case for quick lookup
    outlier_cases = {site_id: case for site_id, _, _, case in precip_outlier_sites}

    gee_vals = []
    cccs_vals = []
    fluxnet_vals = []
    empty_cccs_count = 0
    no_fluxnet_count = 0
    good_agreement_count = 0

    for site_id, gee_val, cccs_val, fluxnet_val in sorted(all_three_source_comparisons, key=lambda x: x[0]):
        # Format values, handle None/NaN
        gee_str = f"{gee_val:.1f}mm" if gee_val is not None else "N/A"
        cccs_str = f"{cccs_val:.1f}mm" if (cccs_val is not None and not np.isnan(cccs_val)) else "EMPTY"
        fluxnet_str = f"{fluxnet_val:.1f}mm" if fluxnet_val is not None else "N/A"

        # Get selected source from dataframe
        selected_source = datasets_df.loc[datasets_df['SITE'] == site_id, 'ERA5_MAP_SOURCE']
        if len(selected_source) > 0:
            selected = selected_source.values[0]
        else:
            selected = "?"

        # Generate reason for selection
        reason = ""
        if site_id in outlier_cases:
            # High-precipitation case - use the case label to generate reason
            case = outlier_cases[site_id]
            if case == 'FLUXNET_used':
                reason = "FLUXNET: Copernicus within ±300mm (validation passed)"
            elif case == 'FLUXNET_lower':
                reason = "FLUXNET: Copernicus higher (prefer lower value)"
            elif case == 'Copernicus_fallback':
                reason = "Copernicus: precip validation failed (fallback)"
            elif case == 'Copernicus_zero_rejected':
                reason = "GEE: Copernicus zero (unrealistic, rejected)"
            elif case == 'GEE_higher_use_FLUXNET':
                reason = "FLUXNET: GEE anomalously high (prefer lower)"
            elif case == 'GEE_fallback':
                reason = "GEE: Copernicus unavailable (fallback)"
        elif fluxnet_val is not None:
            # No special case, FLUXNET available
            reason = "FLUXNET available (normal precipitation ≤2000 mm/year)"
        elif cccs_val is not None and not np.isnan(cccs_val):
            # Copernicus fallback (no FLUXNET)
            if cccs_val == 0:
                reason = "GEE: Copernicus zero (unrealistic, rejected)"
            else:
                reason = "Copernicus: FLUXNET not available (fallback)"
        else:
            # GEE only
            reason = "GEE: Only valid source available"

        # Determine status
        if cccs_val is None or np.isnan(cccs_val):
            status = "[CCCS EMPTY]"
            empty_cccs_count += 1
        elif fluxnet_val is None:
            status = "[No FLUXNET]"
            no_fluxnet_count += 1
        else:
            # All three sources available - check agreement within ±10%
            # Calculate min/max and check if range is within 10% of mean
            values = [gee_val, cccs_val, fluxnet_val]
            mean_val = np.mean(values)
            min_val = min(values)
            max_val = max(values)
            range_percent = ((max_val - min_val) / mean_val * 100) if mean_val != 0 else 0

            if range_percent <= 10:
                status = "[GOOD AGREEMENT]"
                good_agreement_count += 1
            else:
                status = f"[Disagree: {range_percent:.1f}%]"

        print(f"{site_id:<12} {gee_str:<12} {cccs_str:<12} {fluxnet_str:<12} {selected:<12} {reason:<50} {status:<25}")

        # Collect non-null values for statistics
        if gee_val is not None:
            gee_vals.append(gee_val)
        if cccs_val is not None and not np.isnan(cccs_val):
            cccs_vals.append(cccs_val)
        if fluxnet_val is not None:
            fluxnet_vals.append(fluxnet_val)

    # Summary statistics for each source
    print(f"\n{'-' * 100}")
    print(f"Source Availability and Statistics:")
    print(f"{'-' * 100}")
    print(f"GEE:")
    print(f"  Available for: {len(gee_vals)}/{len(all_three_source_comparisons)} sites")
    if gee_vals:
        print(f"  Range: {min(gee_vals):.1f} - {max(gee_vals):.1f} mm/year")
        print(f"  Mean: {np.mean(gee_vals):.1f} mm/year")

    print(f"\nCopernicus/CDS:")
    print(f"  Available for: {len(cccs_vals)}/{len(all_three_source_comparisons)} sites (Empty: {empty_cccs_count})")
    if cccs_vals:
        print(f"  Range: {min(cccs_vals):.1f} - {max(cccs_vals):.1f} mm/year")
        print(f"  Mean: {np.mean(cccs_vals):.1f} mm/year")

    print(f"\nFLUXNET:")
    print(f"  Available for: {len(fluxnet_vals)}/{len(all_three_source_comparisons)} sites (Not available: {no_fluxnet_count})")
    if fluxnet_vals:
        print(f"  Range: {min(fluxnet_vals):.1f} - {max(fluxnet_vals):.1f} mm/year")
        print(f"  Mean: {np.mean(fluxnet_vals):.1f} mm/year")

    # Three-source agreement summary
    sites_with_all_three = len(all_three_source_comparisons) - empty_cccs_count - no_fluxnet_count
    print(f"\nThree-Source Agreement:")
    if sites_with_all_three > 0:
        print(f"  Sites with all three sources: {sites_with_all_three}")
        print(f"  Good agreement (±10%): {good_agreement_count}/{sites_with_all_three}")
        if good_agreement_count > 0:
            agreement_percent = (good_agreement_count / sites_with_all_three * 100)
            print(f"  Agreement rate: {agreement_percent:.1f}%")

    # Pairwise comparisons
    print(f"\n{'-' * 100}")
    print(f"Pairwise Source Comparisons:")
    print(f"{'-' * 100}")

    # GEE vs Copernicus
    if len(gee_vals) > 0 and len(cccs_vals) > 0:
        gee_cccs_ratios = []
        gee_cccs_diffs = []
        for gee_val, cccs_val in zip(gee_vals, cccs_vals):
            ratio = gee_val / cccs_val if cccs_val != 0 else 0
            diff = abs(gee_val - cccs_val) / cccs_val * 100 if cccs_val != 0 else 0
            gee_cccs_ratios.append(ratio)
            gee_cccs_diffs.append(diff)

        print(f"\nGEE vs Copernicus ({len(gee_cccs_ratios)} sites):")
        print(f"  Mean ratio (GEE/Copernicus): {np.mean(gee_cccs_ratios):.2f}x")
        print(f"  Mean difference: {np.mean(gee_cccs_diffs):.1f}%")
        print(f"  Max difference:  {max(gee_cccs_diffs):.1f}%")

    # GEE vs FLUXNET (already computed above)
    if len(precip_comparisons) > 0:
        print(f"\nGEE vs FLUXNET ({len(precip_comparisons)} sites):")
        all_ratios = [x[3] for x in precip_comparisons]
        all_diffs = [x[4] for x in precip_comparisons]
        print(f"  Mean ratio (FLUXNET/GEE): {np.mean(all_ratios):.2f}x")
        print(f"  Mean difference: {np.mean(all_diffs):.1f}%")
        print(f"  Max difference:  {max(all_diffs):.1f}%")

    # Copernicus vs FLUXNET
    cccs_fluxnet_pairs = [(cccs, fluxnet) for _, _, cccs, fluxnet in all_three_source_comparisons
                          if cccs is not None and not np.isnan(cccs) and fluxnet is not None]
    if len(cccs_fluxnet_pairs) > 0:
        cccs_fluxnet_ratios = []
        cccs_fluxnet_diffs = []
        for cccs_val, fluxnet_val in cccs_fluxnet_pairs:
            ratio = fluxnet_val / cccs_val if cccs_val != 0 else 0
            diff = abs(fluxnet_val - cccs_val) / cccs_val * 100 if cccs_val != 0 else 0
            cccs_fluxnet_ratios.append(ratio)
            cccs_fluxnet_diffs.append(diff)

        print(f"\nCopernicus vs FLUXNET ({len(cccs_fluxnet_pairs)} sites):")
        print(f"  Mean ratio (FLUXNET/Copernicus): {np.mean(cccs_fluxnet_ratios):.2f}x")
        print(f"  Mean difference: {np.mean(cccs_fluxnet_diffs):.1f}%")
        print(f"  Max difference:  {max(cccs_fluxnet_diffs):.1f}%")

# ====== TEMPERATURE COMPARISON ANALYSIS (same source as precipitation) ======
print(f"\n{'-' * 100}")
print("TEMPERATURE ANALYSIS (MAT) - Data source selection based on PRECIPITATION validation")
print(f"{'-' * 100}")
print("\nNOTE: Temperature uses the same data source as precipitation for each site.")
print("The source was determined by analyzing precipitation (MAP) with the following logic:")
print("  - Primary: FLUXNET (if available)")
print("  - Validation: For high-precipitation sites (>2000 mm/year), compare against Copernicus & GEE")
print("  - Fallback: Copernicus (if non-zero) then GEE")
print("This unified approach ensures consistency between temperature and precipitation data.\n")

if len(temp_comparisons) > 0:
    # Filter valid temperature pairs (both FLUXNET and Copernicus available)
    valid_temp_comparisons = [x for x in temp_comparisons if x[1] is not None and x[2] is not None and not np.isnan(x[2])]

    if len(valid_temp_comparisons) > 0:
        # Sort by difference percentage (largest differences first)
        temp_comparisons_sorted = sorted(valid_temp_comparisons, key=lambda x: (abs(x[1] - x[2]) / x[2] * 100 if x[2] != 0 else 0), reverse=True)

        print(f"\nComparing FLUXNET vs Copernicus for {len(valid_temp_comparisons)} site(s) with both sources available:\n")
        print(f"{'Site':<12} {'FLUXNET':<12} {'Copernicus':<12} {'GEE':<12} {'Diff %':<10}")
        print(f"{'-' * 100}")

        # Threshold for highlighting large differences
        large_diff_threshold = 30  # 30% difference

        for site_id, fluxnet_val, cccs_val, gee_val in temp_comparisons_sorted:
            diff_percent = abs(fluxnet_val - cccs_val) / cccs_val * 100 if cccs_val != 0 else 0
            print(f"{site_id:<12} {fluxnet_val:>10.2f}°C {cccs_val:>10.2f}°C {gee_val:>10.2f}°C {diff_percent:>8.1f}%")

        # Summary statistics
        print(f"\n{'-' * 100}")
        ratios = [x[1] / x[2] if x[2] != 0 else 0 for x in valid_temp_comparisons]
        diffs = [abs(x[1] - x[2]) / x[2] * 100 if x[2] != 0 else 0 for x in valid_temp_comparisons]

        print(f"Temperature Comparison Summary:")
        print(f"  Mean ratio (FLUXNET/Copernicus): {np.mean(ratios):.2f}x")
        print(f"  Mean difference: {np.mean(diffs):.1f}%")
        print(f"  Max difference:  {max(diffs):.1f}%")
    else:
        print("\nNo temperature data available for both FLUXNET and Copernicus.")
else:
    print("\nNo temperature data available for comparison.")

# ====== ALL THREE SOURCES TEMPERATURE COMPARISON ======
print(f"\n{'-' * 100}")
print("ALL THREE SOURCES TEMPERATURE COMPARISON (GEE, Copernicus, FLUXNET)")
print(f"{'-' * 100}")

if len(all_three_source_temp_comparisons) > 0:
    print(f"\nComparing all three temperature sources for {len(all_three_source_temp_comparisons)} site(s):\n")
    print(f"{'Site':<12} {'GEE':<12} {'Copernicus':<12} {'FLUXNET':<12} {'Selected':<12} {'Reason':<50} {'Status':<25}")
    print(f"{'-' * 185}")

    gee_temp_vals = []
    cccs_temp_vals = []
    fluxnet_temp_vals = []
    empty_cccs_temp_count = 0
    no_fluxnet_temp_count = 0
    good_temp_agreement_count = 0

    for site_id, gee_val, cccs_val, fluxnet_val in sorted(all_three_source_temp_comparisons, key=lambda x: x[0]):
        # Format values, handle None/NaN
        gee_str = f"{gee_val:.2f}°C" if gee_val is not None else "N/A"
        cccs_str = f"{cccs_val:.2f}°C" if (cccs_val is not None and not np.isnan(cccs_val)) else "EMPTY"
        fluxnet_str = f"{fluxnet_val:.2f}°C" if fluxnet_val is not None else "N/A"

        # Get selected source from dataframe (same as precipitation source)
        selected_source = datasets_df.loc[datasets_df['SITE'] == site_id, 'ERA5_MAT_SOURCE']
        if len(selected_source) > 0:
            selected = selected_source.values[0]
        else:
            selected = "?"

        # Generate reason for selection (same as precipitation)
        reason = ""
        if site_id in outlier_cases:
            case = outlier_cases[site_id]
            if case == 'FLUXNET_used':
                reason = "FLUXNET: Copernicus within ±300mm (validation passed)"
            elif case == 'FLUXNET_lower':
                reason = "FLUXNET: Copernicus higher (prefer lower value)"
            elif case == 'Copernicus_fallback':
                reason = "Copernicus: precip validation failed (fallback)"
            elif case == 'Copernicus_zero_rejected':
                reason = "GEE: Copernicus zero (unrealistic, rejected)"
            elif case == 'GEE_higher_use_FLUXNET':
                reason = "FLUXNET: GEE anomalously high (prefer lower)"
            elif case == 'GEE_fallback':
                reason = "GEE: Copernicus unavailable (fallback)"
        elif fluxnet_val is not None:
            reason = "FLUXNET available (normal precipitation ≤2000 mm/year)"
        elif cccs_val is not None and not np.isnan(cccs_val):
            if cccs_val == 0:
                reason = "GEE: Copernicus zero (unrealistic, rejected)"
            else:
                reason = "Copernicus: FLUXNET not available (fallback)"
        else:
            reason = "GEE: Only valid source available"

        # Determine status
        if cccs_val is None or np.isnan(cccs_val):
            status = "[CCCS EMPTY]"
            empty_cccs_temp_count += 1
        elif fluxnet_val is None:
            status = "[No FLUXNET]"
            no_fluxnet_temp_count += 1
        else:
            # All three sources available - check agreement within ±10%
            values = [gee_val, cccs_val, fluxnet_val]
            mean_val = np.mean(values)
            min_val = min(values)
            max_val = max(values)
            range_percent = ((max_val - min_val) / mean_val * 100) if mean_val != 0 else 0

            if range_percent <= 10:
                status = "[GOOD AGREEMENT]"
                good_temp_agreement_count += 1
            else:
                status = f"[Disagree: {range_percent:.1f}%]"

        print(f"{site_id:<12} {gee_str:<12} {cccs_str:<12} {fluxnet_str:<12} {selected:<12} {reason:<50} {status:<25}")

        # Collect non-null values for statistics
        if gee_val is not None:
            gee_temp_vals.append(gee_val)
        if cccs_val is not None and not np.isnan(cccs_val):
            cccs_temp_vals.append(cccs_val)
        if fluxnet_val is not None:
            fluxnet_temp_vals.append(fluxnet_val)

    # Summary statistics
    print(f"\n{'-' * 100}")
    print(f"Temperature Source Availability and Statistics:")
    print(f"{'-' * 100}")
    print(f"GEE:")
    print(f"  Available for: {len(gee_temp_vals)}/{len(all_three_source_temp_comparisons)} sites")
    if gee_temp_vals:
        print(f"  Range: {min(gee_temp_vals):.2f} - {max(gee_temp_vals):.2f} °C")
        print(f"  Mean: {np.mean(gee_temp_vals):.2f} °C")

    print(f"\nCopernicus/CDS:")
    print(f"  Available for: {len(cccs_temp_vals)}/{len(all_three_source_temp_comparisons)} sites (Empty: {empty_cccs_temp_count})")
    if cccs_temp_vals:
        print(f"  Range: {min(cccs_temp_vals):.2f} - {max(cccs_temp_vals):.2f} °C")
        print(f"  Mean: {np.mean(cccs_temp_vals):.2f} °C")

    print(f"\nFLUXNET:")
    print(f"  Available for: {len(fluxnet_temp_vals)}/{len(all_three_source_temp_comparisons)} sites (Not available: {no_fluxnet_temp_count})")
    if fluxnet_temp_vals:
        print(f"  Range: {min(fluxnet_temp_vals):.2f} - {max(fluxnet_temp_vals):.2f} °C")
        print(f"  Mean: {np.mean(fluxnet_temp_vals):.2f} °C")

    # Three-source agreement summary
    sites_with_all_three_temp = len(all_three_source_temp_comparisons) - empty_cccs_temp_count - no_fluxnet_temp_count
    print(f"\nThree-Source Agreement (Temperature):")
    if sites_with_all_three_temp > 0:
        print(f"  Sites with all three sources: {sites_with_all_three_temp}")
        print(f"  Good agreement (±10%): {good_temp_agreement_count}/{sites_with_all_three_temp}")
        if good_temp_agreement_count > 0:
            agreement_percent = (good_temp_agreement_count / sites_with_all_three_temp * 100)
            print(f"  Agreement rate: {agreement_percent:.1f}%")

# Save to file
datasets_df = datasets_df.reset_index(drop=True)
datasets_df = datasets_df.sort_values(by=['SITE'], inplace=False)
outfile = data_path("data/outputs/10_datasets/17_datasets_info_parquet_vars_stats_usedsites_era5.csv")

print(f"\n{'-' * 80}\nSaving info about {len(datasets_df)} datasets to file {outfile}.\n{'-' * 80}")
datasets_df.to_csv(outfile, index=False)

print(f"\n{'-' * 80}")
print("SCRIPT COMPLETED SUCCESSFULLY")
print(f"{'-' * 80}")
print(f"Output CSV: {outfile}")
print(f"Log file: {log_file}")

# Close log file
sys.stdout.close()
