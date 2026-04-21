"""
Add ERA5 Climate Data to FLUXNET Site Information

This script aggregates 30-year average (1991-2020) Mean Annual Temperature (MAT)
and Mean Annual Precipitation (MAP) from ERA5 reanalysis data for each FLUXNET site.

Input:
    - 15_datasets_info_parquet_vars_stats_usedsites.csv
      Contains site metadata including SITE, DOWNLOADED_VIA, and _DIRPATH

Output:
    - 17_datasets_info_parquet_vars_stats_usedsites_era5.csv
      Original site info plus ERA5_MAT_1991_2020 and ERA5_MAP_1991_2020 columns
    - 17_add_era5_info_YYYYMMDD_HHMMSS.log
      Complete execution log with all processing details

ERA5 Data Sources:
    1. SHUTTLE-CLI & AMERIFLUX:
       - ERA5 data embedded in downloaded FLUXNET files
       - Pattern: *_{SITE}_FLUXNET_ERA5_YY_*.csv
       - Columns: TA_ERA (temperature), P_ERA (precipitation)
       - Data range: 1981-2024, filtered to 1991-2020

    2. FLUXNET_ORG:
       - Manually downloaded ERA5 data (separate files)
       - Path: data/outputs/10_datasets/16_ERA5_climate_1991-2020/{SITE}/
       - File: {SITE}_era5_1991-2020_yearly.csv
       - Columns: TA_degC (temperature), PRECIP_TOT_mm (precipitation)
       - Data range: 1991-2020 (pre-filtered)

Validation:
    - Each source must have exactly 30 years of data (1991-2020)
    - All sites must have ERA5 data before proceeding
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

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.log.flush()


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

print("Calculating 30-year MAT and MAP averages for each site...")

dir_era5 = None
file_era5 = None
tacol = None
precipcol = None

# Cycle through the sites and read their corresponding ERA5 CSVs
for index, row in datasets_df.iterrows():
    site_id = row['SITE']
    downloaded_via = row['DOWNLOADED_VIA']

    # SHUTTLE-CLI and AMERIFLUX: have ERA5 already available
    source = ['SHUTTLE-CLI', 'AMERIFLUX']

    if downloaded_via in source:
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
        mean_mat = round(df_site_1991_2020[tacol].mean(), 3)
        mean_map = round(df_site_1991_2020[precipcol].mean(), 3)


    # ERA5 from manual download
    # Files already have the correct range 1991-2020
    elif downloaded_via == 'FLUXNET_ORG':
        dir_era5 = rf"..\..\data\outputs\10_datasets\16_ERA5_climate_1991-2020\{site_id}"
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
        mean_mat = round(df_site[tacol].mean(), 3)
        mean_map = round(df_site[precipcol].mean(), 3)

    else:
        raise Exception(f"Unknown downloaded_via value: {downloaded_via}")

    # Assign the values back to the main dataframe
    datasets_df.at[index, 'ERA5_MAT_1991_2020'] = mean_mat
    datasets_df.at[index, 'ERA5_MAP_1991_2020'] = mean_map
    print(f"  {site_id} ({downloaded_via}): MAT={mean_mat}, MAP={mean_map}")

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
    print(f"\n✓ SUCCESS: All {len(datasets_df)} sites have ERA5 data (MAT and MAP)")
    print(
        f"  MAT range: {datasets_df['ERA5_MAT_1991_2020'].min():.1f} to {datasets_df['ERA5_MAT_1991_2020'].max():.1f} degC")
    print(
        f"  MAP range: {datasets_df['ERA5_MAP_1991_2020'].min():.1f} to {datasets_df['ERA5_MAP_1991_2020'].max():.1f} mm/year")

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
sys.stdout.log.close()
