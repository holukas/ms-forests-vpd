"""
Download hourly ERA5-Land temperature and precipitation from Copernicus CDS and aggregate to yearly values.

Per site, the script merges the two variables, drops duplicate timestamps,
converts to degC (TA_degC) and mm (PRECIP_TOT_mm), shifts the timestamps back
1 hour so that each hourly precipitation amount falls into the hour it covers
(temperature is an instantaneous value, the shift does not matter for it),
keeps 1991-2020, and computes the annual mean temperature and precipitation sum.
A yearly file is saved only if all 30 years are present. Sites with a valid
yearly file are skipped.

Sites in ERA5_LAND_POINT_OVERRIDE lie on an ERA5-Land sea cell and use the
nearest land cell. An override takes effect only for a site without a valid
yearly file: move the site folder to archive/ first, then rerun.

Usually started by 16c with a site index range (`... 0 35`); without arguments
it processes all sites. Needs a configured cdsapi client.

Reads: data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv
Writes, in data/outputs/10_datasets/16_ERA5_climate_1991-2020_Copernicus/:
- {SITE}/{SITE}_era5_1991-2020_shifted.csv (hourly)
- {SITE}/{SITE}_era5_1991-2020_yearly.csv
- {SITE}/raw/ (the downloaded CSV files)
- a batch log 16_download_era5_log_*.txt and the shared ERRORS, WARNINGS
  and NO_DATA_SITES logs
"""

import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import cdsapi
import pandas as pd
from src.paths import data_path

# Parse batch arguments
# Usage: python script.py [start_index] [end_index]
# Example: python script.py 0 60 (processes sites 0-59)
if len(sys.argv) == 3:
    batch_start = int(sys.argv[1])
    batch_end = int(sys.argv[2])
    print(f"Running batch: sites {batch_start} to {batch_end-1}")
else:
    batch_start = 0
    batch_end = None
    print("No batch specified. Running all sites.")

# Load datasets info
infile = data_path("data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv")
datasets_df = pd.read_csv(infile)

# Filter to batch
if batch_end is not None:
    datasets_df = datasets_df.iloc[batch_start:batch_end].reset_index(drop=True)

total_sites = len(datasets_df)
print(f"Processing {total_sites} sites (indices {batch_start} to {batch_start + total_sites - 1})\n")

dataset = "reanalysis-era5-land-timeseries"

# These coastal sites lie on an ERA5-Land sea cell (all-NaN temperature, zero
# precipitation). They use the nearest land cell (0.1 deg grid) instead.
# (lat, lon) of the grid cell, found by testing the neighbouring cells by distance.
ERA5_LAND_POINT_OVERRIDE = {
    "JP-Ynf": (26.7, 128.2),
    "US-HB2": (33.3, -79.3),
    "US-HB3": (33.4, -79.2),
    "VU-Coc": (-15.4, 167.1),
}

# Create output directory
output_dir = data_path("data/outputs/10_datasets/16_ERA5_climate_1991-2020_Copernicus/")
output_dir.mkdir(parents=True, exist_ok=True)

# Create log file with batch info
batch_label = f"batch_{batch_start}-{batch_end if batch_end else 'end'}" if batch_end else "all"
log_file = output_dir / f"16_download_era5_log_{batch_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Create overall summary log files (shared across batches)
overall_errors_log = output_dir / "16_download_era5_ERRORS.log"
overall_warnings_log = output_dir / "16_download_era5_WARNINGS.log"
overall_nodata_log = output_dir / "16_download_era5_NO_DATA_SITES.log"


def log_message(msg):
    """Print and log message to batch log"""
    print(msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


def log_error(site_id, error_msg):
    """Log error to both batch log and overall errors log"""
    full_msg = f"ERROR [{site_id}]: {error_msg}"
    log_message(full_msg)
    with open(overall_errors_log, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {full_msg}\n")


def log_warning(site_id, warning_msg):
    """Log warning to both batch log and overall warnings log"""
    full_msg = f"WARNING [{site_id}]: {warning_msg}"
    log_message(full_msg)
    with open(overall_warnings_log, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {full_msg}\n")


def log_no_data(site_id, reason):
    """Log site with no/empty data to both batch log and overall no-data log"""
    full_msg = f"NO DATA [{site_id}]: {reason}"
    log_message(full_msg)
    with open(overall_nodata_log, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {full_msg}\n")


# Initialize log
log_message(f"=== ERA5 Download Log ===")
log_message(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_message(f"Output directory: {output_dir}\n")

client = cdsapi.Client()

for index, row in datasets_df.iterrows():

    # # TODO TESTING
    # if index > 0:
    #     break
    # # TODO TESTING

    site_id = row['SITE']

    # Create site-specific subfolder
    site_dir = output_dir / site_id
    merged_file = site_dir / f"{site_id}_era5_1991-2020_yearly.csv"

    # Skip if yearly file already exists (with validation)
    if merged_file.exists():
        try:
            existing_df = pd.read_csv(merged_file)

            # Check if required columns exist
            required_cols = ['TA_degC', 'PRECIP_TOT_mm']
            missing_cols = [col for col in required_cols if col not in existing_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Check if columns have valid data
            if existing_df['TA_degC'].isna().all():
                raise ValueError("TA_degC column is empty (all NaN)")
            if existing_df['PRECIP_TOT_mm'].isna().all():
                raise ValueError("PRECIP_TOT_mm column is empty (all NaN)")

            # Check if file has expected 30 years of data
            if len(existing_df) != 30:
                raise ValueError(f"Expected 30 years of data, found {len(existing_df)}")

            log_message(f"Skipping {site_id} (valid yearly file already exists with {len(existing_df)} records)")
            continue

        except Exception as e:
            log_message(f"WARNING: Existing file for {site_id} is invalid ({str(e)}). Re-downloading...")
            # Continue to re-download if validation fails

    site_dir.mkdir(parents=True, exist_ok=True)

    # Every site is downloaded, because script 17 reads the ERA5-Land file of every site
    lon = row['LON']
    lat = row['LAT']
    if site_id in ERA5_LAND_POINT_OVERRIDE:
        lat, lon = ERA5_LAND_POINT_OVERRIDE[site_id]
        log_message(f"{site_id}: site is on an ERA5-Land sea cell, using nearest land cell ({lat}, {lon})")

    request = {
        "variable": [
            "2m_temperature",
            "total_precipitation"
        ],
        "location": {"longitude": lon, "latitude": lat},
        "date": ["1991-01-01/2021-01-01"],  # Timestamp is end of averaging interval
        "data_format": "csv"
    }

    log_message(f"Downloading ERA5 data for {site_id} ({lat}, {lon})...")
    log_message(f"Request: date range {request['date'][0]}, variables: {request['variable']}")
    result = client.retrieve(dataset, request)

    target_file = site_dir / f"{site_id}_era5_1991-2020.zip"
    result.download(str(target_file))

    # Check if ZIP file has content
    file_size = os.path.getsize(target_file)
    log_message(f"Downloaded ZIP file size: {file_size} bytes")

    # Extract ZIP file to site subfolder
    print(f"Extracting ZIP file for {site_id}...")
    with zipfile.ZipFile(target_file, 'r') as zip_ref:
        zip_ref.extractall(site_dir)
    target_file.unlink()

    # Merge the two downloaded CSV files (not the _shifted or _yearly output of an earlier run)
    csv_files = list(site_dir.glob(f'{dataset}-*.csv'))
    if len(csv_files) >= 2:
        log_message(f"Merging {len(csv_files)} CSV files for {site_id}...")

        # Read both files
        dfs = [pd.read_csv(f) for f in csv_files]

        # Check if CSV files have data
        has_data = True
        for i, df in enumerate(dfs):
            if len(df) == 0:
                log_no_data(site_id, f"CSV file {i+1} is empty")
                has_data = False
                break

        if not has_data:
            log_message(f"Skipping {site_id} (no data from ERA5)\n")
            continue

        # Merge on timestamp (first column is timestamp)
        merged_df = dfs[0].copy()
        for df in dfs[1:]:
            merge_col = df.columns[0]
            merged_df = pd.merge(merged_df, df, on=merge_col, how='outer')

        # Check if merged data is empty or all NaN
        if len(merged_df) == 0:
            log_no_data(site_id, "Merged data is empty")
            log_message(f"Skipping {site_id} (no merged data)\n")
            continue

        # Check for duplicate rows and remove them (keep first occurrence)
        timestamp_col = merged_df.columns[0]
        duplicates = merged_df[merged_df.duplicated(subset=[timestamp_col], keep=False)]
        if len(duplicates) > 0:
            log_warning(site_id, f"Found {len(duplicates)} duplicate timestamp rows. Removing duplicates (keeping first occurrence)...")
            merged_df = merged_df.drop_duplicates(subset=[timestamp_col], keep='first')
            log_message(f"After removing duplicates: {len(merged_df)} rows")
        else:
            log_message(f"[OK] No duplicate rows found in merged data")

        # Remove duplicate lat/lon columns (keep first occurrence) and standardize names
        cols_to_keep = []
        lat_found = False
        lon_found = False
        rename_dict = {}

        for col in merged_df.columns:
            col_lower = col.lower()
            # Match latitude/lat (including suffixed versions like latitude_x, lat_y)
            if 'latitude' in col_lower or (col_lower.startswith('lat') and not col_lower.startswith('longitude')):
                if not lat_found:
                    cols_to_keep.append(col)
                    rename_dict[col] = 'LAT'
                    lat_found = True
            # Match longitude/lon (including suffixed versions like longitude_x, lon_y)
            elif 'longitude' in col_lower or col_lower.startswith('lon'):
                if not lon_found:
                    cols_to_keep.append(col)
                    rename_dict[col] = 'LON'
                    lon_found = True
            else:
                cols_to_keep.append(col)

        merged_df = merged_df[cols_to_keep]
        merged_df.rename(columns=rename_dict, inplace=True)
        print(f"Removed duplicate lat/lon columns and renamed to LAT, LON")

        # Convert temperature from K to C (t2m column)
        if 't2m' in merged_df.columns:
            merged_df['t2m'] = merged_df['t2m'] - 273.15
            merged_df.rename(columns={'t2m': 'TA_degC'}, inplace=True)
            print(f"Converted t2m from K to °C and renamed to TA_degC")

        # Convert precipitation from m to mm (tp column)
        if 'tp' in merged_df.columns:
            merged_df['tp'] = merged_df['tp'] * 1000
            merged_df.rename(columns={'tp': 'PRECIP_TOT_mm'}, inplace=True)
            print(f"Converted tp from m to mm and renamed to PRECIP_TOT_mm")

        # Parse timestamp (handle mixed formats with format='mixed')
        timestamp_col = merged_df.columns[0]
        merged_df[timestamp_col] = pd.to_datetime(merged_df[timestamp_col], format='mixed')

        # Subtract 1 hour: the precipitation of an hour is stamped at its end, so the
        # amount of 23:00-24:00 on 31 December belongs to that year
        merged_df[timestamp_col] = merged_df[timestamp_col] - pd.Timedelta(hours=1)
        print(f"Shifted timestamps back by 1 hour (precipitation now stamped at the start of its hour)")

        # Filter to complete years: 1991-01-01 00:00 to 2020-12-31 23:00
        start_time = pd.Timestamp('1991-01-01 00:00')
        end_time = pd.Timestamp('2020-12-31 23:00')
        merged_df_clean = merged_df[
            (merged_df[timestamp_col] >= start_time) &
            (merged_df[timestamp_col] <= end_time)
            ].copy()
        print(f"Filtered to years 1991-2020: {len(merged_df_clean)} rows")

        # Check if filtered data is empty
        if len(merged_df_clean) == 0:
            log_no_data(site_id, "No data in 1991-2020 time range")
            log_message(f"Skipping {site_id} (no data in target period)\n")
            continue

        # Save shifted and filtered data
        shifted_file = site_dir / f"{site_id}_era5_1991-2020_shifted.csv"
        merged_df_clean.to_csv(shifted_file, index=False)
        print(f"Saved shifted file to {shifted_file}")

        # Set timestamp as index for aggregation
        merged_df_clean.set_index(timestamp_col, inplace=True)

        # Build aggregation dictionary only for expected columns
        agg_dict = {}
        for col in merged_df_clean.columns:
            if col in ['LAT', 'LON']:
                agg_dict[col] = 'first'  # Keep lat/lon as-is
            elif col == 'TA_degC':
                agg_dict[col] = 'mean'  # Annual mean temperature
            elif col == 'PRECIP_TOT_mm':
                agg_dict[col] = 'sum'  # Annual total precipitation
            # Skip any unexpected columns (from merge operations, etc.)

        # Ensure required columns exist before aggregating
        required_cols = ['TA_degC', 'PRECIP_TOT_mm']
        missing_cols = [col for col in required_cols if col not in agg_dict]
        if missing_cols:
            log_error(site_id, f"Missing required columns for aggregation: {missing_cols}")
            continue

        print(f"Aggregating to yearly values...")
        yearly_df = merged_df_clean.resample('YS').agg(agg_dict)
        yearly_df.reset_index(inplace=True)

        # Validate 30-year period (1991-2020)
        timestamp_col = yearly_df.columns[0]
        yearly_df['year'] = pd.to_datetime(yearly_df[timestamp_col]).dt.year

        first_year = yearly_df['year'].min()
        last_year = yearly_df['year'].max()
        n_years = len(yearly_df)
        years_list = sorted(yearly_df['year'].unique().tolist())

        # Check if first year is 1991
        if first_year != 1991:
            log_error(site_id, f"First year is {first_year}, expected 1991")
            log_message(f"Skipping {site_id} (invalid year range)\n")
            continue

        # Check if last year is 2020
        if last_year != 2020:
            log_error(site_id, f"Last year is {last_year}, expected 2020")
            log_message(f"Skipping {site_id} (invalid year range)\n")
            continue

        # Check if exactly 30 years available
        if n_years != 30:
            log_error(site_id, f"Expected 30 years of data, found {n_years} years")
            log_message(f"Skipping {site_id} (incomplete 30-year period)\n")
            continue

        # Check if all years are consecutive (no gaps)
        expected_years = list(range(1991, 2021))
        if years_list != expected_years:
            missing_years = [y for y in expected_years if y not in years_list]
            log_error(site_id, f"Missing years: {missing_years}")
            log_message(f"Skipping {site_id} (missing years)\n")
            continue

        log_message(f"[OK] Yearly data validated: 1991-2020 complete ({n_years} years)")

        # Remove temporary year column before saving
        yearly_df = yearly_df.drop('year', axis=1)

        # Save aggregated file
        yearly_df.to_csv(merged_file, index=False)
        print(f"Saved yearly aggregated file to {merged_file}")

        # Organize original CSV files into a subfolder
        raw_dir = site_dir / 'raw'
        raw_dir.mkdir(exist_ok=True)
        for f in csv_files:
            target = raw_dir / f.name
            # If file already exists in raw/ (from previous run), remove the source file
            if target.exists():
                f.unlink()
            else:
                f.rename(target)
        print(f"Organized original CSV files in {raw_dir}")
    else:
        log_message(f"Found {len(csv_files)} CSV files, expected at least 2")

    log_message(f"Completed {site_id}\n")

# Final summary
log_message(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_message(f"Log saved to: {log_file}")
