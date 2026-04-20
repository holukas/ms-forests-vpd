"""
Download ERA5-Land hourly climate data from CDS and aggregate to yearly values.

This is done for sites with older data only (FLUXNET2015) because they do not
have ERA5 data for 1991-2020 in their datasets.

Workflow:
1. Load site coordinates from configuration CSV
2. Download ERA5-Land timeseries data for each site (2m temperature, total precipitation)
3. Extract and merge variable files
4. Remove duplicate lat/lon columns and rename to LAT, LON
5. Convert temperature from K to °C and rename to TA_degC
6. Convert precipitation from m to mm and rename to PRECIP_TOT_mm
7. Shift timestamps back 1 hour (represent START of averaging period, not END)
8. Filter to complete calendar years (1991-2020)
9. Save shifted hourly data
10. Aggregate to yearly values:
    - TA_degC: annual mean temperature (°C)
    - PRECIP_TOT_mm: annual total precipitation (mm)

Output structure per site:
  {SITE}/
  ├── raw/                                # Original downloaded hourly CSV files
  ├── {SITE}_era5_1991-2020_shifted.csv   # Shifted hourly data (1991-2020)
  └── {SITE}_era5_1991-2020_yearly.csv    # Yearly aggregated data

Notes:
- ERA5 timestamps mark the END of the averaging period (e.g., 12:00 = hour from 11:00-12:00)
- Timestamps are shifted back 1 hour for intuitive filtering by calendar year
- Data spans 1991-01-01 00:00 to 2020-12-31 23:00 (shifted coordinates)
- Skips sites if yearly file already exists (resumable)
- All processing logged to timestamped log file in output directory
"""

import zipfile
from datetime import datetime
from pathlib import Path

import cdsapi
import pandas as pd

# Load datasets info
infile = Path('../../data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv')
datasets_df = pd.read_csv(infile)

dataset = "reanalysis-era5-land-timeseries"

# Create output directory
output_dir = Path('../../data/outputs/10_datasets/16_ERA5_climate_1991-2020/')
output_dir.mkdir(parents=True, exist_ok=True)

# Create log file
log_file = output_dir / f"16_download_era5_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


def log_message(msg):
    """Print and log message"""
    print(msg)
    with open(log_file, 'a') as f:
        f.write(msg + '\n')


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
    downloaded_via = row['DOWNLOADED_VIA']

    # Create site-specific subfolder
    site_dir = output_dir / site_id
    merged_file = site_dir / f"{site_id}_era5_1991-2020_yearly.csv"

    # Skip if yearly file already exists
    if merged_file.exists():
        log_message(f"Skipping {site_id} (yearly file already exists)")
        continue

    site_dir.mkdir(parents=True, exist_ok=True)

    # Files from FLUXNET_ORG do not have ERA5 data 1991-2020

    if downloaded_via == 'FLUXNET_ORG':
        # Round coordinates to nearest 0.1° (CDS API requirement)
        lon_rounded = round(row['LON'] * 10) / 10
        lat_rounded = round(row['LAT'] * 10) / 10

        request = {
            "variable": [
                "2m_temperature",
                "total_precipitation"
            ],
            "location": {"longitude": lon_rounded, "latitude": lat_rounded},
            "date": ["1991-01-01/2021-01-01"],  # Timestamp is end of averaging interval
            "data_format": "csv"
        }

        log_message(f"Downloading ERA5 data for {site_id} ({lat_rounded}, {lon_rounded})...")
        result = client.retrieve(dataset, request)

        target_file = site_dir / f"{site_id}_era5_1991-2020.zip"
        result.download(str(target_file))

        # Extract ZIP file to site subfolder
        print(f"Extracting ZIP file for {site_id}...")
        with zipfile.ZipFile(target_file, 'r') as zip_ref:
            zip_ref.extractall(site_dir)
        target_file.unlink()

        # Merge the two CSV files
        csv_files = list(site_dir.glob('*.csv'))
        if len(csv_files) >= 2:
            log_message(f"Merging {len(csv_files)} CSV files for {site_id}...")

            # Read both files
            dfs = [pd.read_csv(f) for f in csv_files]

            # Merge on timestamp (first column is timestamp)
            merged_df = dfs[0].copy()
            for df in dfs[1:]:
                merge_col = df.columns[0]
                merged_df = pd.merge(merged_df, df, on=merge_col, how='outer')

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

            # Parse timestamp
            timestamp_col = merged_df.columns[0]
            merged_df[timestamp_col] = pd.to_datetime(merged_df[timestamp_col])

            # Subtract 1 hour to shift timestamps from END to START of averaging period
            # (makes filtering by calendar year more intuitive)
            merged_df[timestamp_col] = merged_df[timestamp_col] - pd.Timedelta(hours=1)
            print(f"Shifted timestamps back by 1 hour (now represent START of averaging period)")

            # Filter to complete years: 1991-01-01 00:00 to 2020-12-31 23:00
            start_time = pd.Timestamp('1991-01-01 00:00')
            end_time = pd.Timestamp('2020-12-31 23:00')
            merged_df_clean = merged_df[
                (merged_df[timestamp_col] >= start_time) &
                (merged_df[timestamp_col] <= end_time)
                ].copy()
            print(f"Filtered to years 1991-2020: {len(merged_df_clean)} rows")

            # Save shifted and filtered data
            shifted_file = site_dir / f"{site_id}_era5_1991-2020_shifted.csv"
            merged_df_clean.to_csv(shifted_file, index=False)
            print(f"Saved shifted file to {shifted_file}")

            # Set timestamp as index for aggregation
            merged_df_clean.set_index(timestamp_col, inplace=True)

            agg_dict = {}
            for col in merged_df_clean.columns:
                if col in ['LAT', 'LON']:
                    agg_dict[col] = 'first'  # Keep lat/lon as-is
                elif col == 'TA_degC':
                    agg_dict[col] = 'mean'  # Annual mean temperature
                elif col == 'PRECIP_TOT_mm':
                    agg_dict[col] = 'sum'  # Annual total precipitation
                else:
                    agg_dict[col] = 'mean'  # Default to mean for other variables

            print(f"Aggregating to yearly values...")
            yearly_df = merged_df_clean.resample('YS').agg(agg_dict)
            yearly_df.reset_index(inplace=True)

            # Save aggregated file
            yearly_df.to_csv(merged_file, index=False)
            print(f"Saved yearly aggregated file to {merged_file}")

            # Organize original CSV files into a subfolder
            raw_dir = site_dir / 'raw'
            raw_dir.mkdir(exist_ok=True)
            for f in csv_files:
                f.rename(raw_dir / f.name)
            print(f"Organized original CSV files in {raw_dir}")
        else:
            log_message(f"Found {len(csv_files)} CSV files, expected at least 2")

        log_message(f"Completed {site_id}\n")

# Final summary
log_message(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_message(f"Log saved to: {log_file}")

# # Save to file
# datasets_df = datasets_df.reset_index(drop=True)
# datasets_df = datasets_df.sort_values(by=['SITE'], inplace=False)
# outfile = Path('../../data/outputs/10_datasets/17_datasets_info_parquet_vars_stats_usedsites_era5.csv')
#
# print(f"\n{'-' * 80}\nSaving info about {len(datasets_df)} datasets to file {outfile}.\n{'-' * 80}")
# datasets_df.to_csv(outfile, index=False)
