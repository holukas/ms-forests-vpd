"""
Script: ERA5 Climate Data Extraction via Google Earth Engine

Description:
    This script leverages the Google Earth Engine (GEE) Python API to efficiently process
    and extract 30 years (1991-2020) of historical climate data for specific study sites.
    By querying the 'ECMWF/ERA5/DAILY' dataset, it calculates Mean Annual Temperature
    (MAT, converted to °C) and Mean Annual Precipitation (MAP, converted to mm) directly
    on Google's cloud servers. This approach extracts aggregated yearly summaries for
    specific coordinates without the need to download massive raw NetCDF files locally.

Workflow:
    1. Authenticates and initializes the GEE API using a designated Cloud Project ID.
    2. Reads target site coordinates (Latitude/Longitude) from a local dataset info CSV.
    3. Maps over the 1991-2020 timeframe, aggregating daily temperature and precipitation.
    4. Downloads the lightweight summary dictionary and formats it into a Pandas DataFrame.
    5. Validates 30-year period (1991-2020) with no gaps.
    6. Saves the final 30-year summary for each site as an individual CSV.
    * Includes robust error handling, automatic retries, skips already processed sites.
    * Comprehensive logging: batch log, errors log, warnings log, no-data log.

Inputs:
    - Target sites: ../../data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv
      (Requires 'SITE', 'LAT', and 'LON' columns)

Outputs:
    - Individual CSV files containing 'Year', 'MAT_degC', and 'PRECIP_TOT_mm'.
    - Saved to: ../../data/outputs/10_datasets/16_ERA5_climate_1991-2020_GoogleEarthEngine/
    - Log files: ERRORS.log, WARNINGS.log, NO_DATA_SITES.log

Dependencies:
    - ee (earthengine-api), pandas
    - Requires an active Google Cloud Project and one-time local GEE authentication.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import ee
import pandas as pd

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

# 1. Initialize the Google Earth Engine API
GEE_PROJECT_ID = 'soy-blend-488522-g9'

try:
    # Try initializing with the specific project
    ee.Initialize(project=GEE_PROJECT_ID)
    print("Earth Engine initialized successfully!")
except Exception as e:
    print("Earth Engine credentials not found. Opening browser to authenticate...")
    # This will pop up a browser window for you to log in and generate a token
    ee.Authenticate()
    # Once authenticated, initialize again
    ee.Initialize(project=GEE_PROJECT_ID)

# Load datasets info
infile = Path('../../data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv')
datasets_df = pd.read_csv(infile)

# Filter to batch
if batch_end is not None:
    datasets_df = datasets_df.iloc[batch_start:batch_end].reset_index(drop=True)

total_sites = len(datasets_df)
print(f"Processing {total_sites} sites (indices {batch_start} to {batch_start + total_sites - 1})\n")

# Output folder for ERA5 data for each site
dirout_era5 = Path('../../data/outputs/10_datasets/16_ERA5_climate_1991-2020_GoogleEarthEngine')
dirout_era5.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

# Create overall summary log files (shared across runs)
overall_errors_log = dirout_era5 / "16_download_era5_ERRORS.log"
overall_warnings_log = dirout_era5 / "16_download_era5_WARNINGS.log"
overall_nodata_log = dirout_era5 / "16_download_era5_NO_DATA_SITES.log"

# Create batch-specific log file
batch_log_file = dirout_era5 / f"16_download_era5_gee_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


def log_message(msg):
    """Print and log message to batch log"""
    print(msg)
    with open(batch_log_file, 'a', encoding='utf-8') as f:
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
log_message(f"=== ERA5 Download (Google Earth Engine) Log ===")
log_message(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_message(f"Output directory: {dirout_era5}\n")

site_id_col = 'SITE'
lat_col = 'LAT'
lon_col = 'LON'


# 2. Define the GEE extraction function
def get_gee_yearly_climate(lon, lat):
    """
    Tells Google Earth Engine to aggregate ERA5 daily data to yearly
    values for a specific point and return the summary.
    """
    point = ee.Geometry.Point([lon, lat])
    years = ee.List.sequence(1991, 2020)

    def process_year(year):
        start = ee.Date.fromYMD(year, 1, 1)
        end = start.advance(1, 'year')

        # Filter ERA5 daily data for the year
        year_data = ee.ImageCollection("ECMWF/ERA5/DAILY").filterDate(start, end)

        # Calculate MAT: Mean temperature, convert Kelvin to Celsius
        mean_temp = year_data.select('mean_2m_air_temperature').mean().subtract(273.15)

        # Calculate MAP: Sum precipitation, convert meters to mm
        sum_precip = year_data.select('total_precipitation').sum().multiply(1000)

        # Combine into a single image
        yearly_img = mean_temp.addBands(sum_precip).rename(['MAT_degC', 'PRECIP_TOT_mm'])

        # Extract the value for our exact point
        # ERA5 native resolution is ~27.8km (27830 meters)
        stats = yearly_img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=27830
        )

        return ee.Feature(None, {
            'Year': year,
            'MAT_degC': stats.get('MAT_degC'),
            'PRECIP_TOT_mm': stats.get('PRECIP_TOT_mm')
        })

    # Map the function over all 30 years and fetch the data to local memory
    yearly_features = ee.FeatureCollection(years.map(process_year))
    return yearly_features.getInfo()['features']


# 3. Loop through each site and extract data
for index, row in datasets_df.iterrows():
    site_id = row[site_id_col]
    lat = row[lat_col]
    lon = row[lon_col]

    req_filepath = dirout_era5 / f'{site_id}_era5_1991-2020_yearly.csv'
    if Path(req_filepath).is_file():
        log_message(f"Skipping {site_id} (yearly file already exists)")
        continue

    log_message(f"Fetching ERA5 via GEE for {site_id} ({lat}, {lon})...")

    max_attempts = 3
    attempts = 0
    success = False

    while not success and attempts < max_attempts:
        try:
            # Fetch data from Google Earth Engine
            raw_features = get_gee_yearly_climate(lon, lat)

            # Extract properties into a list of dictionaries
            records = [feat['properties'] for feat in raw_features]

            # Build DataFrame
            df_yearly = pd.DataFrame(records)

            # Clean up the DataFrame: reorder columns and handle missing data
            df_yearly = df_yearly[['Year', 'MAT_degC', 'PRECIP_TOT_mm']]
            df_yearly = df_yearly.dropna()  # Drop years if the point is over the ocean with no data

            if df_yearly.empty:
                log_no_data(site_id, "Empty data returned (might be over water)")
                log_message(f"Skipping {site_id}\n")
                break

            # Validate 30-year period (1991-2020)
            first_year = df_yearly['Year'].min()
            last_year = df_yearly['Year'].max()
            n_years = len(df_yearly)
            years_list = sorted(df_yearly['Year'].unique().tolist())

            # Check if first year is 1991
            if first_year != 1991:
                log_error(site_id, f"First year is {first_year}, expected 1991")
                log_message(f"Skipping {site_id} (invalid year range)\n")
                break

            # Check if last year is 2020
            if last_year != 2020:
                log_error(site_id, f"Last year is {last_year}, expected 2020")
                log_message(f"Skipping {site_id} (invalid year range)\n")
                break

            # Check if exactly 30 years available
            if n_years != 30:
                log_error(site_id, f"Expected 30 years of data, found {n_years} years")
                log_message(f"Skipping {site_id} (incomplete 30-year period)\n")
                break

            # Check if all years are consecutive (no gaps)
            expected_years = list(range(1991, 2021))
            if years_list != expected_years:
                missing_years = [y for y in expected_years if y not in years_list]
                log_error(site_id, f"Missing years: {missing_years}")
                log_message(f"Skipping {site_id} (missing years)\n")
                break

            log_message(f"[OK] Yearly data validated: 1991-2020 complete ({n_years} years)")

            # Save the YEARLY data to CSV
            df_yearly.to_csv(req_filepath, index=False)

            log_message(f"[OK] Successfully saved {len(df_yearly)} yearly records for {site_id}\n")
            success = True

        except Exception as e:
            attempts += 1
            log_warning(site_id, f"GEE request failed: {e}. Retrying in 5 seconds... (Attempt {attempts}/{max_attempts})")
            time.sleep(5)

    if not success:
        log_error(site_id, f"Failed to fetch data after {max_attempts} attempts")
        log_message(f"[!] Skipping {site_id}\n")

log_message(f"\n{'='*80}")
log_message(f"Download session complete!")
log_message(f"{'='*80}")
log_message(f"Batch log saved to: {batch_log_file}")
log_message(f"\nSummary log files:")
log_message(f"  - {overall_errors_log.name}")
log_message(f"  - {overall_warnings_log.name}")
log_message(f"  - {overall_nodata_log.name}")
print("\nAll downloads complete!")
