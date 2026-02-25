import time
from pathlib import Path

import ee
import pandas as pd

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

# Output folder for ERA5 data for each site
dirout_era5 = Path('../../data/outputs/10_datasets/16_ERA5_climate_1991-2020')
dirout_era5.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

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
        yearly_img = mean_temp.addBands(sum_precip).rename(['MAT', 'MAP'])

        # Extract the value for our exact point
        # ERA5 native resolution is ~27.8km (27830 meters)
        stats = yearly_img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=27830
        )

        return ee.Feature(None, {
            'Year': year,
            'MAT': stats.get('MAT'),
            'MAP': stats.get('MAP')
        })

    # Map the function over all 30 years and fetch the data to local memory
    yearly_features = ee.FeatureCollection(years.map(process_year))
    return yearly_features.getInfo()['features']


# 3. Loop through each site and extract data
for index, row in datasets_df.iterrows():
    site_id = row[site_id_col]
    lat = row[lat_col]
    lon = row[lon_col]

    req_filepath = dirout_era5 / f'{site_id}_ERA5_Yearly_Climate_1991-2020.csv'
    if Path(req_filepath).is_file():
        print(f"Skipping {site_id} because file {req_filepath} already exists")
        continue

    print(f"Fetching ERA5 via GEE for {site_id} ({lat}, {lon})...")

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
            df_yearly = df_yearly[['Year', 'MAT', 'MAP']]
            df_yearly = df_yearly.dropna()  # Drop years if the point is over the ocean with no data

            if df_yearly.empty:
                print(f" -> No data returned for {site_id} (might be over water). Skipping.")
                break

            # Save the YEARLY data to CSV
            df_yearly.to_csv(req_filepath, index=False)

            print(f" -> Successfully saved {len(df_yearly)} yearly records for {site_id}")
            success = True

        except Exception as e:
            attempts += 1
            print(f" -> GEE request failed: {e}. Retrying in 5 seconds... (Attempt {attempts}/{max_attempts})")
            time.sleep(5)

    if not success:
        print(f" -> [!] Completely failed to fetch data for {site_id} after {max_attempts} attempts.")

print("\nAll downloads complete!")