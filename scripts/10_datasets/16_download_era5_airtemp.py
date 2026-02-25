import time
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Load datasets info
infile = Path('../../data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv')
datasets_df = pd.read_csv(infile)

# Output folder for ERA5 data for each site
dirout_era5 = Path('../../data/outputs/10_datasets/16_ERA5_climate_1991-2020')
dirout_era5.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

# 1. Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

site_id_col = 'SITE'
lat_col = 'LAT'
lon_col = 'LON'

url = "https://archive-api.open-meteo.com/v1/archive"

# 2. Loop through each site and download ERA5 data
for index, row in datasets_df.iterrows():
    if index < 64:
        continue
    site_id = row[site_id_col]
    lat = row[lat_col]
    lon = row[lon_col]

    print(f"Fetching ERA5 Daily Air Temp & Precip for {site_id} ({lat}, {lon})...")

    # Switched 'hourly' to 'daily' and updated the variable names
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "1991-01-01",
        "end_date": "2020-12-31",
        "daily": ["temperature_2m_mean", "precipitation_sum"],
        "models": "era5",
        "timezone": "UTC"
    }

    success = False
    while not success:
        try:
            # Fetch data
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]

            # Process DAILY data (changed from Hourly)
            daily = response.Daily()
            daily_temperature_2m = daily.Variables(0).ValuesAsNumpy()
            daily_precipitation = daily.Variables(1).ValuesAsNumpy()

            # Create a datetime index for daily data
            daily_data = {"date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            )}

            # Build the daily dataframe
            df_era5_daily = pd.DataFrame(data=daily_data)
            df_era5_daily["ERA5_TA_2m_mean"] = daily_temperature_2m
            df_era5_daily["ERA5_PRECIP_sum"] = daily_precipitation

            # --- AGGREGATE TO YEARLY ---
            # Group by year: Mean for temperature, Sum for precipitation
            df_yearly = df_era5_daily.groupby(df_era5_daily['date'].dt.year).agg(
                MAT=('ERA5_TA_2m_mean', 'mean'),
                MAP=('ERA5_PRECIP_sum', 'sum')
            ).reset_index()
            df_yearly.rename(columns={'date': 'Year'}, inplace=True)

            # Save the YEARLY data to CSV
            outpath = dirout_era5 / f"{site_id}_ERA5_Yearly_Climate_1991-2020.csv"
            df_yearly.to_csv(outpath, index=False)

            print(f" -> Aggregated to {len(df_yearly)} yearly records for {site_id}")

            success = True
            time.sleep(5)  # Can be shorter now since daily requests are lighter

            # datasets_df.loc[datasets_df['SITE'] == site_id, 'MAT_ERA5_1991-2020'] = df_yearly['MAT'].mean()
            # datasets_df.loc[datasets_df['SITE'] == site_id, 'MAP_ERA5_1991-2020'] = df_yearly['MAP'].mean()

        except Exception as e:
            error_msg = str(e)
            if "Minutely API request limit" in error_msg:
                print(" -> Rate limit hit. Sleeping for 20 seconds before retrying...")
                time.sleep(20)
            else:
                print(f" -> Failed to fetch data for {site_id}: {e}")
                break

print("\nAll downloads complete!")

# # Save to file
# datasets_df = datasets_df.reset_index(drop=True)
# datasets_df = datasets_df.sort_values(by=['SITE'], inplace=False)
# outfile = Path('../../data/outputs/10_datasets/16_datasets_info_parquet_vars_stats_usedsites_era5.csv')
# print(f"\n{'-' * 80}\nSaving info about {len(datasets_df)} datasets to file {outfile}.\n{'-' * 80}")
# datasets_df.to_csv(outfile, index=False)
