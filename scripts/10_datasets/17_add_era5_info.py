from pathlib import Path
import pandas as pd
import numpy as np

# Load datasets info
infile = Path('../../data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv')
datasets_df = pd.read_csv(infile)

# Output folder for ERA5 data for each site
dir_era5 = Path('../../data/outputs/10_datasets/16_ERA5_climate_1991-2020')

# Initialize new columns to store the 30-year averages
datasets_df['ERA5_MAT_1991_2020'] = np.nan
datasets_df['ERA5_MAP_1991_2020'] = np.nan

print("Calculating 30-year MAT and MAP averages for each site...")

# Cycle through the sites and read their corresponding ERA5 CSVs
for index, row in datasets_df.iterrows():
    site_id = row['SITE']
    site_csv = dir_era5 / f"{site_id}_ERA5_Yearly_Climate_1991-2020.csv"

    if site_csv.is_file():
        # Read the yearly data
        df_site = pd.read_csv(site_csv)

        # Calculate the 30-year mean for MAT and MAP, rounded to 3 decimal places
        mean_mat = round(df_site['MAT'].mean(), 3)
        mean_map = round(df_site['MAP'].mean(), 3)

        # Assign the values back to the main dataframe
        datasets_df.at[index, 'ERA5_MAT_1991_2020'] = mean_mat
        datasets_df.at[index, 'ERA5_MAP_1991_2020'] = mean_map
    else:
        print(f" -> Warning: No ERA5 data file found for {site_id}. Leaving as NaN.")

# Save to file
datasets_df = datasets_df.reset_index(drop=True)
datasets_df = datasets_df.sort_values(by=['SITE'], inplace=False)
outfile = Path('../../data/outputs/10_datasets/17_datasets_info_parquet_vars_stats_usedsites_era5.csv')

print(f"\n{'-' * 80}\nSaving info about {len(datasets_df)} datasets to file {outfile}.\n{'-' * 80}")
datasets_df.to_csv(outfile, index=False)