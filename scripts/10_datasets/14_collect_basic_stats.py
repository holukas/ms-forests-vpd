"""
Record counts, date ranges and basic statistics per site and variable.

Reads `13_datasets_info_parquet_vars.csv`. Writes `14_datasets_info_parquet_vars_stats.csv`.
"""
from pathlib import Path

import pandas as pd

import src.stats as stats
from src.paths import data_path

# Load datasets info
infile = data_path("data/outputs/10_datasets/13_datasets_info_parquet_vars.csv")
datasets_df = pd.read_csv(infile)

# Calculate basic stats
_datasets_df = datasets_df.copy()
for ix, siteconfig in _datasets_df.iterrows():
    # # TODO testing
    # if ix != 1:
    #     continue
    # # TODO testing

    datasets_df = stats.basic_stats(
        siteinfo_df=datasets_df,
        siteconfig=siteconfig,
        ix=ix)

# Save to file
datasets_df = datasets_df.reset_index(drop=True)
datasets_df = datasets_df.sort_values(by=['SITE'], inplace=False)
outfile = data_path("data/outputs/10_datasets/14_datasets_info_parquet_vars_stats.csv")
print(f"\n{'-' * 80}\nSaving info about {len(datasets_df)} datasets to file {outfile}.\n{'-' * 80}")
datasets_df.to_csv(outfile, index=False)
