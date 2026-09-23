"""
Merge the datasets of each site into one parquet file, by source priority.

Reads `11_datasets_info.csv`. Writes `12_parquet_merged/` and `12_datasets_info_parquet.csv`.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import src.files as files
from src.paths import data_path, load_settings

# Load settings
settings = load_settings()

# Load datasets info
infile = data_path("data/outputs/10_datasets/11_datasets_info.csv")
datasets_df = pd.read_csv(infile)
datasets_df = datasets_df.fillna(np.nan)

# Create parquet files
data_nrows = None  # for testing
sites_done = []  # List of sites that were already processed
_datasets_df = datasets_df.copy()
for ix, datasetinfo in _datasets_df.iterrows():

    # # TODO testing
    # if ix < 204:
    #     continue
    # # TODO testing

    datasets_df, sites_done = files.create_parquet_files(
        datasets_df=datasets_df,
        # data_nrows=10000,
        data_nrows=data_nrows,
        settings=settings,
        site=datasetinfo['SITE'],
        ix=ix,
        sites_done=sites_done,
        showplot=True
    )

# Save to file
datasets_df = datasets_df.reset_index(drop=True)
# datasets_df = datasets_df.fillna("n.a.")
datasets_df = datasets_df.sort_values(by=['SITE'], inplace=False)
outfile = data_path("data/outputs/10_datasets/12_datasets_info_parquet.csv")
print(f"\n{'-' * 80}\nSaving info about {len(datasets_df)} datasets to file {outfile}.\n{'-' * 80}")
datasets_df.to_csv(outfile, index=False)
