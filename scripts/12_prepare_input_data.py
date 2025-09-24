"""
Prepare input data for XGBoost models.
"""
from pathlib import Path

import pandas as pd

import src.files as files

# datasets_df: pd.DataFrame

# Load datasets info
infile = Path('../data/outputs/11_datasets_parquet_vars_stats.csv')
datasets_df = pd.read_csv(infile)

# Load settings
settings = files.read_settings_file("../config/settings.yaml")

_datasets_df = datasets_df.copy()
for ix, siteconfig in _datasets_df.iterrows():
    datasets_df = files.prepare_input_data(
        ix=ix,
        siteconfig=siteconfig,
        settings=settings,
        siteinfo_df=datasets_df
    )

# Save to file
datasets_df = datasets_df.reset_index(drop=True)
datasets_df = datasets_df.sort_values(by=['SITE'], inplace=False)
outfile = Path('../data/outputs/12_datasets_parquet_vars_stats_subsets.csv')
print(f"\n{'-' * 80}\nSaving info about {len(datasets_df)} datasets to file {outfile}.\n{'-' * 80}")
datasets_df.to_csv(outfile, index=False)
