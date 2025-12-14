"""
Prepare input data for XGBoost models.
"""
import time
from pathlib import Path
from src.common import get_variable_names
import pandas as pd

import src.files as files

# Load datasets info
infile = Path('../../data/outputs/10_datasets/14_datasets_info_parquet_vars_stats.csv')
datasets_df = pd.read_csv(infile)

# Keep sites where SWC is available and that are not DNF (only 2 sites)
datasets_df = datasets_df.loc[
    (datasets_df['SWC_AVG'] != '-MISSING-') &  # Condition 1: SWC must be available
    (datasets_df['IGBP'] != 'DNF')  # Condition 2: IGBP must not be 'DNF'
    ].reset_index(drop=True)

# Load settings
settings = files.read_settings_file("../../config/settings.yaml")

_datasets_df = datasets_df.copy()
subsetinfo_df = pd.DataFrame()
counter = 0
for ix, siteconfig in _datasets_df.iterrows():
    # if ix > 1:
    #     continue
    counter += 1
    varnames = get_variable_names(siteconfig)  # Variable names for this site
    subsetinfo = files.create_subsets_parquet_files(
        site=siteconfig['SITE'],
        ix=ix,
        settings=settings,
        filepath_parquet_fullset=siteconfig['_FILEPATH_PARQUET'],
        varnames=varnames
    )
    subsetinfo['LAT'] = siteconfig['LAT']
    subsetinfo['LON'] = siteconfig['LON']
    subsetinfo['ELEVATION'] = siteconfig['ELEVATION']
    subsetinfo['IGBP'] = siteconfig['IGBP']

    newrow = pd.DataFrame.from_dict(subsetinfo, orient='index').transpose()
    if counter == 1:
        subsetinfo_df = newrow
    else:
        subsetinfo_df = pd.concat([subsetinfo_df, newrow], ignore_index=True)

# # Keep required columns
# keepcols = ['SITE', 'ORIGIN', 'N_YEARS', 'DATE_FIRST', 'DATE_LAST', 'LAT', 'LON', 'ELEVATION', 'IGBP']
# subsets_df = datasets_df[keepcols].copy()
# subsets_df = subsets_df.sort_values(by=['SITE'], inplace=False).reset_index(drop=True)

# Save to file
OUTDIR = Path('../../data/outputs/20_subsets/')
outfile = OUTDIR / '21_SUBSETS_parquet_vars_stats_subsets.csv'
print(f"\n{'-' * 80}\nSaving info about {len(subsetinfo_df)} subsets to file {outfile.resolve()}.\n{'-' * 80}")
subsetinfo_df.to_csv(outfile, index=False)
