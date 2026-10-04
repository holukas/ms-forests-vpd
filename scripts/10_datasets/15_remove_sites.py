"""
Filter the site table down to the sites used in the analysis.

Keeps sites that have SWC, GPP and RECO, are not IGBP class DNF, and have at
least 3 years of data.

Reads: data/outputs/10_datasets/14_datasets_info_parquet_vars_stats.csv
Writes: data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv
"""
from pathlib import Path

import pandas as pd
from src.paths import data_path

# Load datasets info
infile = data_path("data/outputs/10_datasets/14_datasets_info_parquet_vars_stats.csv")
datasets_df = pd.read_csv(infile)

# Keep sites with SWC, GPP and RECO, not DNF, and with at least 3 years
datasets_df = datasets_df.loc[
    (datasets_df['SWC_AVG'] != '-MISSING-') &  # Condition 1: SWC must be available
    (datasets_df['GPP_AVG'] != '-MISSING-') &  # Condition 2: GPP must be available
    (datasets_df['RECO_AVG'] != '-MISSING-') &  # Condition 3: RECO must be available
    (datasets_df['IGBP'] != 'DNF') &  # Condition 2: IGBP must not be 'DNF' (too few sites)
    (datasets_df['N_YEARS'] >= 3)  # Condition 3: N_YEARS must be >= 3
    ].reset_index(drop=True)

print(f"Site years: {datasets_df['N_YEARS'].sum()}")

# # Extended info (MAT, MAP)
# extfile = data_path("data/outputs/10_datasets/11_datasets_info.csv")
# extended_df = pd.read_csv(extfile)
# datasets_df.loc[:, 'MAT'] = extended_df['MAT']
# datasets_df.loc[:, 'MAP'] = extended_df['MAP']

# Save to file
datasets_df = datasets_df.reset_index(drop=True)
datasets_df = datasets_df.sort_values(by=['SITE'], inplace=False)
outfile = data_path("data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv")
print(f"\n{'-' * 80}\nSaving info about {len(datasets_df)} datasets to file {outfile}.\n{'-' * 80}")
datasets_df.to_csv(outfile, index=False)
