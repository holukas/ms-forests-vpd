"""
Prepare input data for XGBoost models.
"""
from pathlib import Path

import pandas as pd

# Load datasets info
infile = Path('../../data/outputs/10_datasets/14_datasets_info_parquet_vars_stats.csv')
datasets_df = pd.read_csv(infile)

# Keep sites where SWC is available and that are not DNF (only 2 sites)
datasets_df = datasets_df.loc[
    (datasets_df['SWC_AVG'] != '-MISSING-') &  # Condition 1: SWC must be available
    (datasets_df['IGBP'] != 'DNF') &  # Condition 2: IGBP must not be 'DNF' (too few sites)
    (datasets_df['N_YEARS'] >= 3)  # Condition 3: N_YEARS must be >= 3
    ].reset_index(drop=True)

print(f"Site years: {datasets_df['N_YEARS'].sum()}")

# Save to file
datasets_df = datasets_df.reset_index(drop=True)
datasets_df = datasets_df.sort_values(by=['SITE'], inplace=False)
outfile = Path('../../data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv')
print(f"\n{'-' * 80}\nSaving info about {len(datasets_df)} datasets to file {outfile}.\n{'-' * 80}")
datasets_df.to_csv(outfile, index=False)
