"""
Train XGBoost model for each site and save SHAP values to file.
"""

from pathlib import Path

import pandas as pd

import src.files as files

# ------------------------------
# Variables
FLUX = 'RECO'  # NEP, ET, GPP, RECO, NEE, LE
FEATURES = ['TA', 'SWIN', 'VPD', 'SWC']
CONDITIONAL = True  # Use conditional SHAP instead of standard SHAP
# ------------------------------

# Load settings
settings = files.read_settings_file("../config/settings.yaml")
subfolder = 'conditional' if CONDITIONAL else 'standard'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / subfolder

# Load datasets info
infile = Path('../data/outputs/12_datasets_parquet_vars_stats_subsets.csv')
datasets_df = pd.read_csv(infile)

# Sites where SWC is available and that are not DNF (only 2 sites)
subset = datasets_df.loc[datasets_df['SWC_AVG'] != '-MISSING-'].copy()
subset = subset.loc[subset['IGBP'] != 'DNF'].copy()
subset = subset.reset_index(drop=True)
keepcols = ['SITE', 'IGBP', 'LAT', 'LON', 'ELEVATION', 'N_YEARS', 'DATE_FIRST', 'DATE_LAST', 'ORIGIN']
subset = subset[keepcols].copy()
print(f"{subset}")

print(f"\nSite years:\n  "
      f"Total: {int(subset['N_YEARS'].sum())}\n  "
      f"Min: {int(subset['N_YEARS'].min())}\n  "
      f"Max: {int(subset['N_YEARS'].max())}\n  "
      f"Top 10 most years: {subset.sort_values(by='N_YEARS', ascending=False).head(10)['SITE'].values}\n  "
      f"Top 10 least years: XXX")
print(f"\nElevation:\n  "
      f"Max: {int(subset['ELEVATION'].max())}\n  "
      f"Min: {int(subset['ELEVATION'].min())}")

# Rename cols
rename_dict = {
    'SITE': 'Site',
    'ORIGIN': 'Origin',
    'ELEVATION': 'Elevation',
    'LAT': 'Latitude',
    'LON': 'Longitude',
    'DATE_FIRST': 'From',
    'DATE_LAST': 'To',
    'N_YEARS': 'Years',
}
subset = subset.rename(columns=rename_dict, inplace=False)

# Save to file
subset = subset.reset_index(drop=True)
subset = subset.sort_values(by=['Site'], inplace=False)
outfile = Path('../data/outputs/13_OVERVIEW_USED-SITES.csv')
print(f"\n{'-' * 80}\nSaving info about {len(subset)} datasets to file {outfile}.\n{'-' * 80}")
subset.to_csv(outfile, index=False)
