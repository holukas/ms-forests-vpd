"""
Aggregation of SHAP Values across sites

    This script aggregates **SHAP (SHapley Additive exPlanations) values** from multiple
    sites to create a consolidated dataset. It iterates through a collection of individual
    site data files, which contain SHAP values for each data point. For each site, it bins
    the SHAP values based on two key variables, `xvar` and `yvar`, and then calculates the
    median (or aggregation function defined in `aggfunc`) SHAP value for each bin.

    After processing all sites, the script combines the binned data and performs a final
    aggregation to calculate the median, 25th percentile, 75th percentile, and the total
    count of SHAP values for each unique bin combination across all sites. The final output
    is saved to both a Parquet and a CSV file, providing a summary of how the variables'
    contributions, as measured by SHAP values, are distributed across the predefined bins.
"""
from pathlib import Path

import diive as dv

import src.files as files
from src.aggregation import aggregate_shap_values_across_sites

# ------------------------------
# Variables
# NEP, ET, GPP, RECO, TA, VPD, SWIN, SWC

# Settings for searching in the correct (sub)folder
FLUX = 'NEP_ZSCORE'
# FLUX = 'ET_ZSCORE'
# FLUX = 'GPP_ZSCORE'
# FLUX = 'RECO_ZSCORE'
CONDITIONAL = True  # SHAP

# Aggregation combos: xvar / yvar
VARS = ['TA_ZSCORE', 'VPD_ZSCORE']
# VARS = ['SWIN_ZSCORE', 'TA_ZSCORE']
# VARS = ['SWC_ZSCORE', 'VPD_ZSCORE']
# VARS = ['SWIN_ZSCORE', 'VPD_ZSCORE']
# VARS = ['TA_ZSCORE', 'SWC_ZSCORE']
aggfunc = 'median'

# ------------------------------
# Agg groups, use z-scores:
# NEP:  [x]TA/VPD [x]SWIN/TA [x]SWC/VPD [x]SWIN/VPD [x]TA/SWC
# ET:   [x]TA/VPD [x]SWIN/TA [x]SWC/VPD [x]SWIN/VPD [x]TA/SWC
# GPP:  [ ]TA/VPD [ ]SWIN/TA [ ]SWC/VPD [ ]SWIN/VPD [ ]TA/SWC
# RECO: [ ]TA/VPD [ ]SWIN/TA [ ]SWC/VPD [ ]SWIN/VPD [ ]TA/SWC
# ------------------------------

xvar = VARS[0]
yvar = VARS[1]
binx = f"BIN_{xvar}"
biny = f"BIN_{yvar}"

# Load settings
settings = files.read_settings_file("../../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'
dir_prev_results = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type

# Create output directory
dir_out = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type
# parents=True: Creates any necessary parent directories that don't exist.
# exist_ok=True: Prevents an error if the directory already exists.
dir_out.mkdir(parents=True, exist_ok=True)

# Load SHAP values aggregated per site
filepath = Path(
    dir_prev_results) / f"41_SHAPVALUES-{shap_type}_{aggfunc}AggregatedPerSite_BIN-{xvar}+BIN-{yvar}+{FLUX}.parquet"
shapvals_sites_agg_long_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
shapvals_sites_agg_long_df = shapvals_sites_agg_long_df.loc[shapvals_sites_agg_long_df['IGBP'] != 'DNF']
print(f"Number of sites: {len(shapvals_sites_agg_long_df['SITE'].unique())}")

# Keep required columns only
targets = ('_SHAPVALS', 'BIN_')
keepcols = [c for c in shapvals_sites_agg_long_df.columns if
            any(str(c).startswith(t) or str(c).endswith(t) for t in targets)]
shapvals_sites_agg_long_df = shapvals_sites_agg_long_df[keepcols].copy()

# Remove all rows where all records are NaN,
# binx and biny columns are ignored for this check
cols_to_ignore = [binx, biny]
cols_to_check = [col for col in shapvals_sites_agg_long_df.columns if col not in cols_to_ignore]

# Filter out rows where all values in the selected columns are NaN
shapvals_sites_agg_long_df = shapvals_sites_agg_long_df.loc[
    ~shapvals_sites_agg_long_df[cols_to_check].isna().all(axis=1)].copy()

# Aggregate SHAP values across all sites
shapvals_sites_grouped_agg_df = aggregate_shap_values_across_sites(
    df=shapvals_sites_agg_long_df, binx=binx, biny=biny
)

# Save to Parquet
# 41_SHAPVALUES-conditional_AggregatedPerSite_BIN-TA_ZSCORE+BIN-VPD_ZSCORE+NEP_ZSCORE
outfilepath = dv.save_parquet(
    filename=f"42_SHAPVALUES-{shap_type}_{aggfunc}AggregatedAcrossSites_BIN-{xvar}+BIN-{yvar}+{FLUX}",
    data=shapvals_sites_grouped_agg_df,
    outpath=dir_out)
shapvals_sites_grouped_agg_df.to_csv(outfilepath.replace('.parquet', '.csv'))
