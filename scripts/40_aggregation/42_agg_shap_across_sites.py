"""
Aggregation of SHAP Values

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
# FLUX = 'NEP'  # Go to FLUX folder
FLUX = 'GPP'  # Go to FLUX folder
# FLUX = 'RECO'  # Go to FLUX folder
FLUX = 'ET'  # Go to FLUX folder
xvar = 'TA'
yvar = 'SWC'
aggfunc = 'median'
CONDITIONAL = True  # SHAP
# Agg groups: [X]TA/VPD [X]SWIN/TA [X]SWC/VPD [X]SWIN/VPD [X]TA/SWC
# ------------------------------

binx = f"BIN_{xvar}"
biny = f"BIN_{yvar}"

# Load settings
settings = files.read_settings_file("../../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type
filepath = Path(results_outdir) / f"2_PerSite_Aggregated_SHAPValues-{shap_type}_BIN-{xvar}_BIN-{yvar}_{FLUX}.parquet"
shapvals_sites_agg_long_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
shapvals_sites_agg_long_df = shapvals_sites_agg_long_df.loc[shapvals_sites_agg_long_df['IGBP'] != 'DNF']

# Remove all rows where all records are NaN,
# binx and biny columns are ignored for this check
cols_to_ignore = [binx, biny]
cols_to_check = [col for col in shapvals_sites_agg_long_df.columns if col not in cols_to_ignore]

# Filter out rows where all values in the selected columns are NaN
shapvals_sites_agg_long_df = shapvals_sites_agg_long_df.loc[
    ~shapvals_sites_agg_long_df[cols_to_check].isna().all(axis=1)].copy()

print(f"Number of sites: {len(shapvals_sites_agg_long_df['SITE'].unique())}")


# Remove site info and IGBP, cannot be aggregated (strings)
shapvals_sites_agg_long_df = shapvals_sites_agg_long_df.drop('SITE', axis=1, inplace=False)
shapvals_sites_agg_long_df = shapvals_sites_agg_long_df.drop('IGBP', axis=1, inplace=False)

# Aggregate SHAP values across all sites
shapvals_sites_grouped_agg_df = aggregate_shap_values_across_sites(
    df=shapvals_sites_agg_long_df, binx=binx, biny=biny
)

# Save to Parquet
outfilepath = dv.save_parquet(
    filename=f"3_AllSites_Aggregated_SHAPValues-{shap_type}_BIN-{xvar}_BIN-{yvar}_{FLUX}",
    data=shapvals_sites_grouped_agg_df,
    outpath=results_outdir)
# print(f"Saved SHAP values across all files as mean to file {outfilepath}.")
# shapvals_sites_grouped_agg_df.to_csv(outfilepath.replace('.parquet', '.csv'))
