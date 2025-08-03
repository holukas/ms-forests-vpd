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
from aggregation import aggregate_shap_values_for_site, aggregate_shap_values_across_all_sites

# VARIABLES
# ---------
# 'ta_var', 'vpd_var', 'swc_var', 'swin_var'
binx = f"BIN_VPD_F"
biny = f"BIN_TA_F"
z = f"VPD_F_SHAPVALS"

# Load settings
settings = files.read_settings_file("../config/settings.yaml")

filepath = Path(settings['DIR_DATA_OUT_SHAPVALS_STANDARD']) / "2_PerSite_Aggregated_SHAPValues.parquet"
shapvals_sites_agg_long_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)

# Aggregate SHAP values across all sites
shapvals_sites_grouped_agg_df = aggregate_shap_values_across_all_sites(
    df=shapvals_sites_agg_long_df, binx=binx, biny=biny, z=z
)

# Save to Parquet
outfilepath = dv.save_parquet(
    filename=f"3_AllSites_Aggregated_SHAPValues",
    data=shapvals_sites_grouped_agg_df,
    outpath=Path(settings['DIR_DATA_OUT_SHAPVALS_STANDARD']))
# print(f"Saved SHAP values across all files as mean to file {outfilepath}.")
shapvals_sites_grouped_agg_df.to_csv(outfilepath.replace('.parquet', '.csv'))
