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
# NEP:  [ ]TA/VPD [ ]SWIN/TA [ ]SWC/VPD [ ]SWIN/VPD [ ]TA/SWC
# ET:   [ ]TA/VPD [ ]SWIN/TA [ ]SWC/VPD [ ]SWIN/VPD [ ]TA/SWC
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
filepath = Path(dir_prev_results) / f"41_SHAPVALUES-{shap_type}_AggregatedPerSite_BIN-{xvar}+BIN-{yvar}+{FLUX}.parquet"
shapvals_sites_agg_long_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
shapvals_sites_agg_long_df = shapvals_sites_agg_long_df.loc[shapvals_sites_agg_long_df['IGBP'] != 'DNF']
print(f"Number of sites: {len(shapvals_sites_agg_long_df['SITE'].unique())}")
igbps = shapvals_sites_agg_long_df['IGBP'].unique()
print(f"Found IGBPs: {igbps}")

# Remove site info and IGBP, cannot be aggregated
shapvals_sites_agg_long_df = shapvals_sites_agg_long_df.drop('SITE', axis=1, inplace=False)

# Keep required columns only
targets = ('_SHAPVALS', 'BIN_')
keepcols = [c for c in shapvals_sites_agg_long_df.columns
            if any(str(c).startswith(t)
                   or str(c).endswith(t)
                   or str(c) == 'IGBP' for t in targets)]
shapvals_sites_agg_long_df = shapvals_sites_agg_long_df[keepcols].copy()

for i in igbps:
    # if i != 'EBF':
    #     continue
    print(f"\nProcessing IGBP {i} ...")
    subset = shapvals_sites_agg_long_df.loc[shapvals_sites_agg_long_df['IGBP'] == i].copy()
    subset = subset.drop('IGBP', axis=1, inplace=False)

    # Aggregate SHAP values per IGBP
    subset_agg_df = aggregate_shap_values_across_sites(
        df=subset, binx=binx, biny=biny
    )

    # Save to Parquet
    # f"41_SHAPVALUES-{shap_type}_AggregatedPerSite_BIN-{xvar}+BIN-{yvar}+{FLUX}.parquet"
    outfilepath = dv.save_parquet(
        filename=f"43_SHAPVALUES-{shap_type}_AggregatedAcrossIGBP-{i}_BIN+{xvar}+BIN+{yvar}_{FLUX}",
        data=subset_agg_df,
        outpath=dir_out)
    # print(f"Saved SHAP values across all files as mean to file {outfilepath}.")
    # shapvals_sites_grouped_agg_df.to_csv(outfilepath.replace('.parquet', '.csv'))
