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
from src.paths import load_settings

# ------------------------------
# Variables
# NEP, ET, GPP, RECO, TA, VPD, SWIN, SWC

# Settings for searching in the correct (sub)folder
FLUX = 'NEP_ZSCORE'
# FLUX = 'ET_ZSCORE'
# FLUX = 'GPP_ZSCORE'
# FLUX = 'RECO_ZSCORE'
CONDITIONAL = True  # SHAP

# Run variant. An empty string reads the baseline results and overwrites the
# submitted aggregation. Any other value adds a folder level on both sides, so
# the inputs come from <stage>/<FLUX>/<shap_type>/<VARIANT>/ and the outputs go
# to the matching variant folder. The earlier stages must have run with the same
# value, otherwise there is nothing to read.
VARIANT = ""
# Site subset. An empty string keeps every site. "deeper-only" reads what 41
# wrote for the 128 sites whose soil water comes from below layer 1, and writes
# next to it. The value has to match the one 41 ran with.
SITE_SUBSET = ""

# Aggregation combos: xvar / yvar. Every pair listed here is processed in one run,
# which keeps a variant complete: the figures need all five.
VAR_PAIRS = [
    ['TA_ZSCORE', 'VPD_ZSCORE'],
    ['SWC_ZSCORE', 'VPD_ZSCORE'],
    ['TA_ZSCORE', 'SWC_ZSCORE'],
    ['ET_ZSCORE', 'VPD_ZSCORE'],
    ['ET_ZSCORE', 'SWC_ZSCORE'],
]

aggfunc = 'mean'
# aggfunc = 'median'

# ------------------------------
# Agg groups, use z-scores:
# NEP:  [x]TA/VPD [x]SWC/VPD [x]TA/SWC [x] ET/VPD [x] ET/SWC [ ]SWIN/TA [ ]SWIN/VPD
# ET:   [ ]TA/VPD [ ]SWC/VPD [ ]TA/SWC [ ] ET/VPD [ ] ET/SWC [ ]SWIN/TA [ ]SWIN/VPD
# GPP:  [ ]TA/VPD [ ]SWC/VPD [ ]TA/SWC [ ] ET/VPD [ ] ET/SWC [ ]SWIN/TA [ ]SWIN/VPD
# RECO: [ ]TA/VPD [ ]SWC/VPD [ ]TA/SWC [ ] ET/VPD [ ] ET/SWC [ ]SWIN/TA [ ]SWIN/VPD
# ------------------------------

for VARS in VAR_PAIRS:
    print('')
    print('=' * 70)
    print(f'{VARS[0]} x {VARS[1]}')
    print('=' * 70)

    xvar = VARS[0]
    yvar = VARS[1]
    binx = f"BIN_{xvar}"
    biny = f"BIN_{yvar}"

    # Load settings
    settings = load_settings()
    shap_type = 'conditional' if CONDITIONAL else 'standard'
    dir_prev_results = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT / SITE_SUBSET

    # Create output directory
    dir_out = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT / SITE_SUBSET
    # parents=True: Creates any necessary parent directories that don't exist.
    # exist_ok=True: Prevents an error if the directory already exists.
    dir_out.mkdir(parents=True, exist_ok=True)

    # Load SHAP values aggregated per site
    filepath = Path(
        dir_prev_results) / f"41_SHAPVALUES-{shap_type}_{aggfunc}AggregatedPerSite_BIN-{xvar}+BIN-{yvar}+{FLUX}.parquet"
    shapvals_sites_agg_long_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
    shapvals_sites_agg_long_df = shapvals_sites_agg_long_df.loc[shapvals_sites_agg_long_df['IGBP'] != 'DNF']

    # Total number of sites
    n_sites = len(shapvals_sites_agg_long_df['SITE'].unique())
    print(f"Number of sites: {n_sites}")

    # Keep required columns only
    targets = ('_SHAPVALS', 'BIN_', '_ZSCORE')
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

    # Add total number of sites aggregated
    # Added here in this extra step, b/c the counts per bin only
    # give the number of sites in the respective bin, not the
    # overall number of sites that were aggregated.
    shapvals_sites_grouped_agg_df['N_SITES'] = n_sites

    # Save to Parquet
    # 41_SHAPVALUES-conditional_AggregatedPerSite_BIN-TA_ZSCORE+BIN-VPD_ZSCORE+NEP_ZSCORE
    outfilepath = dv.save_parquet(
        filename=f"42_SHAPVALUES-{shap_type}_{aggfunc}AggregatedAcrossSites_BIN-{xvar}+BIN-{yvar}+{FLUX}",
        data=shapvals_sites_grouped_agg_df,
        outpath=dir_out)
    shapvals_sites_grouped_agg_df.to_csv(outfilepath.replace('.parquet', '.csv'))
