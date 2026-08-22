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
    dir_prev_results = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT

    # Create output directory
    dir_out = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT
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

    # Number of unique IGBPs
    igbps = shapvals_sites_agg_long_df['IGBP'].unique()
    print(f"Found IGBPs: {igbps}")

    # Count number of sites per IGBP
    igbps_n_sites = []
    for i in igbps:
        check_n_sites = shapvals_sites_agg_long_df.loc[shapvals_sites_agg_long_df['IGBP'] == i]
        n_sites = len(check_n_sites['SITE'].unique())
        igbps_n_sites.append(n_sites)

    # Remove site info and IGBP, cannot be aggregated
    shapvals_sites_agg_long_df = shapvals_sites_agg_long_df.drop('SITE', axis=1, inplace=False)

    # Keep required columns only
    targets = ('_SHAPVALS', 'BIN_', '_ZSCORE')
    keepcols = [c for c in shapvals_sites_agg_long_df.columns
                if any(str(c).startswith(t)
                       or str(c).endswith(t)
                       or str(c) == 'IGBP' for t in targets)]
    shapvals_sites_agg_long_df = shapvals_sites_agg_long_df[keepcols].copy()

    for ix, i in enumerate(igbps):
        print(f"\nProcessing IGBP {i} ...")
        subset = shapvals_sites_agg_long_df.loc[shapvals_sites_agg_long_df['IGBP'] == i].copy()
        subset = subset.drop('IGBP', axis=1, inplace=False)

        # Aggregate SHAP values per IGBP
        subset_agg_df = aggregate_shap_values_across_sites(
            df=subset, binx=binx, biny=biny
        )

        # Add total number of sites aggregated
        # Added here in this extra step, b/c the counts per bin only
        # give the number of sites in the respective bin, not the
        # overall number of sites that were aggregated.
        subset_agg_df['N_SITES'] = igbps_n_sites[ix]

        # Save to Parquet
        # f"41_SHAPVALUES-{shap_type}_AggregatedPerSite_BIN-{xvar}+BIN-{yvar}+{FLUX}.parquet"
        outfilepath = dv.save_parquet(
            filename=f"43_SHAPVALUES-{shap_type}_{aggfunc}AggregatedAcrossIGBP-{i}_BIN-{xvar}+BIN-{yvar}+{FLUX}",
            data=subset_agg_df,
            outpath=dir_out)
        subset_agg_df.to_csv(outfilepath.replace('.parquet', '.csv'))
