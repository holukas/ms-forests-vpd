from pathlib import Path

import diive as dv
import pandas as pd

import src.files as files
import src.stages as s
from src.common import deeper_swc_sites
from src.paths import load_settings

# ------------------------------
# Variables
# NEP, NEE, LE, GPP, RECO, TA, VPD, SWIN, SWC

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
# Site subset. An empty string keeps every site. "deeper-only" keeps the 128
# sites whose soil water comes from below layer 1, which is the set needed to
# compare shallow against deep without the sites that cannot move. It adds a
# folder level to the output, so a subset run never overwrites the full one.
# Run it against both VARIANT values to get the two matched arms.
SITE_SUBSET = ""

# Load settings
settings = load_settings()
shap_type = 'conditional' if CONDITIONAL else 'standard'
dir_prev_results = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type / VARIANT

# Create output directory
dir_out = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT / SITE_SUBSET
# parents=True: Creates any necessary parent directories that don't exist.
# exist_ok=True: Prevents an error if the directory already exists.
dir_out.mkdir(parents=True, exist_ok=True)

# Load subsets info
infile = (Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
          / "21_SUBSETS_parquet_vars_stats_subsets.csv")
subsets_df = pd.read_csv(infile)

if SITE_SUBSET == "deeper-only":
    keep = deeper_swc_sites()
    before = len(subsets_df)
    subsets_df = subsets_df[subsets_df['SITE'].isin(keep)].reset_index(drop=True)
    print(f"Site subset: {len(subsets_df)} of {before} sites use a layer below SWC_F_MDS_1")
elif SITE_SUBSET:
    raise ValueError(f"SITE_SUBSET must be '' or 'deeper-only', not {SITE_SUBSET!r}")


shapvals_sites_agg_long_df = None
sites_df = None

for ix, siteconfig in subsets_df.iterrows():
    igbp = siteconfig['IGBP']
    site = siteconfig['SITE']

    filename = f"{site}_shap-{shap_type}_{FLUX}.parquet"
    filepath = dir_prev_results / filename
    print(f"\nLoading data for site #{ix + 1} {site} ({filepath})")
    # Read as written. The subset is sparse by design, four months, daytime,
    # QC 0, so sanitizing would regularize the index and insert empty rows,
    # which inflates N_VALUES roughly tenfold.
    shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
    # keepcols = [c for c in shapvals_df.columns if "_SHAPVALS" in c]

    scenarios = [s.stage_0, s.stage_1, s.stage_2, s.stage_3,
                 s.stage_4, s.stage_5, s.stage_6, s.stage_7, s.stage_8]

    for i, scen in enumerate(scenarios):
        subset, ta_class, vpd_class, swc_class, condition = scen(shapvals_df)
        n_records = len(subset.index)
        cur_scenario_dict = dict()
        cur_scenario_dict['SITE'] = site
        cur_scenario_dict['IGBP'] = igbp
        cur_scenario_dict['SCENARIO'] = i
        cur_scenario_dict['CONDITION'] = condition
        cur_scenario_dict['N_VALUES'] = n_records
        cur_scenario_dict['TA_CLASS'] = ta_class
        cur_scenario_dict['VPD_CLASS'] = vpd_class
        cur_scenario_dict['SWC_CLASS'] = swc_class

        for c in shapvals_df.columns:
            series = subset[c].copy()
            cur_scenario_dict[f'{c}_POS_AVG'] = series[series > 0].mean()
            cur_scenario_dict[f'{c}_NEG_AVG'] = series[series < 0].mean()
            cur_scenario_dict[f'{c}_OVR_AVG'] = series.mean()
            cur_scenario_dict[f'{c}_OVR_SD'] = series.std()
            cur_scenario_dict[f'{c}_OVR_MEDIAN'] = series.median()
            cur_scenario_dict[f'{c}_OVR_ABS_AVG'] = series.abs().mean()
            cur_scenario_dict[f'{c}_OVR_ABS_MEDIAN'] = series.abs().median()
            cur_scenario_dict[f'{c}_OVR_ABS_SD'] = series.abs().std()

        # Calculate net for each high-resolution observation
        # shapcols = [c for c in shapvals_df.columns if "_SHAPVALS" in c]
        cur_scenario_dict['NET_SHAPVALS_OVR_AVG'] = subset['SUM'].mean()
        cur_scenario_dict['NET_SHAPVALS_OVR_SEM'] = subset['SUM'].sem()

        # Calculating the net effect SD directly from individual observations ensures
        # that the internal relationships and trade-offs between variables are preserved
        # at every time step. This row-wise approach allows the resulting standard
        # deviation to naturally incorporate the covariance between drivers, providing
        # a more accurate measure of total uncertainty than simply aggregating
        # pre-calculated feature statistics.
        cur_scenario_dict['NET_SHAPVALS_OVR_SD'] = subset['SUM'].std()

        new_row_df = pd.DataFrame([cur_scenario_dict], index=[site])

        if not isinstance(sites_df, pd.DataFrame):
            sites_df = new_row_df.copy()
        else:
            sites_df = pd.concat([sites_df, new_row_df], axis=0)

# Save to file
outfilepath = dv.save_parquet(
    filename=f"44_SHAPVALUES-{shap_type}_AggregatedAcrossScenarios_{FLUX}",
    data=sites_df,
    outpath=dir_out)
sites_df.to_csv(outfilepath.replace('.parquet', '.csv'))

print(sites_df)
