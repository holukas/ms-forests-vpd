from pathlib import Path

import diive as dv
import pandas as pd

import src.files as files
import src.scenarios as s

# ------------------------------
# Variables
# NEP, NEE, LE, GPP, RECO, TA, VPD, SWIN, SWC

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
# aggfunc = 'median'

# ------------------------------
# Agg groups, use z-scores:
# NEP:  [ ]TA/VPD [ ]SWIN/TA [ ]SWC/VPD [ ]SWIN/VPD [ ]TA/SWC
# ET:   [ ]TA/VPD [ ]SWIN/TA [ ]SWC/VPD [ ]SWIN/VPD [ ]TA/SWC
# GPP:  [ ]TA/VPD [ ]SWIN/TA [ ]SWC/VPD [ ]SWIN/VPD [ ]TA/SWC
# RECO: [ ]TA/VPD [ ]SWIN/TA [ ]SWC/VPD [ ]SWIN/VPD [ ]TA/SWC
# ------------------------------

# Load settings
settings = files.read_settings_file("../../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'
dir_prev_results = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type

# Create output directory
dir_out = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type
# parents=True: Creates any necessary parent directories that don't exist.
# exist_ok=True: Prevents an error if the directory already exists.
dir_out.mkdir(parents=True, exist_ok=True)

# Load subsets info
infile = Path('../../data/outputs/20_subsets/21_SUBSETS_parquet_vars_stats_subsets.csv')
subsets_df = pd.read_csv(infile)

shapvals_sites_agg_long_df = None
sites_df = None
for ix, siteconfig in subsets_df.iterrows():

    igbp = siteconfig['IGBP']
    site = siteconfig['SITE']

    filename = f"{site}_shap-{shap_type}_{FLUX}.parquet"
    filepath = dir_prev_results / filename
    print(f"\nLoading data for site #{ix + 1} {site} ({filepath})")
    shapvals_df = dv.load_parquet(filepath)
    keepcols = [c for c in shapvals_df.columns if "_SHAPVALS" in c]

    scenarios = [s.scenario_0, s.scenario_1, s.scenario_2, s.scenario_3,
                 s.scenario_4, s.scenario_5]

    for i, scen in enumerate(scenarios):
        subset, ta, vpd, swc, condition = scen(shapvals_df)
        n_records = len(subset.index)
        cur_scenario_dict = dict()
        cur_scenario_dict['SITE'] = site
        cur_scenario_dict['IGBP'] = igbp
        cur_scenario_dict['SCENARIO'] = i
        cur_scenario_dict['CONDITION'] = condition
        cur_scenario_dict['N_VALUES'] = n_records
        cur_scenario_dict['TA'] = ta
        cur_scenario_dict['VPD'] = vpd
        cur_scenario_dict['SWC'] = swc

        for k in keepcols:
            series = subset[k].copy()
            cur_scenario_dict[f'{k}_POS_AVG'] = series[series > 0].mean()
            cur_scenario_dict[f'{k}_NEG_AVG'] = series[series < 0].mean()
            cur_scenario_dict[f'{k}_OVR_AVG'] = series.mean()
            cur_scenario_dict[f'{k}_OVR_SD'] = series.std()
            cur_scenario_dict[f'{k}_OVR_MEDIAN'] = series.median()
            cur_scenario_dict[f'{k}_OVR_ABS_AVG'] = series.abs().mean()
            cur_scenario_dict[f'{k}_OVR_ABS_MEDIAN'] = series.abs().median()
            cur_scenario_dict[f'{k}_OVR_ABS_SD'] = series.abs().std()

        new_row_df = pd.DataFrame([cur_scenario_dict], index=[site])

        if not isinstance(sites_df, pd.DataFrame):
            sites_df = new_row_df.copy()
        else:
            sites_df = pd.concat([sites_df, new_row_df], axis=0)

    # print(sites_df[['SCENARIO', 'N_VALUES', 'VPD_SHAPVALS_OVR_AVG', 'TA_SHAPVALS_OVR_AVG', 'SWIN_SHAPVALS_OVR_AVG',
    #                 'SWC_SHAPVALS_OVR_AVG']])
    # print(sites_df)

_sites_df = sites_df.copy()
_sites_df = _sites_df.drop('SITE', axis=1, inplace=False)
_sites_df = _sites_df.drop('CONDITION', axis=1, inplace=False)
_sites_df = _sites_df.drop('IGBP', axis=1, inplace=False)
_sites_df = _sites_df.groupby('SCENARIO').mean()

# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# fig = plt.figure(figsize=(9, 6), dpi=150, facecolor="white")
# gs = gridspec.GridSpec(1, 1)  # rows, cols
# # gs.update(wspace=.3, hspace=.2, left=0.03, right=0.94, top=0.97, bottom=0.04)
# ax = fig.add_subplot(gs[0, 0])
# plotdf = _sites_df[['TA_SHAPVALS_OVR_AVG', 'VPD_SHAPVALS_OVR_AVG',
#                    'SWIN_SHAPVALS_OVR_AVG', 'SWC_SHAPVALS_OVR_AVG']].copy()
# plotdf.plot.bar(ax=ax)
# # plotdf = plotdf.pivot(index='SITE', columns='Group', values='Value')
# # sites_df['TA_SHAPVALS_OVRSUMRANGE'].plot.bar(color='#EF5350', ax=ax, legend=False, width=.7)
# # sites_df['TA_SHAPVALS_OVRSUMRANGE'].plot.bar(color='#03A9F4', ax=ax, legend=False, width=.7)
# # sites_df['TA_SHAPVALS_NEGSUM'].plot.bar(color='#03A9F4', legend=False, width=.7)
# fig.tight_layout()
# fig.show()

# 43_SHAPVALUES-conditional_AggregatedAcrossIGBP-MF_BIN+TA_ZSCORE+BIN+VPD_ZSCORE_NEP_ZSCORE.parquet
outfilepath = dv.save_parquet(
    filename=f"44_SHAPVALUES-{shap_type}_AggregatedAcrossScenarios_{FLUX}",
    data=sites_df,
    outpath=dir_out)
sites_df.to_csv(outfilepath.replace('.parquet', '.csv'))

print(sites_df)
