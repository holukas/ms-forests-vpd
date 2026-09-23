"""
Find which of NEE, LE, GPP, RECO, TA, VPD, SWIN and SWC each site provides, and under which name.

Redundant: `13b_check_available_vars_SWC.py` writes the same file and more, and is the one
behind the main analysis.

Reads `12_datasets_info_parquet.csv`. Writes `13_datasets_info_parquet_vars.csv`.
"""
from pathlib import Path

import pandas as pd
from diive.core.io.files import load_parquet
from src.paths import data_path, resolve_stored_path

# Load datasets info
infile = data_path("data/outputs/10_datasets/12_datasets_info_parquet.csv")
datasets_df = pd.read_csv(infile)

required_vars = dict(
    NEE_VAR=['NEE_VUT_50', 'NEE_vUT_USTAR50', 'NEE_CUT_50'],  # Used to calculate NEP
    NEE_QC_VAR=['NEE_VUT_50_QC', 'NEE_vUT_USTAR50', 'NEE_CUT_50_QC'],
    LE_VAR='LE_F_MDS',  # Used to calculate ET
    LE_QC_VAR='LE_F_MDS_QC',
    GPP_VAR=['GPP_NT_VUT_50', 'GPP_NT_vUT_USTAR50', 'GPP_NT_CUT_50',
             'GPP_DT_VUT_50', 'GPP_DT_vUT_USTAR50', 'GPP_DT_CUT_50'],
    RECO_VAR=['RECO_NT_VUT_50', 'RECO_NT_vUT_USTAR50', 'RECO_NT_CUT_50',
              'RECO_DT_VUT_50', 'RECO_DT_vUT_USTAR50', 'RECO_DT_CUT_50'],
    SWIN_VAR='SW_IN_F',
    SWIN_QC_VAR='SW_IN_F_QC',
    TA_VAR='TA_F',
    TA_QC_VAR='TA_F_QC',
    VPD_VAR='VPD_F',
    VPD_QC_VAR='VPD_F_QC',
    PREC_VAR='P_F',
    SWC_VAR=['SWC_F_MDS_1', 'SWC_F_MDS_2'],
    SWC_QC_VAR=['SWC_F_MDS_1_QC', 'SWC_F_MDS_2_QC'],
)

_datasets_df = datasets_df.copy()
for ix, site in _datasets_df.iterrows():

    # --- TODO testing
    # if site['SITE'] != 'CN-Din':
    #     continue
    # if ix > 10:
    #     break
    # --- TODO testing

    print(f"\nLoading data for site #{ix + 1} {site['SITE']} ...")
    filepath = resolve_stored_path(site['_FILEPATH_PARQUET'])
    sitedata = load_parquet(filepath)
    available_vars = sitedata.columns

    for var_key, expected_col_name in required_vars.items():
        if isinstance(expected_col_name, str):
            if expected_col_name not in available_vars:
                expected_col_name = '-MISSING-'
                # raise Exception(f"Required variable '{expected_col_name}' not found in site data for site #{ix} {row['SITE']}.")
        elif isinstance(expected_col_name, list):
            found_one = False
            for alt_name in expected_col_name:
                if alt_name in available_vars:
                    if len(sitedata[alt_name].dropna()) > 0:
                        found_one = True
                        expected_col_name = alt_name
                        break
                    else:
                        found_one = False
            if not found_one:
                expected_col_name = '-MISSING-'
                # raise Exception(f"Required variable '{expected_col_name}' not found in site data for site #{ix} {row['SITE']}.")

        datasets_df.loc[ix, var_key] = expected_col_name

# Save to file
datasets_df = datasets_df.reset_index(drop=True)
datasets_df = datasets_df.sort_values(by=['SITE'], inplace=False)
outfile = data_path("data/outputs/10_datasets/13_datasets_info_parquet_vars.csv")
print(f"\n{'-' * 80}\nSaving info about {len(datasets_df)} datasets to file {outfile}.\n{'-' * 80}")
datasets_df.to_csv(outfile, index=False)
