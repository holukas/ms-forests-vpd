from pathlib import Path

import pandas as pd
from diive.core.io.files import load_parquet

# Load datasets info
infile = Path('../../data/outputs/10_datasets/12_datasets_info_parquet.csv')
datasets_df = pd.read_csv(infile)

required_vars = dict(
    NEE_VAR=['NEE_VUT_50', 'NEE_vUT_USTAR50'],  # Used to calculate NEP
    NEE_QC_VAR=['NEE_VUT_50_QC', 'NEE_vUT_USTAR50'],
    LE_VAR='LE_F_MDS',  # Used to calculate ET
    LE_QC_VAR='LE_F_MDS_QC',
    GPP_VAR=['GPP_NT_VUT_50', 'GPP_NT_vUT_USTAR50', 'GPP_DT_VUT_50', 'GPP_DT_vUT_USTAR50'],
    RECO_VAR=['RECO_NT_VUT_50', 'RECO_NT_vUT_USTAR50', 'RECO_DT_VUT_50', 'RECO_DT_vUT_USTAR50'],
    SWIN_VAR='SW_IN_F',
    TA_VAR='TA_F',
    VPD_VAR='VPD_F',
    PREC_VAR='P_F',
    SWC_VAR=['SWC_F_MDS_1', 'SWC_F_MDS_2'],
    # RH_VAR=['RH', 'RH_F', 'RH_1_1_1'],
)

_datasets_df = datasets_df.copy()
for ix, site in _datasets_df.iterrows():

    # # --- TODO testing
    # # if site['SITE'] != 'AU-Cum':
    # #     continue
    # if ix != 1:
    #     continue
    # # --- TODO testing

    print(f"\nLoading data for site #{ix + 1} {site['SITE']} ...")
    filepath = site['_FILEPATH_PARQUET']
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
outfile = Path('../../data/outputs/10_datasets/13_datasets_info_parquet_vars.csv')
print(f"\n{'-' * 80}\nSaving info about {len(datasets_df)} datasets to file {outfile}.\n{'-' * 80}")
datasets_df.to_csv(outfile, index=False)
