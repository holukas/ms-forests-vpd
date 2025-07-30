from diive.core.io.files import load_parquet

import src.files as files

# Load settings
settings = files.read_settings_file("../config/settings.yaml")

# Load site info
siteinfo_df = files.load_siteinfo(settings)

required_vars = dict(
    NEE_VAR='NEE_VUT_50',
    NEE_QC_VAR='NEE_VUT_50_QC',
    LE_VAR='LE_F_MDS',
    LE_QC_VAR='LE_F_MDS_QC',
    GPP_VAR='GPP_NT_VUT_50',
    RECO_VAR='RECO_NT_VUT_50',
    SWIN_VAR='SW_IN_F',
    TA_VAR='TA_F',
    VPD_VAR='VPD_F',
    PREC_VAR='P_F',
    SWC_VAR=['SWC_F_MDS_1', 'SWC_F_MDS_2']
)

_siteinfo_df = siteinfo_df.copy()
for ix, site in _siteinfo_df.iterrows():
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
                    found_one = True
                    expected_col_name = alt_name
                    break
            if not found_one:
                expected_col_name = '-MISSING-'
                # raise Exception(f"Required variable '{expected_col_name}' not found in site data for site #{ix} {row['SITE']}.")

        siteinfo_df.loc[ix, var_key] = expected_col_name

# Save updated site info
files.save_siteinfo(siteinfo_df=siteinfo_df, settings=settings)
