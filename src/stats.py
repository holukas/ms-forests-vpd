import pandas as pd
from diive.core.io.files import load_parquet

from src.common import get_variable_names


def basic_stats(siteinfo_df, siteconfig, ix) -> pd.DataFrame:
    _df = siteinfo_df.copy()

    # # -- TODO testing
    # if siteconfig['SITE'] != 'AU-Cum':
    #     return siteinfo_df
    # if ix + 1 < 128:
    #     return siteinfo_df
    # # -- TODO testing

    print(f"\nLoading data for site #{ix + 1} {siteconfig['SITE']} ...")

    filepath = siteconfig['_FILEPATH_PARQUET']
    sitedata = load_parquet(filepath)

    # Collect info
    date_first = sitedata.index[0]
    date_last = sitedata.index[-1]
    date_first_str = str(date_first.strftime('%d %b %Y'))
    date_last_str = str(date_last.strftime('%d %b %Y'))
    n_records = len(sitedata.index)
    n_years = (date_last.year - date_first.year) + 1

    # Get variable names for this site
    varnames = get_variable_names(siteconfig)

    siteinfo_df.loc[ix, 'DATE_FIRST'] = date_first_str
    siteinfo_df.loc[ix, 'DATE_LAST'] = date_last_str
    siteinfo_df.loc[ix, 'N_YEARS'] = n_years
    siteinfo_df.loc[ix, 'N_RECORDS'] = n_records
    siteinfo_df.loc[ix, 'NEE_AVG'] = sitedata[varnames['nee_var']].mean()
    siteinfo_df.loc[ix, 'NEE_N_RECORDS'] = sitedata[varnames['nee_var']].dropna().count()
    siteinfo_df.loc[ix, 'LE_AVG'] = sitedata[varnames['le_var']].mean()
    siteinfo_df.loc[ix, 'LE_N_RECORDS'] = sitedata[varnames['le_var']].dropna().count()
    siteinfo_df.loc[ix, 'SWIN_AVG'] = sitedata[varnames['swin_var']].mean()
    siteinfo_df.loc[ix, 'SWIN_N_RECORDS'] = sitedata[varnames['swin_var']].dropna().count()
    siteinfo_df.loc[ix, 'TA_AVG'] = sitedata[varnames['ta_var']].mean()
    siteinfo_df.loc[ix, 'TA_N_RECORDS'] = sitedata[varnames['ta_var']].dropna().count()
    siteinfo_df.loc[ix, 'VPD_AVG'] = sitedata[varnames['vpd_var']].mean()
    siteinfo_df.loc[ix, 'VPD_N_RECORDS'] = sitedata[varnames['vpd_var']].dropna().count()
    siteinfo_df.loc[ix, 'PREC/YR'] = sitedata[varnames['prec_var']].sum() / n_years
    siteinfo_df.loc[ix, 'PREC_N_RECORDS'] = sitedata[varnames['prec_var']].dropna().count()

    # SWC is completely missing for some sites
    if siteconfig['SWC_VAR'] != '-MISSING-':
        siteinfo_df.loc[ix, 'SWC_AVG'] = sitedata[varnames['swc_var']].mean()
        siteinfo_df.loc[ix, 'SWC_N_RECORDS'] = sitedata[varnames['swc_var']].dropna().count()
    else:
        siteinfo_df.loc[ix, 'SWC_AVG'] = '-MISSING-'
        siteinfo_df.loc[ix, 'SWC_N_RECORDS'] = '-MISSING-'

    # GPP is completely missing for some sites
    if siteconfig['GPP_VAR'] != '-MISSING-':
        siteinfo_df.loc[ix, 'GPP_AVG'] = sitedata[varnames['gpp_var']].mean()
        siteinfo_df.loc[ix, 'GPP_N_RECORDS'] = sitedata[varnames['gpp_var']].dropna().count()
    else:
        siteinfo_df.loc[ix, 'GPP_AVG'] = '-MISSING-'
        siteinfo_df.loc[ix, 'GPP_N_RECORDS'] = '-MISSING-'

    # RECO is completely missing for some sites
    if siteconfig['RECO_VAR'] != '-MISSING-':
        siteinfo_df.loc[ix, 'RECO_AVG'] = sitedata[varnames['reco_var']].mean()
        siteinfo_df.loc[ix, 'RECO_N_RECORDS'] = sitedata[varnames['reco_var']].dropna().count()
    else:
        siteinfo_df.loc[ix, 'RECO_AVG'] = '-MISSING-'
        siteinfo_df.loc[ix, 'RECO_N_RECORDS'] = '-MISSING-'

    # # RH is missing for one site (My-)
    # if siteconfig['RH_VAR'] != '-MISSING-':
    #     siteinfo_df.loc[ix, 'RH_AVG'] = sitedata[varnames['rh_var']].mean()
    #     siteinfo_df.loc[ix, 'RH_N_RECORDS'] = sitedata[varnames['rh_var']].dropna().count()
    # else:
    #     siteinfo_df.loc[ix, 'RH_AVG'] = '-MISSING-'
    #     siteinfo_df.loc[ix, 'RH_N_RECORDS'] = '-MISSING-'

    return siteinfo_df
