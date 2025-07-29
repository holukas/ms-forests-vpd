import pandas as pd
from diive.core.io.files import load_parquet


def basic_stats(siteinfo_df) -> pd.DataFrame:

    _df = siteinfo_df.copy()
    for ix, site in _df.iterrows():

        print(f"\nLoading data for site #{ix + 1} {site['SITE']} ...")

        filepath = site['_FILEPATH_PARQUET']
        sitedata = load_parquet(filepath)

        # Collect info
        date_first = sitedata.index[0]
        date_last = sitedata.index[-1]
        date_first_str = str(date_first.strftime('%d %b %Y'))
        date_last_str = str(date_last.strftime('%d %b %Y'))
        n_records = len(sitedata.index)
        n_years = (date_last.year - date_first.year) + 1

        # Get variable names for this site
        swin_var = site['SWIN_VAR']
        ta_var = site['TA_VAR']
        vpd_var = site['VPD_VAR']
        prec_var = site['PREC_VAR']
        swc_var = site['SWC_VAR']

        siteinfo_df.loc[ix, 'DATE_FIRST'] = date_first_str
        siteinfo_df.loc[ix, 'DATE_LAST'] = date_last_str
        siteinfo_df.loc[ix, 'N_YEARS'] = n_years
        siteinfo_df.loc[ix, 'N_RECORDS'] = n_records
        siteinfo_df.loc[ix, 'SWIN_AVG'] = sitedata[swin_var].mean()
        siteinfo_df.loc[ix, 'SWIN_N_RECORDS'] = sitedata[swin_var].dropna().count()
        siteinfo_df.loc[ix, 'TA_AVG'] = sitedata[ta_var].mean()
        siteinfo_df.loc[ix, 'TA_N_RECORDS'] = sitedata[ta_var].dropna().count()
        siteinfo_df.loc[ix, 'VPD_AVG'] = sitedata[vpd_var].mean()
        siteinfo_df.loc[ix, 'VPD_N_RECORDS'] = sitedata[vpd_var].dropna().count()
        siteinfo_df.loc[ix, 'PREC/YR'] = sitedata[prec_var].sum() / n_years
        siteinfo_df.loc[ix, 'PREC_N_RECORDS'] = sitedata[prec_var].dropna().count()

        # SWC is completely missing for some sites
        if site['SWC_VAR'] != '-MISSING-':
            siteinfo_df.loc[ix, 'SWC_AVG'] = sitedata[swc_var].mean()
            siteinfo_df.loc[ix, 'SWC_N_RECORDS'] = sitedata[swc_var].dropna().count()
        else:
            siteinfo_df.loc[ix, 'SWC_AVG'] = '-MISSING-'
            siteinfo_df.loc[ix, 'SWC_N_RECORDS'] = '-MISSING-'


    return siteinfo_df
