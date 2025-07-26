import pandas as pd
from diive.core.io.files import load_parquet


def basic_stats(siteinfo_df) -> pd.DataFrame:
    swincol = 'SW_IN_F'
    tacol = 'TA_F'
    vpdcol = 'VPD_F'
    preccol = 'P_F'
    swccol = 'SWC_F_MDS_1'

    _df = siteinfo_df.copy()
    for ix, row in _df.iterrows():

        # if row['SITE'] != 'BE-Bra':
        #     continue

        print(f"\nLoading data for site {row['SITE']} ...")

        filepath = row['_FILEPATH_PARQUET']
        sitedata = load_parquet(filepath)

        # # todo make list of available vars
        # sitevars_available = sitedata.columns
        # for sv in sitevars_available:
        #     # sitevars_all.loc[row['SITE'], sv] = sitedata[sv].count()

        # Collect info
        date_first = sitedata.index[0]
        date_last = sitedata.index[-1]
        date_first_str = str(date_first.strftime('%d %b %Y'))
        date_last_str = str(date_last.strftime('%d %b %Y'))
        n_records = len(sitedata.index)
        n_years = (date_last.year - date_first.year) + 1

        siteinfo_df.loc[ix, 'DATE_FIRST'] = date_first_str
        siteinfo_df.loc[ix, 'DATE_LAST'] = date_last_str
        siteinfo_df.loc[ix, 'N_YEARS'] = n_years
        siteinfo_df.loc[ix, 'N_RECORDS'] = n_records
        siteinfo_df.loc[ix, 'SW_IN_AVG'] = sitedata[swincol].mean()
        siteinfo_df.loc[ix, 'SW_IN_N_RECORDS'] = sitedata[swincol].dropna().count()
        siteinfo_df.loc[ix, 'TA_AVG'] = sitedata[tacol].mean()
        siteinfo_df.loc[ix, 'TA_N_RECORDS'] = sitedata[tacol].dropna().count()
        siteinfo_df.loc[ix, 'VPD_AVG'] = sitedata[vpdcol].mean()
        siteinfo_df.loc[ix, 'VPD_N_RECORDS'] = sitedata[vpdcol].dropna().count()
        siteinfo_df.loc[ix, 'PREC/YR'] = sitedata[preccol].sum() / n_years
        siteinfo_df.loc[ix, 'PREC_N_RECORDS'] = sitedata[preccol].dropna().count()
        siteinfo_df.loc[ix, 'SWC_1_AVG'] = sitedata[swccol].mean()
        siteinfo_df.loc[ix, 'SWC_1_N_RECORDS'] = sitedata[swccol].dropna().count()

    return siteinfo_df
