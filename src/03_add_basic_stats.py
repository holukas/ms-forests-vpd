import numpy as np
import pandas as pd
from diive.core.io.files import load_parquet


df = pd.read_csv('../OUT/02_siteinfo.csv')
df = df.fillna(np.nan)
print(df)

# # Number of IGBPs
# # {'ENF': 72, 'DBF': 47, 'MF': 12, 'DNF': 2, 'EBF': 3, 'OSH': 1}
# from collections import Counter
# counts_igbps = Counter(df['IGBP'])
# print(dict(counts_igbps))



# sitevars_all = pd.DataFrame()
# sitevars = dict()

_df = df.copy()
for ix, row in _df.iterrows():

    print(f"\nLoading data for site {row['SITE']} ...")

    filepath = row['_FILEPATH_PARQUET']
    sitedata = load_parquet(filepath)

    # todo make list of available vars
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

    df.loc[ix, 'DATE_FIRST'] = date_first_str
    df.loc[ix, 'DATE_LAST'] = date_last_str
    df.loc[ix, 'N_YEARS'] = n_years
    df.loc[ix, 'N_RECORDS'] = n_records
    df.loc[ix, 'TA_AVG'] = sitedata['TA_F'].mean()
    df.loc[ix, 'VPD_AVG'] = sitedata['VPD_F'].mean()
    df.loc[ix, 'PREC/YR'] = sitedata['P_F'].sum() / n_years

df.to_csv("../OUT/03_siteinfo.csv", index=False)
