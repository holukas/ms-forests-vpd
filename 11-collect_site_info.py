from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from diive.core.io.filereader import search_files
from diive.core.io.files import load_parquet

pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

SEARCHDIRS = r"L:\Sync\luhk_work\40 - DATA\Datasets\2024 - ICOS - Ecosystem final quality (L2) product in ETC-Archive format - release 2024-1\2-FLUXNET_HH_PARQUET"
IDENTIFIERS = ['ICOSETC_', '_FLUXNET_HH_L2', '.csv.parquet']

filepaths = search_files(searchdirs=SEARCHDIRS, pattern=r"ICOSETC_*_FLUXNET_HH_L2.csv.parquet")

sites_df = pd.DataFrame()
for ix, filepath in enumerate(filepaths):
    if ix > 2:
        break
    _filename = Path(filepath).name
    splits = _filename.split('_')
    site = splits[1]
    print(f"Reading {site} ...")
    df = load_parquet(filepath=filepath)
    # [print(c) for c in df.columns if "P_F" in c];
    date_first = df.index[0]
    date_last = df.index[-1]
    date_first_str = str(date_first.strftime('%d %b %Y'))
    date_last_str = str(date_last.strftime('%d %b %Y'))
    n_records = len(df.index)
    n_years = (date_last.year - date_first.year) + 1
    d = {
        'SITE': [site],
        'IGBP': '-not-found-',
        'DATE_FIRST': date_first_str,
        'DATE_LAST': date_last_str,
        'N_YEARS': n_years,
        'N_RECORDS': n_records,
        'TA_AVG': df['TA_F'].mean(),
        'TA_MIN': df['TA_F'].min(),
        'TA_MAX': df['TA_F'].max(),
        'VPD_AVG': df['VPD_F'].mean(),
        'VPD_MIN': df['VPD_F'].min(),
        'VPD_MAX': df['VPD_F'].max(),
        'PREC/YR': df['P_F'].sum() / n_years,
        '.source': 'ICOS_L2',
        '.filepath': filepath
    }
    site_df = pd.DataFrame.from_dict(d, orient='columns')
    sites_df = pd.concat([sites_df, site_df], axis=0, ignore_index=True)
    # df['TA_F'].plot(x_compat=True)
    plt.show()

# # SITE INFO
# CONFIGFILEPATH = r"F:\Sync\luhk_work\40 - DATA\Datasets\2024 - ICOS - Ecosystem final quality (L2) product in ETC-Archive format - release 2024-1\3-ARCHIVE_L2"
# filepaths = search_files(searchdirs=CONFIGFILEPATH, pattern=r"ICOSETC_*_SITEINFO_L2.csv")
# for ix, filepath in enumerate(filepaths):
#     if ix > 2:
#         break
#     _filename = Path(filepath).name
#     splits = _filename.split('_')
#     site = splits[1]
#     print(f"Reading site info for site {site} ...")
#     df = pd.read_csv(filepath, on_bad_lines='warn')
#     igbp = df.loc[df['VARIABLE'] == 'IGBP', 'DATAVALUE'].values[0]
#     location_lat = df.loc[df['VARIABLE'] == 'LOCATION_LAT', 'DATAVALUE'].values[0]
#     location_long = df.loc[df['VARIABLE'] == 'LOCATION_LONG', 'DATAVALUE'].values[0]
#     location_elev = df.loc[df['VARIABLE'] == 'LOCATION_ELEV', 'DATAVALUE'].values[0]
#
#     sites_df.loc[sites_df['SITE'] == site, 'IGBP'] = igbp
#     sites_df.loc[sites_df['SITE'] == site, 'LOCATION_LAT'] = location_lat
#     sites_df.loc[sites_df['SITE'] == site, 'LOCATION_LONG'] = location_long
#     sites_df.loc[sites_df['SITE'] == site, 'LOCATION_ELEV'] = location_elev

print(sites_df)
sites_df.to_csv("OUT/11.1-site_info.csv")
