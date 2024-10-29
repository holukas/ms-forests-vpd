import pandas as pd
from diive.core.io.filereader import search_files, ReadFileType
from diive.core.io.files import load_parquet
from diive.core.io.files import save_parquet


def convert_datafiles_to_parquet(filepatterns: list, filetype: str, searchdir: str, outpath: str):
    filelist = []
    for filepattern in filepatterns:
        _filelist = search_files(
            searchdirs=searchdir,
            pattern=filepattern)
        filelist = filelist + _filelist

    for f in filelist:
        filename = f.name
        loaddatafile = ReadFileType(filetype=filetype, filepath=f, data_nrows=None)
        data_df, metadata_df = loaddatafile.get_filedata()
        filepath = save_parquet(filename=filename, data=data_df, outpath=outpath)



def make_sitelist(filepaths: list, source_info: str, testing: bool = False):
    sites_df = pd.DataFrame()
    for ix, ip in enumerate(filepaths):
        if testing and ix > 1:
            break
        splits = ip.name.split('_')
        site = splits[1]
        igbp = ip.parent.parent.name
        print(f"Reading {site} ...")
        df = load_parquet(filepath=ip)
        date_first = df.index[0]
        date_last = df.index[-1]
        date_first_str = str(date_first.strftime('%d %b %Y'))
        date_last_str = str(date_last.strftime('%d %b %Y'))
        n_records = len(df.index)
        n_years = (date_last.year - date_first.year) + 1

        info = {
            'SITE': [site],
            'IGBP': igbp,
            'DATE_FIRST': date_first_str,
            'DATE_LAST': date_last_str,
            'N_YEARS': n_years,
            'N_RECORDS': n_records,
            # 'TA_AVG': df['TA_F'].mean(),
            # 'VPD_AVG': df['VPD_F'].mean(),
            # 'PREC/YR': df['P_F'].sum() / n_years,
            '.source': source_info,
            '.filepath': ip
        }

        site_df = pd.DataFrame.from_dict(info, orient='columns')
        sites_df = pd.concat([sites_df, site_df], axis=0, ignore_index=True)
    return sites_df

# def collect_sitestats(filepattern: str, searchdirs: str, source_info: str, testing: bool = False) -> pd.DataFrame:
#     filepaths = search_files(searchdirs=searchdirs, pattern=filepattern)
#
#     sites_df = pd.DataFrame()
#     for ix, filepath in enumerate(filepaths):
#         if testing and ix > 2:
#             break
#         _filename = Path(filepath).name
#         splits = _filename.split('_')
#         site = splits[1]
#         print(f"Reading {site} ...")
#         df = load_parquet(filepath=filepath)
#         # [print(c) for c in df.columns if "P_F" in c];
#         date_first = df.index[0]
#         date_last = df.index[-1]
#         date_first_str = str(date_first.strftime('%d %b %Y'))
#         date_last_str = str(date_last.strftime('%d %b %Y'))
#         n_records = len(df.index)
#         n_years = (date_last.year - date_first.year) + 1
#
#         d = {
#             'SITE': [site],
#             'IGBP': '-not-found-',
#             'DATE_FIRST': date_first_str,
#             'DATE_LAST': date_last_str,
#             'N_YEARS': n_years,
#             'N_RECORDS': n_records,
#             'TA_AVG': df['TA_F'].mean(),
#             'TA_MIN': df['TA_F'].min(),
#             'TA_MAX': df['TA_F'].max(),
#             'VPD_AVG': df['VPD_F'].mean(),
#             'VPD_MIN': df['VPD_F'].min(),
#             'VPD_MAX': df['VPD_F'].max(),
#             'PREC/YR': df['P_F'].sum() / n_years,
#             '.source_icos': source_info,
#             '.source_fluxnet': source_info,
#             '.filepath': filepath
#         }
#         site_df = pd.DataFrame.from_dict(d, orient='columns')
#         sites_df = pd.concat([sites_df, site_df], axis=0, ignore_index=True)
#     return sites_df
#
#
# def collect_siteinfo(sites_df: pd.DataFrame, filepattern: str, sourcedir: str, testing: bool = False):
#     filepaths = search_files(searchdirs=sourcedir, pattern=filepattern)
#     for ix, filepath in enumerate(filepaths):
#         if testing and ix > 2:
#             break
#         _filename = Path(filepath).name
#         splits = _filename.split('_')
#         site = splits[1]
#         print(f"Reading site info for site {site} ...")
#         df = pd.read_csv(filepath, on_bad_lines='warn')
#         igbp = df.loc[df['VARIABLE'] == 'IGBP', 'DATAVALUE'].values[0]
#         location_lat = df.loc[df['VARIABLE'] == 'LOCATION_LAT', 'DATAVALUE'].values[0]
#         location_long = df.loc[df['VARIABLE'] == 'LOCATION_LONG', 'DATAVALUE'].values[0]
#         location_elev = df.loc[df['VARIABLE'] == 'LOCATION_ELEV', 'DATAVALUE'].values[0]
#
#         sites_df.loc[sites_df['SITE'] == site, 'IGBP'] = igbp
#         sites_df.loc[sites_df['SITE'] == site, 'LOCATION_LAT'] = location_lat
#         sites_df.loc[sites_df['SITE'] == site, 'LOCATION_LONG'] = location_long
#         sites_df.loc[sites_df['SITE'] == site, 'LOCATION_ELEV'] = location_elev
#     return sites_df
