from pathlib import Path

import numpy as np
import pandas as pd
from diive.core.funcs.funcs import filter_strings_by_elements
from diive.core.io.filereader import search_files
from diive.core.io.filereader import search_folders

pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)


def merge_site_info(allsites_fxn, allsites_icos, allsites_amf, outfile):
    allsites = pd.concat([allsites_fxn, allsites_icos, allsites_amf], axis=0, ignore_index=True)
    allsites = allsites.reset_index(drop=True)
    allsites = allsites.fillna(np.nan)
    # print(allsites)

    allsites_combined = pd.DataFrame(columns=allsites.columns)

    uniq_sites = list(set(allsites['SITE'].tolist()))

    for ix, u in enumerate(uniq_sites):
        row = None
        _df = allsites.loc[allsites['SITE'] == u, :]
        n_records = len(_df)

        # Both ICOS and FLUXNET data available
        if n_records == 2:
            row = _df.loc[_df['ORIGIN'] == 'ICOS']
            row = row.set_index('SITE', drop=False)  # Set index for .fillna()
            row_fxn = _df.loc[_df['ORIGIN'] == 'FLUXNET']
            row_fxn = row_fxn.set_index('SITE', drop=False)
            row = row.fillna(row_fxn)
            row['ORIGIN'] = 'ICOS+FLUXNET'

        # Only 1 available
        elif n_records == 1:
            if not _df.loc[_df['ORIGIN'] == 'ICOS'].empty:
                row = _df.loc[_df['ORIGIN'] == 'ICOS']
                # row = row.set_index('SITE', drop=False)
            elif not _df.loc[_df['ORIGIN'] == 'FLUXNET'].empty:
                row = _df.loc[_df['ORIGIN'] == 'FLUXNET']
                # row = row.set_index('SITE', drop=False)
            elif not _df.loc[_df['ORIGIN'] == 'AMERIFLUX'].empty:
                row = _df.loc[_df['ORIGIN'] == 'AMERIFLUX']
                # row = row.set_index('SITE', drop=False)

        else:
            raise Exception(f"{n_records} entries not allowed, only 1 or 2.")

        allsites_combined = pd.concat([allsites_combined, row], axis=0, ignore_index=True)

    cols = [c for c in allsites_combined.columns if not str(c).startswith('_')]
    auxcols = [cols.append(c) for c in allsites_combined.columns if str(c).startswith('_')]
    allsites_combined = allsites_combined[cols]
    allsites_combined = allsites_combined.sort_values(by='SITE', inplace=False, ascending=True, ignore_index=True)

    return allsites_combined


def get_site_info_icos(pattern_icos, searchdir):
    # Get info for ICOS sites
    icos = SiteList(searchdir=searchdir, identifiers=pattern_icos, origin='ICOS')
    icos.run()
    allsites_icos = icos.get_site_info()

    # Read CSV with additional site info from ICOS
    _allsites_icos = allsites_icos.copy()
    for ix, row in _allsites_icos.iterrows():
        site = row['SITE']

        # Get info from this site's ICOS SITEINFO file
        infofile_icos = f"ICOSETC_{site}_SITEINFO_L2.csv"
        infofile_icos_path = Path(row['_DIRPATH_ICOS']) / infofile_icos
        info_icos = pd.read_csv(infofile_icos_path)

        # Check if sites match
        checksite = str(list(set(info_icos['SITE_ID'].tolist()))[0])
        if site != checksite:
            raise Exception("Site does not match.")

        elev_icos = info_icos[info_icos['VARIABLE'] == 'LOCATION_ELEV']['DATAVALUE'].iloc[0]
        lon_icos = info_icos[info_icos['VARIABLE'] == 'LOCATION_LONG']['DATAVALUE'].iloc[0]
        lat_icos = info_icos[info_icos['VARIABLE'] == 'LOCATION_LAT']['DATAVALUE'].iloc[0]
        igbp_icos = info_icos[info_icos['VARIABLE'] == 'IGBP']['DATAVALUE'].iloc[0]

        allsites_icos.loc[allsites_icos['SITE'] == site, 'ELEVATION'] = elev_icos
        allsites_icos.loc[allsites_icos['SITE'] == site, 'LON'] = lon_icos
        allsites_icos.loc[allsites_icos['SITE'] == site, 'LAT'] = lat_icos
        allsites_icos.loc[allsites_icos['SITE'] == site, 'IGBP'] = igbp_icos
    return allsites_icos


def get_site_info_fxn(searchdir, pattern_fxn, infofile_fxn) -> pd.DataFrame:
    # Get info for FLUXNET sites
    fxn = SiteList(searchdir=searchdir, identifiers=pattern_fxn, origin="FLUXNET")
    fxn.run()
    allsites_fxn = fxn.get_site_info()

    if allsites_fxn.empty:
        return allsites_fxn

    # Read CSV with additional site info from EFDC / FLUXNET
    siteinfo_fxn = pd.read_csv(infofile_fxn)
    siteinfo_fxn = siteinfo_fxn[['Site Code', 'IGBP Code', 'Site Latitude', 'Site Longitude']].copy()

    # Add info to site df
    allsites_fxn = allsites_fxn.merge(siteinfo_fxn, left_on='SITE', right_on='Site Code')
    allsites_fxn = allsites_fxn.drop(columns='Site Code', inplace=False)

    # Renaming
    rename_dict = {
        'IGBP Code': 'IGBP',
        'Site Latitude': 'LAT',
        'Site Longitude': 'LON',
    }
    allsites_fxn = allsites_fxn.rename(columns=rename_dict, inplace=False)
    allsites_fxn = allsites_fxn.sort_values(by=['SITE'], ascending=True, inplace=False)
    return allsites_fxn


def get_site_info_ameriflux(searchdir, pattern_amf, infofile_amf) -> pd.DataFrame:
    # Get info for AMERIFLUX sites
    amf = SiteList(searchdir=searchdir, identifiers=pattern_amf, origin='AMERIFLUX')
    amf.run()
    allsites_amf = amf.get_site_info()

    # Read CSV with additional site info from EFDC / FLUXNET

    info_amf = pd.read_csv(infofile_amf)

    for ix, row in allsites_amf.iterrows():
        site = row['SITE']
        sitelocs = info_amf['SITE_ID'] == site
        siteinfo = info_amf[sitelocs].copy()

        try:
            elev_amf = siteinfo[siteinfo['VARIABLE'] == 'LOCATION_ELEV']['DATAVALUE'].iloc[0]
        except IndexError:
            elev_amf = np.nan

        try:
            lon_amf = siteinfo[siteinfo['VARIABLE'] == 'LOCATION_LONG']['DATAVALUE'].iloc[0]
        except IndexError:
            lon_amf = np.nan

        try:
            lat_amf = siteinfo[siteinfo['VARIABLE'] == 'LOCATION_LAT']['DATAVALUE'].iloc[0]
        except IndexError:
            lat_amf = np.nan

        try:
            igbp_amf = siteinfo[siteinfo['VARIABLE'] == 'IGBP']['DATAVALUE'].iloc[0]
        except IndexError:
            igbp_amf = np.nan

        allsites_amf.loc[allsites_amf['SITE'] == site, 'ELEVATION'] = elev_amf
        allsites_amf.loc[allsites_amf['SITE'] == site, 'LON'] = lon_amf
        allsites_amf.loc[allsites_amf['SITE'] == site, 'LAT'] = lat_amf
        allsites_amf.loc[allsites_amf['SITE'] == site, 'IGBP'] = igbp_amf
        return allsites_amf


class SiteList:

    def __init__(self,
                 searchdir: str,
                 identifiers: list,
                 origin: str):

        self.searchdir = searchdir
        self.identifiers = identifiers
        self.origin = origin

        self.valid_folders = []
        self.sites = pd.DataFrame()

    def get_site_info(self) -> pd.DataFrame:
        return self.sites

    def _search_folders(self) -> list:
        found_folders = search_folders(searchdirs=self.searchdir)
        valid_folders = filter_strings_by_elements(found_folders, self.identifiers)
        return valid_folders

    def _extract_sitename(self, dirname):
        site = dirname
        for i in self.identifiers:
            site = site.replace(i, "")
        # Split folder string to extract site info
        splits = site.split('_')
        site = splits[0]
        return site

    def _collect_info(self):
        sites = pd.DataFrame()

        for v in self.valid_folders:
            site = np.nan
            dirpath_fxn = np.nan
            dirname_fxn = np.nan
            filepath_fxn = np.nan
            dirpath_icos = np.nan
            dirname_icos = np.nan
            filepath_icos = np.nan
            dirpath_amf = np.nan
            dirname_amf = np.nan
            filepath_amf = np.nan

            if self.origin == 'FLUXNET':
                # FLX_FI-Var_FLUXNET2015_FULLSET_HH_2017-2023_1-3.csv
                dirpath_fxn = Path(v)
                dirname_fxn = dirpath_fxn.name
                site = self._extract_sitename(dirname=dirname_fxn)
                filepattern = 'FLX_*_FLUXNET2015_FULLSET_HH_*.csv'
                foundfile = search_files(searchdirs=str(dirpath_fxn), pattern=filepattern)
                filepath_fxn = foundfile[0]

            elif self.origin == 'ICOS':
                dirpath_icos = Path(v)
                dirname_icos = dirpath_icos.name
                site = self._extract_sitename(dirname=dirname_icos)
                filepattern = 'ICOSETC_*_FLUXNET_HH_L2.csv'
                foundfile = search_files(searchdirs=str(dirpath_icos), pattern=filepattern)
                filepath_icos = foundfile[0]

            elif self.origin == 'AMERIFLUX':
                dirpath_amf = Path(v)
                dirname_amf = dirpath_amf.name
                site = self._extract_sitename(dirname=dirname_amf)
                filepattern = 'AMF_*_FLUXNET_FULLSET_HH_*.csv'
                foundfile = search_files(searchdirs=str(dirpath_amf), pattern=filepattern)
                if not foundfile:
                    # Few sites have hourly instead of half-hourly data
                    filepattern = 'AMF_*_FLUXNET_FULLSET_HR_*.csv'
                    foundfile = search_files(searchdirs=str(dirpath_amf), pattern=filepattern)
                filepath_amf = foundfile[0]

            d = {
                'SITE': [site],
                'ORIGIN': self.origin,
                '_DIRNAME_FXN': [dirname_fxn],
                '_DIRPATH_FXN': [dirpath_fxn],
                '_FILEPATH_FXN': [filepath_fxn],
                '_DIRNAME_ICOS': [dirname_icos],
                '_DIRPATH_ICOS': [dirpath_icos],
                '_FILEPATH_ICOS': [filepath_icos],
                '_DIRNAME_AMF': [dirname_amf],
                '_DIRPATH_AMF': [dirpath_amf],
                '_FILEPATH_AMF': [filepath_amf]
            }
            site = pd.DataFrame.from_dict(d, orient='columns')
            sites = pd.concat([sites, site], axis=0, ignore_index=True)
        return sites

    def run(self):
        self.valid_folders = self._search_folders()
        self.sites = self._collect_info()
