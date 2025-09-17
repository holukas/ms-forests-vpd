import os
from pathlib import Path

import numpy as np
import pandas as pd
from diive.core.funcs.funcs import filter_strings_by_elements
from diive.core.io.filereader import search_files

pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)


def merge_site_info(allsites_fxn_cp, allsites_icos, allsites_amf, allsites_fxn, allsites_jpf):
    allsites = pd.concat([allsites_fxn_cp, allsites_icos, allsites_amf, allsites_fxn, allsites_jpf],
                         axis=0, ignore_index=True)
    allsites = allsites.reset_index(drop=True)
    allsites = allsites.fillna(np.nan)
    # print(allsites)

    allsites_combined = pd.DataFrame(columns=allsites.columns)

    uniq_sites = list(set(allsites['SITE'].tolist()))

    for ix, u in enumerate(uniq_sites):
        row = None
        _df = allsites.loc[allsites['SITE'] == u, :]
        n_records = len(_df)
        has_icos = any(_df['ORIGIN'] == 'ICOS')
        has_fxn_cp = any(_df['ORIGIN'] == 'FLUXNET_CP')
        has_fxn_org = any(_df['ORIGIN'] == 'FLUXNET_ORG')
        has_amf = any(_df['ORIGIN'] == 'AMERIFLUX')
        has_jpf = any(_df['ORIGIN'] == 'JAPANFLUX')

        if n_records > 1:

            # ICOS data + FLUXNET_CP data + FLUXNET data (in this order)
            if has_icos:
                row = _df.loc[_df['ORIGIN'] == 'ICOS']
                row = row.set_index('SITE', drop=False)
                originstr = "ICOS"
                if has_fxn_cp:
                    row_fxn_cp = _df.loc[_df['ORIGIN'] == 'FLUXNET_CP']
                    row_fxn_cp = row_fxn_cp.set_index('SITE', drop=False)
                    row = row.fillna(row_fxn_cp)
                    originstr += "+FLUXNET_CP"
                if has_fxn_org:
                    row_fxn = _df.loc[_df['ORIGIN'] == 'FLUXNET_ORG']
                    row_fxn = row_fxn.set_index('SITE', drop=False)
                    row = row.fillna(row_fxn)
                    originstr += "+FLUXNET_ORG"
                row['ORIGIN'] = originstr



            # FLUXNET_CP data + FLUXNET data
            elif has_fxn_cp:
                row = _df.loc[_df['ORIGIN'] == 'FLUXNET_CP']
                row = row.set_index('SITE', drop=False)
                originstr = "FLUXNET_CP"
                if has_fxn_org:
                    row_fxn = _df.loc[_df['ORIGIN'] == 'FLUXNET_ORG']
                    row_fxn = row_fxn.set_index('SITE', drop=False)
                    row = row.fillna(row_fxn)
                    originstr += "+FLUXNET_ORG"
                row['ORIGIN'] = originstr

            # AMERIFLUX data + FLUXNET data
            elif has_amf:
                row = _df.loc[_df['ORIGIN'] == 'AMERIFLUX']
                row = row.set_index('SITE', drop=False)
                originstr = "AMERIFLUX"
                if has_fxn_org:
                    row_fxn = _df.loc[_df['ORIGIN'] == 'FLUXNET_ORG']
                    row_fxn = row_fxn.set_index('SITE', drop=False)
                    row = row.fillna(row_fxn)
                    originstr += "+FLUXNET_ORG"
                row['ORIGIN'] = originstr

            # JAPANFLUX data + FLUXNET data
            elif has_jpf:
                row = _df.loc[_df['ORIGIN'] == 'JAPANFLUX']
                row = row.set_index('SITE', drop=False)
                originstr = "JAPANFLUX"
                if has_fxn_org:
                    row_fxn = _df.loc[_df['ORIGIN'] == 'FLUXNET_ORG']
                    row_fxn = row_fxn.set_index('SITE', drop=False)
                    row = row.fillna(row_fxn)
                    originstr += "+FLUXNET_ORG"
                row['ORIGIN'] = originstr


        # Only 1 available
        elif n_records == 1:
            if not _df.loc[_df['ORIGIN'] == 'ICOS'].empty:
                row = _df.loc[_df['ORIGIN'] == 'ICOS']
            elif not _df.loc[_df['ORIGIN'] == 'FLUXNET_CP'].empty:
                row = _df.loc[_df['ORIGIN'] == 'FLUXNET_CP']
            elif not _df.loc[_df['ORIGIN'] == 'FLUXNET_ORG'].empty:
                row = _df.loc[_df['ORIGIN'] == 'FLUXNET_ORG']
            elif not _df.loc[_df['ORIGIN'] == 'AMERIFLUX'].empty:
                row = _df.loc[_df['ORIGIN'] == 'AMERIFLUX']
            elif not _df.loc[_df['ORIGIN'] == 'JAPANFLUX'].empty:
                row = _df.loc[_df['ORIGIN'] == 'JAPANFLUX']

        else:
            raise Exception(f"{n_records} entries not allowed, only 1,2 or 3 allowed for each site.")

        allsites_combined = pd.concat([allsites_combined, row], axis=0, ignore_index=True)

    cols = [c for c in allsites_combined.columns if not str(c).startswith('_')]
    auxcols = [cols.append(c) for c in allsites_combined.columns if str(c).startswith('_')]
    allsites_combined = allsites_combined[cols]
    allsites_combined = allsites_combined.sort_values(by='SITE', inplace=False, ascending=True, ignore_index=True)

    return allsites_combined


def get_site_info_icos(pattern_dir, searchdir, pattern_file):
    # Get info for ICOS sites
    icos = SiteList(searchdir=searchdir, identifiers=pattern_dir, pattern_file=pattern_file, origin='ICOS')
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

        elev_icos = float(info_icos[info_icos['VARIABLE'] == 'LOCATION_ELEV']['DATAVALUE'].iloc[0])
        lon_icos = float(info_icos[info_icos['VARIABLE'] == 'LOCATION_LONG']['DATAVALUE'].iloc[0])
        lat_icos = float(info_icos[info_icos['VARIABLE'] == 'LOCATION_LAT']['DATAVALUE'].iloc[0])
        igbp_icos = str(info_icos[info_icos['VARIABLE'] == 'IGBP']['DATAVALUE'].iloc[0])

        allsites_icos.loc[allsites_icos['SITE'] == site, 'ELEVATION'] = elev_icos
        allsites_icos.loc[allsites_icos['SITE'] == site, 'LON'] = lon_icos
        allsites_icos.loc[allsites_icos['SITE'] == site, 'LAT'] = lat_icos
        allsites_icos.loc[allsites_icos['SITE'] == site, 'IGBP'] = igbp_icos
    return allsites_icos


def get_site_info_fxn_cp(searchdir, pattern_dir, infofile, pattern_file) -> pd.DataFrame:
    # Get info for FLUXNET sites
    fxn = SiteList(searchdir=searchdir, identifiers=pattern_dir, origin="FLUXNET_CP", pattern_file=pattern_file)
    fxn.run()
    allsites_fxn = fxn.get_site_info()

    if allsites_fxn.empty:
        return allsites_fxn

    # Read CSV with additional site info from EFDC / FLUXNET
    siteinfo_fxn = pd.read_csv(infofile)
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


def get_site_info_japanflux(searchdir, pattern_dir, infofile, origin, pattern_file) -> pd.DataFrame:
    """JapanFlux2024"""

    # Read site info from JapanFlux2024
    info = pd.read_csv(infofile)
    sitelist = SiteList(searchdir=searchdir, identifiers=pattern_dir, origin=origin, pattern_file=pattern_file,
                        info=info)
    sitelist.run()
    allsites = sitelist.get_site_info()

    for ix, row in allsites.iterrows():
        site = row['SITE']
        sitelocs = info['Site Code'] == site
        siteinfo = info[sitelocs].copy()

        try:
            elev = siteinfo['Elevation'].iloc[0]
            elev = elev.replace('m', '')
            elev = float(elev)
        except IndexError:
            elev = np.nan

        try:
            lon = float(siteinfo['Longitude'].iloc[0])
        except IndexError:
            lon = np.nan

        try:
            lat = float(siteinfo['Latitude'].iloc[0])
        except IndexError:
            lat = np.nan

        try:
            igbp = str(siteinfo['IGBP (land use)'].iloc[0])
        except IndexError:
            igbp = np.nan

        allsites.loc[allsites['SITE'] == site, 'ELEVATION'] = elev
        allsites.loc[allsites['SITE'] == site, 'LON'] = lon
        allsites.loc[allsites['SITE'] == site, 'LAT'] = lat
        allsites.loc[allsites['SITE'] == site, 'IGBP'] = igbp
    return allsites


def get_site_info_fluxnet_ameriflux(searchdir, pattern_dir, infofile, origin, pattern_file) -> pd.DataFrame:
    """FLUXNET_ORG and AMERIFLUX files have the same structure."""
    # Get info for AMERIFLUX sites
    sitelist = SiteList(searchdir=searchdir, identifiers=pattern_dir, origin=origin, pattern_file=pattern_file)
    sitelist.run()
    allsites = sitelist.get_site_info()

    # Read CSV with additional site info from EFDC / FLUXNET

    info = pd.read_csv(infofile)

    for ix, row in allsites.iterrows():
        site = row['SITE']
        sitelocs = info['SITE_ID'] == site
        siteinfo = info[sitelocs].copy()

        try:
            elev = float(siteinfo[siteinfo['VARIABLE'] == 'LOCATION_ELEV']['DATAVALUE'].iloc[0])
        except IndexError:
            elev = np.nan

        try:
            lon = float(siteinfo[siteinfo['VARIABLE'] == 'LOCATION_LONG']['DATAVALUE'].iloc[0])
        except IndexError:
            lon = np.nan

        try:
            lat = float(siteinfo[siteinfo['VARIABLE'] == 'LOCATION_LAT']['DATAVALUE'].iloc[0])
        except IndexError:
            lat = np.nan

        try:
            igbp = str(siteinfo[siteinfo['VARIABLE'] == 'IGBP']['DATAVALUE'].iloc[0])
        except IndexError:
            igbp = np.nan

        allsites.loc[allsites['SITE'] == site, 'ELEVATION'] = elev
        allsites.loc[allsites['SITE'] == site, 'LON'] = lon
        allsites.loc[allsites['SITE'] == site, 'LAT'] = lat
        allsites.loc[allsites['SITE'] == site, 'IGBP'] = igbp
    return allsites


class SiteList:

    def __init__(self,
                 searchdir: str,
                 identifiers: list,
                 pattern_file: str,
                 origin: str,
                 info: pd.DataFrame = None):

        self.searchdir = searchdir
        self.identifiers = identifiers
        self.pattern_file = pattern_file
        self.origin = origin
        self.info_df = info  # Only needed for JPF, contains site names in connection with ID number

        self.valid_folders = []
        self.sites = pd.DataFrame()

    def get_site_info(self) -> pd.DataFrame:
        return self.sites

    def _search_folders(self) -> list:
        """Search for folders in searchdir that contain the identifiers."""
        root = self.searchdir
        found_folders = [f.name for f in os.scandir(root) if f.is_dir()]
        found_folders = [os.path.join(root, folder) for folder in found_folders]
        found_folders = [str(Path(folder)) for folder in found_folders]
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
            site = None

            dirpath_fxn_cp = None
            dirname_fxn_cp = None
            filepath_fxn_cp = None

            dirpath_fxn_org = None
            dirname_fxn_org = None
            filepath_fxn_org = None

            dirpath_icos = None
            dirname_icos = None
            filepath_icos = None

            dirpath_amf = None
            dirname_amf = None
            filepath_amf = None

            dirpath_jpf = None
            dirname_jpf = None
            filepath_jpf = None

            if self.origin == 'FLUXNET_CP':
                # FLX_FI-Var_FLUXNET2015_FULLSET_HH_2017-2023_1-3.csv
                dirpath_fxn_cp = Path(v)
                dirname_fxn_cp = dirpath_fxn_cp.name
                site = self._extract_sitename(dirname=dirname_fxn_cp)
                # site = self._extract_sitename(dirname=dirname_fxn_cp)
                # filepattern = 'FLX_*_FLUXNET2015_FULLSET_HH_*.csv'
                foundfile = search_files(searchdirs=str(dirpath_fxn_cp), pattern=self.pattern_file)
                filepath_fxn_cp = foundfile[0]

            elif self.origin == 'JAPANFLUX':
                dirpath_jpf = Path(v)
                dirname_jpf = dirpath_jpf.name
                id_jpf = str(dirpath_jpf.name).replace('JPF_', '')  # Site ID number
                site = self.info_df.loc[self.info_df['Metadata ID'] == id_jpf, 'Site Code'].values[0]
                # filepattern = 'FLX_*_JapanFLUX2024_ALLVARS_HH_*.csv'
                foundfile = search_files(searchdirs=str(dirpath_jpf), pattern=self.pattern_file)
                filepath_jpf = foundfile[0]

            elif self.origin == 'ICOS':
                dirpath_icos = Path(v)
                dirname_icos = dirpath_icos.name
                site = self._extract_sitename(dirname=dirname_icos)
                # filepattern = 'ICOSETC_*_FLUXNET_HH_L2.csv'
                foundfile = search_files(searchdirs=str(dirpath_icos), pattern=self.pattern_file)
                filepath_icos = foundfile[0]

            elif self.origin == 'AMERIFLUX':
                dirpath_amf = Path(v)
                dirname_amf = dirpath_amf.name
                site = self._extract_sitename(dirname=dirname_amf)
                # filepattern = 'AMF_*_FLUXNET_FULLSET_HH_*.csv'
                foundfile = search_files(searchdirs=str(dirpath_amf), pattern=self.pattern_file)
                if not foundfile:
                    # Few sites have hourly instead of half-hourly data
                    filepattern = 'AMF_*_FLUXNET_FULLSET_HR_*.csv'
                    foundfile = search_files(searchdirs=str(dirpath_amf), pattern=filepattern)
                filepath_amf = foundfile[0]

            elif self.origin == 'FLUXNET_ORG':
                dirpath_fxn_org = Path(v)
                dirname_fxn_org = dirpath_fxn_org.name
                site = self._extract_sitename(dirname=dirname_fxn_org)
                # filepattern = 'FLX_*_FLUXNET2015_FULLSET_HH_*.csv'
                foundfile = search_files(searchdirs=str(dirpath_fxn_org), pattern=self.pattern_file)
                if not foundfile:
                    # Few sites have hourly instead of half-hourly data
                    filepattern = 'FLX_*_FLUXNET2015_FULLSET_HR_*.csv'
                    foundfile = search_files(searchdirs=str(dirpath_fxn_org), pattern=filepattern)
                    # TODO check
                    # raise Exception("Script tried to find hourly data.")
                filepath_fxn_org = foundfile[0]

            d = {
                'SITE': [site],
                'ORIGIN': self.origin,
                '_DIRNAME_FXN_CP': [dirname_fxn_cp],
                '_DIRPATH_FXN_CP': [dirpath_fxn_cp],
                '_FILEPATH_FXN_CP': [filepath_fxn_cp],
                '_DIRNAME_FXN_ORG': [dirname_fxn_org],
                '_DIRPATH_FXN_ORG': [dirpath_fxn_org],
                '_FILEPATH_FXN_ORG': [filepath_fxn_org],
                '_DIRNAME_ICOS': [dirname_icos],
                '_DIRPATH_ICOS': [dirpath_icos],
                '_FILEPATH_ICOS': [filepath_icos],
                '_DIRNAME_AMF': [dirname_amf],
                '_DIRPATH_AMF': [dirpath_amf],
                '_FILEPATH_AMF': [filepath_amf],
                '_DIRNAME_JPF': [dirname_jpf],
                '_DIRPATH_JPF': [dirpath_jpf],
                '_FILEPATH_JPF': [filepath_jpf]
            }
            site = pd.DataFrame.from_dict(d, orient='columns')
            sites = pd.concat([sites, site], axis=0, ignore_index=True)
        return sites

    def run(self):
        self.valid_folders = self._search_folders()
        self.sites = self._collect_info()

def consolidate_duplicate_site_entries_amf(site_info_ameriflux):
    # Get all rows that are duplicates (including the first occurrence)
    duplicates = site_info_ameriflux[site_info_ameriflux['SITE'].duplicated(keep=False)]
    # Get the unique site names from the duplicated rows
    duplicate_site_names = duplicates['SITE'].unique()

    # Define a function to combine unique values into a list
    def combine_unique_values(series):
        unique_values = series.dropna().unique().tolist()
        if len(unique_values) == 1:
            return unique_values[0]
        if not unique_values:
            return None
        return unique_values

    for d in duplicate_site_names:
        subset = site_info_ameriflux.loc[site_info_ameriflux['SITE'] == d].copy()

        # Remove the duplicates from site info
        index_to_drop = site_info_ameriflux[site_info_ameriflux['SITE'] == d].index
        site_info_ameriflux = site_info_ameriflux.drop(index_to_drop, inplace=False)

        # Group by the 'SITE' column and apply the aggregation
        # Create a dictionary for aggregation, applying the custom function to all columns except 'SITE'
        agg_dict = {col: combine_unique_values for col in subset.columns if col != 'SITE'}
        combined_entry = subset.groupby('SITE', as_index=False).agg(agg_dict)

        # Add new (consolidated) record back to dataframe
        site_info_ameriflux = pd.concat([site_info_ameriflux, combined_entry], ignore_index=True)

    return site_info_ameriflux
