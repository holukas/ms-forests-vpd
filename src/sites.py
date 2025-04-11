from pathlib import Path
from diive.core.io.filereader import search_files, ReadFileType
import numpy as np
import pandas as pd
from diive.core.funcs.funcs import filter_strings_by_elements
from diive.core.io.filereader import search_folders

pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)


class FluxnetIcosSiteList:

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
            dirpath_fxn = np.nan
            dirname_fxn = np.nan
            dirpath_icos = np.nan
            dirname_icos = np.nan
            filepath_icos = np.nan
            filepath_fxn = np.nan
            site = np.nan

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

            d = {
                'SITE': [site],
                'ORIGIN': self.origin,
                '_DIRNAME_FXN': [dirname_fxn],
                '_DIRPATH_FXN': [dirpath_fxn],
                '_DIRNAME_ICOS': [dirname_icos],
                '_DIRPATH_ICOS': [dirpath_icos],
                '_FILEPATH_ICOS': [filepath_icos],
                '_FILEPATH_FXN': [filepath_fxn],
            }
            site = pd.DataFrame.from_dict(d, orient='columns')
            sites = pd.concat([sites, site], axis=0, ignore_index=True)
        return sites

    def run(self):
        self.valid_folders = self._search_folders()
        self.sites = self._collect_info()
