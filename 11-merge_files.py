"""
Convert ICOS L2 and FLUXNET half-hourly CSV files to parquet
"""
from pathlib import Path

import pandas as pd
from diive.core.io.filereader import search_files
from diive.core.io.files import load_parquet, save_parquet


class MergeFiles:
    # Folder and name settings
    basedir = r"L:\Sync\luhk_work\40 - DATA\Datasets\2024 - FLUXNET ICOS FORESTS"
    ecosystems = ['DBF', 'DNF', 'EBF', 'ENF', 'MF']
    filepatterns = ["ICOSETC_*_FLUXNET_HH_L2.csv.parquet", "FLX_*_FLUXNET2015_FULLSET_HH_*.csv.parquet"]

    def __init__(self):
        self.parquetfilepaths = []
        self.icosfilepaths = []
        self.fxnfilepaths = []
        self.sitelist = pd.DataFrame()
        self.unique_sites = []
        self.sites_df = pd.DataFrame()

    def run(self):
        # Find filepaths to parquet files and store in dataframe
        self.parquetfilepaths, self.icosfilepaths, self.fxnfilepaths = self.find_filepaths()
        self.sitelist, self.unique_sites = self.make_sitelist()
        self.merge_data()

    def merge_data(self):
        for site in self.unique_sites:
            locs = self.sitelist['SITE'] == site
            siteinfo = self.sitelist[locs].copy()
            merged_df = pd.DataFrame()
            igbp = -9999
            sourcetxt = -9999

            # ICOS or FLUXNET data available
            if len(siteinfo) == 1:
                source = siteinfo['.source'].iloc[0]
                print(f"Single dataset only ({site}, {source})")
                if source == "ICOS":
                    merged_df, icos_igbp = self.loadfile(siteinfo=siteinfo, source=source)
                    fxndata, fxn_igbp = None, None
                    sourcetxt = "ICOS"
                    igbp = icos_igbp
                elif source == "FLUXNET":
                    icosdata, icos_igbp = None, None
                    merged_df, igbp = self.loadfile(siteinfo=siteinfo, source=source)
                    sourcetxt = "FXN"

            # ICOS and FLUXNET data available
            elif len(siteinfo) == 2:
                print(f"Two datasets ({site}, {siteinfo['.source'].tolist()})")
                # Load ICOS and FLUXNET data
                icosdata, icos_igbp = self.loadfile(siteinfo=siteinfo, source="ICOS")
                fxndata, fxn_igbp = self.loadfile(siteinfo=siteinfo, source="FLUXNET")
                # Check if IGBP info is the same for both
                if icos_igbp == fxn_igbp:
                    igbp = icos_igbp
                else:
                    raise ValueError("ICOS and FLUXNET IGBP do not match")
                # Keep records from FLUXNET data that are not in ICOS data
                keeplocs = fxndata.index < icosdata.index[0]
                fxndata_keep = fxndata[keeplocs].copy()
                # Merge ICOS and FLUXNET data
                merged_df = pd.concat([icosdata, fxndata_keep], axis=0)
                merged_df = merged_df.sort_index()
                sourcetxt = "ICOS+FXN"

            elif len(siteinfo) > 2:
                raise ValueError("Too many sites")

            # Save merged data to parquet file
            start = merged_df.index[0].year
            end = merged_df.index[-1].year
            outpath = Path(self.basedir) / igbp / "3-FLUXFILES_MERGED"
            outfilepath = save_parquet(filename=f"DATA-MERGED_{site}_{igbp}_{sourcetxt}_{start}-{end}",
                                       data=merged_df,
                                       outpath=outpath)

            # Collect info
            date_first = merged_df.index[0]
            date_last = merged_df.index[-1]
            date_first_str = str(date_first.strftime('%d %b %Y'))
            date_last_str = str(date_last.strftime('%d %b %Y'))
            n_records = len(merged_df.index)
            n_years = (date_last.year - date_first.year) + 1
            info = {
                'SITE': [site],
                'IGBP': igbp,
                'DATE_FIRST': date_first_str,
                'DATE_LAST': date_last_str,
                'N_YEARS': n_years,
                'N_RECORDS': n_records,
                'TA_AVG': merged_df['TA_F'].mean(),
                'VPD_AVG': merged_df['VPD_F'].mean(),
                'PREC/YR': merged_df['P_F'].sum() / n_years,
                '.source': sourcetxt,
                '.filepath': outfilepath
            }
            site_df = pd.DataFrame.from_dict(info, orient='columns')
            self.sites_df = pd.concat([self.sites_df, site_df], axis=0, ignore_index=True)
            self.sites_df.to_csv(r"OUT/11.2-sitelist.csv", index=False)

            # [print(v) for v in icosdata.columns if v in fxndata_keep.columns]

    @staticmethod
    def loadfile(siteinfo: pd.DataFrame, source: str) -> tuple[pd.DataFrame, str]:
        locs = siteinfo['.source'] == source
        info = siteinfo[locs]
        filepath = Path(info['.filepath'].iloc[0])
        igbp = filepath.parent.parent.name
        df = load_parquet(filepath=filepath)
        df['.source'] = source
        return df, igbp

    def find_filepaths(self) -> tuple[list, list, list]:
        """Collect paths to parquet files in list"""
        parquetfilepaths = []
        for filepattern in self.filepatterns:
            filepaths = search_files(
                searchdirs=self.basedir,
                pattern=filepattern)
            parquetfilepaths = parquetfilepaths + filepaths
        # Identify ICOS and FLUXNET files
        icosfilepaths = [f for f in parquetfilepaths if str(f.name).startswith('ICOSETC_')]
        fxnfilepaths = [f for f in parquetfilepaths if str(f.name).startswith('FLX_')]
        return parquetfilepaths, icosfilepaths, fxnfilepaths

    def make_sitelist(self) -> tuple[pd.DataFrame, list]:
        icos_sites_df = self.make_filepath_list(filepaths=self.icosfilepaths, source_info="ICOS", testing=False)
        fxn_sites_df = self.make_filepath_list(filepaths=self.fxnfilepaths, source_info="FLUXNET", testing=False)
        sites_df = pd.concat([icos_sites_df, fxn_sites_df], axis=0)
        filepaths = sites_df.reset_index(drop=True, inplace=False)
        filepaths.to_csv("OUT/11.1-basic_sitelist.csv", index=False)
        unique_sites = filepaths['SITE'].unique()
        return filepaths, unique_sites

    @staticmethod
    def make_filepath_list(filepaths: list, source_info: str, testing: bool = False):
        sites_df = pd.DataFrame()
        for ix, ip in enumerate(filepaths):
            if testing and ix > 1:
                break
            splits = ip.name.split('_')
            site = splits[1]
            info = {
                'SITE': [site],
                '.source': source_info,
                '.filepath': ip
            }
            site_df = pd.DataFrame.from_dict(info, orient='columns')
            sites_df = pd.concat([sites_df, site_df], axis=0, ignore_index=True)
        return sites_df


if __name__ == '__main__':
    hf = MergeFiles()
    hf.run()
