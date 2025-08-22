from pathlib import Path

import diive as dv
import numpy as np
import pandas as pd
import yaml
from scipy.stats import zscore

from src.common import get_variable_names


def prepare_input_data(settings, siteinfo_df, siteconfig, ix):
    print(f"\nLoading data for site #{ix + 1} {siteconfig['SITE']} ...")

    # Load site data
    filepath = siteconfig['_FILEPATH_PARQUET']
    sitedata = dv.load_parquet(filepath)
    # [print(c) for c in sitedata.columns if "LE" in c];

    # Get variable names for this site
    varnames = get_variable_names(siteconfig)

    if varnames['swc_var'] == '-MISSING-':
        siteinfo_df.loc[ix, '_FILEPATH_PARQUET_SUBSET'] = '-MISSING-'
        return siteinfo_df

    # Make subset
    subset = sitedata[
        [
            varnames['nee_var'], varnames['nee_qc_var'],
            varnames['le_var'], varnames['le_qc_var'],
            varnames['gpp_var'],
            varnames['reco_var'],
            varnames['swinpot_var'], varnames['swin_var'],
            varnames['ta_var'], varnames['vpd_var'],
            varnames['swc_var']
        ]
    ].copy()

    # Keep 6 warmest months
    ta = sitedata[[varnames['ta_var']]].copy()
    ta['MONTH'] = ta.index.month
    monthly_avg = ta.groupby('MONTH').mean()
    monthly_avg = monthly_avg.sort_values(by=varnames['ta_var'], ascending=False, inplace=False)
    warmest6 = monthly_avg.head(6).index.to_list()
    subset = subset.loc[sitedata.index.month.isin(warmest6)].copy()

    # Keep directly measured fluxes, no gap-filled data
    if varnames['nee_qc_var'] is not None:
        subset = subset.loc[subset[varnames['nee_qc_var']] == 0].copy()

    # Keep daytime records
    subset = subset.loc[subset[varnames['swinpot_var']] > 20].copy()

    # Keep required cols
    subset = subset[
        [
            varnames['nee_var'],
            varnames['le_var'], varnames['gpp_var'], varnames['reco_var'],
            varnames['ta_var'],
            varnames['vpd_var'], varnames['swin_var'],
            varnames['swc_var']
        ]
    ].copy()

    # Keep records where all vars available
    subset = subset.dropna()

    # Convert z-scores, ignoring NaNs
    # z-scores are calculated from subset records
    subset = subset.apply(lambda x: zscore(x, nan_policy='omit'))

    # Rename variables to have the same var names for all sites
    rename_dict = {
        varnames['nee_var']: "NEE",
        varnames['le_var']: "LE",
        varnames['gpp_var']: "GPP",
        varnames['reco_var']: "RECO",
        varnames['ta_var']: "TA",
        varnames['vpd_var']: "VPD",
        varnames['swin_var']: "SWIN",
        varnames['swc_var']: "SWC",
    }
    subset = subset.rename(columns=rename_dict, inplace=False)

    # Convert NEE to NEP
    subset['NEP'] = subset['NEE'].multiply(-1)

    print(f"Records: {len(subset)}")

    # TODO testing: Limit time range
    # subset = subset.loc[subset.index.year == 2019].copy()
    # subset = subset.loc[subset.index.month == 7].copy()
    # TODO testing: Limit time range

    outfilepath = dv.save_parquet(
        filename=f"{siteconfig['SITE']}_subset_warmest6_qc0_daytime_zscores",
        data=subset,
        outpath=Path(settings['DIR_DATA_PROC_SUBSETS']))
    print(f"Saved subset data for {siteconfig['SITE']} to file {outfilepath}.")
    siteinfo_df.loc[ix, '_FILEPATH_PARQUET_SUBSET'] = outfilepath

    return siteinfo_df


def save_siteinfo(siteinfo_df: pd.DataFrame, settings: dict) -> None:
    outfile = Path(Path(settings['OUTFILE_SITEINFO']))
    siteinfo_df.to_csv(outfile, index=False)
    print(f"Saved updated site info to file {outfile}.")
    return None


def load_siteinfo(settings: dict) -> pd.DataFrame:
    infile = Path(Path(settings['OUTFILE_SITEINFO']))
    siteinfo_df = pd.read_csv(infile)
    siteinfo_df = siteinfo_df.fillna(np.nan)
    return siteinfo_df


def create_parquet_files(siteinfo_df, data_nrows, settings, siteconfig, ix) -> pd.DataFrame:
    site = siteconfig['SITE']
    igbp = siteconfig['IGBP']
    origin = siteconfig['ORIGIN']
    filepath_icos = siteconfig['_FILEPATH_ICOS']
    filepath_fxn = siteconfig['_FILEPATH_FXN']
    filepath_amf = siteconfig['_FILEPATH_AMF']

    # Load data
    if origin == 'ICOS+FLUXNET':
        load_icos = dv.readfiletype(filetype="FLUXNET-FULLSET-HH-CSV-30MIN", filepath=filepath_icos,
                                    data_nrows=data_nrows)
        icosdf, _ = load_icos.get_filedata()
        load_fxn = dv.readfiletype(filetype="FLUXNET-FULLSET-HH-CSV-30MIN", filepath=filepath_fxn,
                                   data_nrows=data_nrows)
        fxndf, _ = load_fxn.get_filedata()

        # Check for overlapping years
        # Find the first overlapping year between the two datasets.
        # The logic is that the first year of ICOS measurements for a site can
        # be incomplete, but from the second year onwards it should be fine.
        # In case more records are available for the FLUXNET dataset, the FLUXNET dataset is used
        # for this year. Otherwise ICOS.
        # Count number of directly measured NEE values to decide which dataset to use for this year.
        yrs_icos = list(set(icosdf.index.year))
        yrs_fxn = list(set(fxndf.index.year))
        firstyr_icos = yrs_icos[0]

        if firstyr_icos in yrs_fxn:
            qcseries_icos = icosdf.loc[icosdf.index.year == firstyr_icos, 'NEE_VUT_REF_QC']
            n_measured_icos = qcseries_icos[qcseries_icos == 0].count()
            qcseries_fxn = fxndf.loc[fxndf.index.year == firstyr_icos, 'NEE_VUT_REF_QC']
            n_measured_fxn = qcseries_fxn[qcseries_fxn == 0].count()

            # In case FXN has more records for the first common year, remove year from ICOS
            if n_measured_fxn > n_measured_icos:
                keeplocs_icos = icosdf.index.year > firstyr_icos
                icosdf = icosdf.loc[keeplocs_icos]
            # In case ICOS has more records, remove year from FXN
            elif n_measured_icos > n_measured_fxn:
                keeplocs_fxn = fxndf.index.year < firstyr_icos
                fxndf = fxndf.loc[keeplocs_fxn]

        # Generally, only keep records from FLUXNET data that are not in ICOS data
        keeplocs_fxn = fxndf.index < icosdf.index[0]
        fxndf = fxndf[keeplocs_fxn].copy()

        # Merge ICOS and FLUXNET data
        merged_df = pd.concat([icosdf, fxndf], axis=0)
        merged_df = merged_df.sort_index()
        sourcetxt = "ICOS+FXN"

        # # Heatmap plots
        # var = 'NEE_VUT_REF'
        # hm = dv.heatmapdatetime(series=icosdf[var], title=f"{site} ICOS (2025)", vmin=-20, vmax=20)
        # hm.show()
        # hm = dv.heatmapdatetime(series=fxndf[var], title=f"{site} FLUXNET (2024)", vmin=-20, vmax=20)
        # hm.show()
        # hm = dv.heatmapdatetime(series=merged_df[var], title=f"{site} ICOS+FLUXNET", vmin=-20, vmax=20)
        # hm.show()

    elif origin == 'FLUXNET_CP':
        icosdf = None
        load_fxn = dv.readfiletype(filetype="FLUXNET-FULLSET-HH-CSV-30MIN", filepath=filepath_fxn,
                                   data_nrows=data_nrows)
        merged_df, _ = load_fxn.get_filedata()
        sourcetxt = "FXN"

    elif origin == 'ICOS':
        fxndf = None
        load_icos = dv.readfiletype(filetype="FLUXNET-FULLSET-HH-CSV-30MIN", filepath=filepath_icos,
                                    data_nrows=data_nrows)
        merged_df, _ = load_icos.get_filedata()
        sourcetxt = "ICOS"

    elif origin == 'AMERIFLUX':
        fxndf = None
        filename = Path(filepath_amf).name
        # Some files are at 60MIN time resolution
        if "_FLUXNET_FULLSET_HH_" in filename:
            filetype = "FLUXNET-FULLSET-HH-CSV-30MIN"
        elif "_FLUXNET_FULLSET_HR_" in filename:
            filetype = "FLUXNET-FULLSET-HR-CSV-60MIN"
        else:
            raise NotImplementedError
        load_amf = dv.readfiletype(filetype=filetype, filepath=filepath_amf, data_nrows=data_nrows)
        merged_df, _ = load_amf.get_filedata()
        sourcetxt = "AMERIFLUX"

    else:
        raise Exception("Unknown origin.")

    # Save merged data to parquet file
    start = merged_df.index[0].year
    end = merged_df.index[-1].year
    outfilepath = dv.save_parquet(filename=f"{site}_{igbp}_{sourcetxt}_{start}-{end}",
                                  data=merged_df,
                                  outpath=Path(settings['DIR_DATA_PROC_PARQUET']))

    siteinfo_df.loc[ix, '_FILEPATH_PARQUET'] = Path(outfilepath)

    return siteinfo_df


# def convert_datafiles_to_parquet(filepatterns: list, filetype: str, searchdir: str, outpath: str):
#     filelist = []
#     for filepattern in filepatterns:
#         _filelist = search_files(
#             searchdirs=searchdir,
#             pattern=filepattern)
#         filelist = filelist + _filelist
#
#     for f in filelist:
#         filename = f.name
#         loaddatafile = ReadFileType(filetype=filetype, filepath=f, data_nrows=None)
#         data_df, metadata_df = loaddatafile.get_filedata()
#         filepath = save_parquet(filename=filename, data=data_df, outpath=outpath)


def read_settings_file(filepath_settings) -> dict:
    """Read start values from settings file as strings into dict, with same variable names as in file"""
    with open(filepath_settings, 'r', encoding='utf-8') as f:
        settings_dict = yaml.safe_load(f)
    return settings_dict

# def search_files(searchdirs: str or list, pattern: str) -> list:
#     """ Search files and store their filename and the path to the file in dictionary. """
#     # found_files_dict = {}
#     foundfiles = []
#     if isinstance(searchdirs, str):
#         searchdirs = [searchdirs]  # Use str as list
#     for searchdir in searchdirs:
#         for root, dirs, files in os.walk(searchdir):
#             for idx, settings_file_name in enumerate(files):
#                 if fnmatch.fnmatch(settings_file_name, pattern):
#                     filepath = Path(root) / settings_file_name
#                     # found_files_dict[settings_file_name] = filepath
#                     foundfiles.append(filepath)
#     foundfiles.sort()
#     return foundfiles
