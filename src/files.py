from pathlib import Path

import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from diive.core.times.times import insert_timestamp
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


def readfile(filetype, filepath_icos, data_nrows):
    data = dv.readfiletype(filetype=filetype, filepath=filepath_icos, data_nrows=data_nrows)
    df, _ = data.get_filedata()
    return df


def _compare_years(primary_df, secondary_df, first_year_primary):
    """
    Reconciles two pandas DataFrames by adjusting them based on a common year.

    This function compares the number of valid records in a specified common year (`first_year_primary`)
    for two dataframes, `primary_df` and `secondary_df`. It truncates the dataframe with fewer
    records to remove that year's data. Afterward, it truncates the secondary dataframe
    to include only records that occurred before the start of the primary dataframe.

    Parameters:
        primary_df (pd.DataFrame): The primary DataFrame with a DatetimeIndex.
        secondary_df (pd.DataFrame): The secondary DataFrame with a DatetimeIndex.
        first_year_primary (int): The common year used for the initial comparison.

    Returns:
        tuple: A tuple containing the reconciled primary and secondary DataFrames.
               (primary_df, secondary_df)

    Raises:
        TypeError: If either DataFrame does not have a pandas DatetimeIndex.
        IndexError: If primary_df becomes empty, preventing subsequent operations.
    """
    # Ensure indices are datetime-like to allow year-based comparisons
    if not isinstance(primary_df.index, pd.DatetimeIndex) or not isinstance(secondary_df.index, pd.DatetimeIndex):
        raise TypeError("DataFrames must have a DatetimeIndex.")

    # Get the number of measured records for the specific year in each DataFrame
    n_measured_primary = (primary_df.loc[primary_df.index.year == first_year_primary, 'NEE_VUT_REF_QC'] == 0).sum()
    n_measured_secondary = (
            secondary_df.loc[secondary_df.index.year == first_year_primary, 'NEE_VUT_REF_QC'] == 0).sum()

    # Compare the number of records and truncate the appropriate DataFrame
    if n_measured_primary > n_measured_secondary:
        # If primary has more records, truncate secondary to remove the first year's data
        secondary_df = secondary_df[secondary_df.index.year > first_year_primary].copy()
    elif n_measured_secondary > n_measured_primary:
        # If secondary has more records, truncate primary to remove the first year's data
        primary_df = primary_df[primary_df.index.year > first_year_primary].copy()

    # If primary_df is empty after truncation, the next line will fail.
    # We must handle this edge case to prevent an IndexError.
    if primary_df.empty:
        return primary_df, secondary_df

    # Truncate secondary_df to only keep records that precede the start of primary_df
    # This logic assumes primary_df is chronologically later than secondary_df
    secondary_df = secondary_df[secondary_df.index < primary_df.index.min()].copy()

    return primary_df, secondary_df


def _resample_to_lower_freq(higher, lower, origin):
    higher = higher.drop('ORIGIN', axis=1).resample(lower.index.freq, closed='left', label='right').mean()
    higher.index.name = 'TIMESTAMP_END'
    higher = insert_timestamp(data=higher, convention='middle', insert_as_first_col=True, verbose=True)
    higher = higher.set_index('TIMESTAMP_MIDDLE', inplace=False, drop=True)
    higher['ORIGIN'] = origin
    return higher


def create_parquet_files(siteinfo_df, data_nrows, settings, siteconfig, ix, showplot=False) -> pd.DataFrame:
    site = siteconfig['SITE']

    # # # --- TODO testing
    # # if site != "US-UMB":
    # #     return pd.DataFrame()
    # if ix < 166:
    #     return pd.DataFrame()
    # # # --- TODO testing

    igbp = siteconfig['IGBP']
    origin = siteconfig['ORIGIN']
    filetype = "FLUXNET-FULLSET-HH-CSV-30MIN"
    filepath_icos = siteconfig['_FILEPATH_ICOS']
    filepath_fxn_cp = siteconfig['_FILEPATH_FXN_CP']
    filepath_fxn_org = siteconfig['_FILEPATH_FXN_ORG']
    filepath_amf = siteconfig['_FILEPATH_AMF']

    icos_df = None
    fxn_cp_df = None
    fxn_org_df = None
    amf_df = None
    merged_df = None
    icos_yrs = None
    fxn_cp_yrs = None
    fxn_org_yrs = None
    amf_yrs = None
    icos_firstyr = None
    fxn_cp_firstyr = None
    fxn_org_firstyr = None
    amf_firstyr = None
    sourcetxt = "-NO-SOURCE-ERROR-"

    # Read available files
    print(f"Reading data for #{ix + 1} {site}...")
    if isinstance(filepath_icos, str):
        icos_df = readfile(filetype, filepath_icos, data_nrows)
        icos_df['ORIGIN'] = 'ICOS'  # Add origin for each data record
        icos_yrs = list(set(icos_df.index.year))
        icos_firstyr = icos_df.index.year[0]

    if isinstance(filepath_fxn_cp, str):
        fxn_cp_df = readfile(filetype, filepath_fxn_cp, data_nrows)
        fxn_cp_df['ORIGIN'] = 'FLUXNET-CP'
        fxn_cp_yrs = list(set(fxn_cp_df.index.year))
        fxn_cp_firstyr = fxn_cp_df.index.year[0]

    if isinstance(filepath_fxn_org, str):
        # Sometime hourly data
        filetype = "FLUXNET-FULLSET-HR-CSV-60MIN" if '_FULLSET_HR_' in str(Path(
            filepath_fxn_org).name) else "FLUXNET-FULLSET-HH-CSV-30MIN"
        fxn_org_df = readfile(filetype, filepath_fxn_org, data_nrows)
        fxn_org_df['ORIGIN'] = 'FLUXNET-ORG'
        fxn_org_yrs = list(set(fxn_org_df.index.year))
        fxn_org_firstyr = fxn_org_df.index.year[0]

    if isinstance(filepath_amf, str):
        # Sometime hourly data
        filetype = "FLUXNET-FULLSET-HR-CSV-60MIN" if '_FULLSET_HR_' in str(Path(
            filepath_amf).name) else "FLUXNET-FULLSET-HH-CSV-30MIN"
        amf_df = readfile(filetype, filepath_amf, data_nrows)
        amf_df['ORIGIN'] = 'AMERIFLUX'
        amf_yrs = list(set(amf_df.index.year))
        amf_firstyr = amf_df.index.year[0]

    # ---------------------------------------------
    # ICOS + FXN-CP + FXN-ORG
    # Check for overlapping years
    # Find the first overlapping year between the two datasets.
    # The logic is that the first year of ICOS measurements for a site can
    # be incomplete, but from the second year onwards it should be fine.
    # In case more records are available for the FLUXNET dataset, the FLUXNET dataset is used
    # for this year. Otherwise ICOS.
    # Count number of directly measured NEE values to decide which dataset to use for this year.
    if (isinstance(icos_df, pd.DataFrame)
            and isinstance(fxn_cp_df, pd.DataFrame)
            and isinstance(fxn_org_df, pd.DataFrame)
            and not isinstance(amf_df, pd.DataFrame)):
        if len(set([icos_df.index.freqstr, fxn_cp_df.index.freqstr, fxn_org_df.index.freqstr])) > 1:
            raise Exception("Frequency mismatch between ICOS, FXN-CP and FXN-ORG not implemented yet")
        # Check if the first ICOS year appears in the FXN CP dataset
        # If yes, keep data from the dataset with more directly measured values for that year
        if icos_firstyr in fxn_cp_yrs:
            icos_df, fxn_cp_df = _compare_years(
                primary_df=icos_df,
                secondary_df=fxn_cp_df,
                first_year_primary=icos_firstyr)
        merged_df = pd.concat([icos_df, fxn_cp_df], axis=0)

        # Now check if the first merged year appears in the FXN ORG dataset
        if merged_df.index.year.min() in fxn_org_yrs:
            merged_df, fxn_org_df = _compare_years(
                primary_df=merged_df,
                secondary_df=fxn_org_df,
                first_year_primary=merged_df.index.year.min())
        merged_df = pd.concat([merged_df, fxn_org_df], axis=0)
        sourcetxt = "ICOS+FXN-CP+FXN-ORG"

    # ---------------------------------------------
    # ICOS + FXN-CP
    elif (isinstance(icos_df, pd.DataFrame)
          and isinstance(fxn_cp_df, pd.DataFrame)
          and not isinstance(fxn_org_df, pd.DataFrame)
          and not isinstance(amf_df, pd.DataFrame)):
        if len(set([icos_df.index.freqstr, fxn_cp_df.index.freqstr])) > 1:
            raise Exception("Frequency mismatch between ICOS and FXN-CP not implemented yet")
        if icos_firstyr in fxn_cp_yrs:
            icos_df, fxn_cp_df = _compare_years(
                primary_df=icos_df,
                secondary_df=fxn_cp_df,
                first_year_primary=icos_firstyr)
        merged_df = pd.concat([icos_df, fxn_cp_df], axis=0)
        sourcetxt = "ICOS+FXN-CP"

    # ---------------------------------------------
    # ICOS + FXN-ORG
    elif (isinstance(icos_df, pd.DataFrame)
          and not isinstance(fxn_cp_df, pd.DataFrame)
          and isinstance(fxn_org_df, pd.DataFrame)
          and not isinstance(amf_df, pd.DataFrame)):
        if len(set([icos_df.index.freqstr, fxn_org_df.index.freqstr])) > 1:
            raise Exception("Frequency mismatch between ICOS and FXN-ORG not implemented yet")
        if icos_firstyr in fxn_org_yrs:
            icos_df, fxn_org_df = _compare_years(
                primary_df=icos_df,
                secondary_df=fxn_org_df,
                first_year_primary=icos_firstyr)
        merged_df = pd.concat([icos_df, fxn_org_df], axis=0)
        sourcetxt = "ICOS+FXN-ORG"

    # ---------------------------------------------
    # FXN-CP + FXN-ORG
    elif (not isinstance(icos_df, pd.DataFrame)
          and isinstance(fxn_cp_df, pd.DataFrame)
          and isinstance(fxn_org_df, pd.DataFrame)
          and not isinstance(amf_df, pd.DataFrame)):
        if len(set([fxn_cp_df.index.freqstr, fxn_org_df.index.freqstr])) > 1:
            raise Exception("Frequency mismatch between FXN-CP and FXN-ORG not implemented yet")
        if fxn_cp_firstyr in fxn_org_yrs:
            fxn_cp_df, fxn_org_df = _compare_years(
                primary_df=fxn_cp_df,
                secondary_df=fxn_org_df,
                first_year_primary=fxn_cp_firstyr)
        merged_df = pd.concat([fxn_cp_df, fxn_org_df], axis=0)
        sourcetxt = "FXN-CP+FXN-ORG"

    # ---------------------------------------------
    # AMF + FXN-ORG
    elif (not isinstance(icos_df, pd.DataFrame)
          and not isinstance(fxn_cp_df, pd.DataFrame)
          and isinstance(fxn_org_df, pd.DataFrame)
          and isinstance(amf_df, pd.DataFrame)):

        # Dataframes need to have the same time resolution
        # It is possible that one of the dataframes is in hourly time resolution,
        # and the other in half-hourly time resolution. In such a case, resample
        # the dataframe with the higher resolution to the frequency of the dataframe
        # with the lower resolution. Typically, this means that the half-hourly
        # dataframe is resampled to hourly.
        if amf_df.index.freq != fxn_org_df.index.freq:
            if fxn_org_df.index.freq > amf_df.index.freq:
                amf_df = _resample_to_lower_freq(higher=amf_df, lower=fxn_org_df, origin='AMERIFLUX')
            elif fxn_org_df.index.freq < amf_df.index.freq:
                fxn_org_df = _resample_to_lower_freq(higher=fxn_org_df, lower=amf_df, origin='FLUXNET-ORG')

        if amf_firstyr in fxn_org_yrs:
            amf_df, fxn_org_df = _compare_years(
                primary_df=amf_df,
                secondary_df=fxn_org_df,
                first_year_primary=amf_firstyr)
        merged_df = pd.concat([amf_df, fxn_org_df], axis=0)
        sourcetxt = "AMF+FXN-ORG"


    # ---------------------------------------------
    # ICOS only
    elif (isinstance(icos_df, pd.DataFrame)
          and not isinstance(fxn_cp_df, pd.DataFrame)
          and not isinstance(fxn_org_df, pd.DataFrame)
          and not isinstance(amf_df, pd.DataFrame)):
        merged_df = icos_df.copy()
        sourcetxt = "ICOS"

    # ---------------------------------------------
    # FXN-CP only
    elif (not isinstance(icos_df, pd.DataFrame)
          and isinstance(fxn_cp_df, pd.DataFrame)
          and not isinstance(fxn_org_df, pd.DataFrame)
          and not isinstance(amf_df, pd.DataFrame)):
        merged_df = fxn_cp_df.copy()
        sourcetxt = "FXN-CP"

    # ---------------------------------------------
    # FXN-ORG only
    elif (not isinstance(icos_df, pd.DataFrame)
          and not isinstance(fxn_cp_df, pd.DataFrame)
          and isinstance(fxn_org_df, pd.DataFrame)
          and not isinstance(amf_df, pd.DataFrame)):
        merged_df = fxn_org_df.copy()
        sourcetxt = "FXN-ORG"

    # ---------------------------------------------
    # AMF only
    elif (not isinstance(icos_df, pd.DataFrame)
          and not isinstance(fxn_cp_df, pd.DataFrame)
          and not isinstance(fxn_org_df, pd.DataFrame)
          and isinstance(amf_df, pd.DataFrame)):
        merged_df = amf_df.copy()
        sourcetxt = "AMF"

    # Save merged data to parquet file
    merged_df = merged_df.sort_index()
    start = merged_df.index[0].year
    end = merged_df.index[-1].year
    outname = f"{site}_{igbp}_{sourcetxt}_{start}-{end}"
    outfilepath = dv.save_parquet(filename=outname,
                                  data=merged_df,
                                  outpath=Path(settings['DIR_DATA_PROC_PARQUET']))

    _save_mergeplot(outname, icos_df, fxn_cp_df, fxn_org_df, amf_df, merged_df,
                    site, igbp, sourcetxt, settings, showplot)
    siteinfo_df.loc[ix, '_FILEPATH_PARQUET'] = Path(outfilepath)
    return siteinfo_df


def _save_mergeplot(outname, icos_df, fxn_cp_df, fxn_org_df, amf_df, merged_df,
                    site, igbp, sourcetxt, settings, showplot):
    # Heatmap plots
    outfile = Path(settings['DIR_DATA_PROC_PARQUET_PLOTS']) / outname
    print(f"Saving heatmap plot to {outfile} ...")
    var = 'NEE_VUT_REF'
    fig = plt.figure(facecolor='white', figsize=(21, 9), dpi=72)
    gs = gridspec.GridSpec(1, 5)  # rows, cols
    gs.update(wspace=0.5, hspace=0.3, left=0.05, right=0.95, top=0.95, bottom=0.07)
    ax_icos = fig.add_subplot(gs[0, 0])
    ax_fxn_cp = fig.add_subplot(gs[0, 1], sharey=ax_icos)
    ax_fxn_org = fig.add_subplot(gs[0, 2], sharey=ax_icos)
    ax_amf = fig.add_subplot(gs[0, 3], sharey=ax_icos)
    ax_merged = fig.add_subplot(gs[0, 4], sharey=ax_icos)

    # Heatmaps
    plotkwargs = dict(zlabel=f"{var}", cb_digits_after_comma=0, vmin=-20, vmax=20)
    if isinstance(icos_df, pd.DataFrame):
        if not icos_df.empty:
            dv.heatmapdatetime(ax=ax_icos, series=icos_df[var], **plotkwargs).plot()
    if isinstance(fxn_cp_df, pd.DataFrame):
        if not fxn_cp_df.empty:
            dv.heatmapdatetime(ax=ax_fxn_cp, series=fxn_cp_df[var], **plotkwargs).plot()
    if isinstance(fxn_org_df, pd.DataFrame):
        if not fxn_org_df.empty:
            dv.heatmapdatetime(ax=ax_fxn_org, series=fxn_org_df[var], **plotkwargs).plot()
    if isinstance(amf_df, pd.DataFrame):
        if not amf_df.empty:
            dv.heatmapdatetime(ax=ax_amf, series=amf_df[var], **plotkwargs).plot()
    if isinstance(merged_df, pd.DataFrame):
        if not merged_df.empty:
            dv.heatmapdatetime(ax=ax_merged, series=merged_df[var], **plotkwargs).plot()

    # Titles
    ax_icos.set_title(f"{site} ICOS\nused data", color='black')
    ax_fxn_cp.set_title(f"{site} FLUXNET-CP\nused data", color='black')
    ax_fxn_org.set_title(f"{site} FLUXNET-ORG\nused data", color='black')
    ax_amf.set_title(f"{site} AMERIFLUX\nused data", color='black')
    ax_merged.set_title(f"{site} {sourcetxt}\nmerged data", color='black')

    tickkwargs = dict(labeltop=False, labelbottom=True, top=False, bottom=True,
                      left=True, right=False)
    ax_icos.tick_params(labelleft=True, labelright=False, **tickkwargs)
    ax_fxn_cp.tick_params(labelleft=True, labelright=False, **tickkwargs)
    ax_fxn_org.tick_params(labelleft=True, labelright=False, **tickkwargs)
    ax_amf.tick_params(labelleft=True, labelright=False, **tickkwargs)
    ax_merged.tick_params(labelleft=True, labelright=False, **tickkwargs)

    fig.savefig(outfile, dpi=72)
    if showplot:
        fig.show()


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
