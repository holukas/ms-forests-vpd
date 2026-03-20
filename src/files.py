import logging
from pathlib import Path

import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from diive.core.times.times import insert_timestamp
from scipy.stats import zscore


def load_data(suffix, shap_type, dir_res, flux, aggfunc, subsetcols: list,
              x_in_filename: str, y_in_filename: str, count_vals_col: tuple[str, str] = False, site_filter=None):
    """
    Loads parquet, flattens cols, optionally filters by index.
    suffix: 'Sites' (for main plot, prefix 42) or 'IGBP-X' (for subplots, prefix 43)
    """
    # Select 42 for 'Sites' (AggregatedAcrossSites) and 43 for IGBP (AggregatedAcrossIGBP)
    prefix = "42" if suffix == 'Sites' else "43"
    filename = (f"{prefix}_SHAPVALUES-{shap_type}_{aggfunc}AggregatedAcross{suffix}"
                f"_{x_in_filename}+{y_in_filename}+{flux}.parquet")
    fp = dir_res / filename
    filedf = dv.load_parquet(fp, sanitize_timestamp=False, output_middle_timestamp=False)

    # Filter to match the index of the main plot (for IGBP subplots)
    if site_filter is not None:
        filedf = filedf[filedf.index.isin(site_filter)].copy()

    # Get total number of sites used for aggregations
    n_sites = list(set(filedf['N_SITES'].tolist()))
    if len(n_sites) > 1:
        raise ValueError(f"Multiple values for N_SITES: {n_sites}")
    n_sites = int(n_sites[0])

    # Apply count threshold, at least half of total sites needed
    # threshold = np.ceil(filedf[count_vals_col].max() / 2)
    threshold = np.ceil(n_sites / 2)
    # threshold = n_sites_min if suffix == 'Sites' else 9
    mask = filedf[count_vals_col] >= threshold
    filedf = filedf[mask].copy()

    # Count number of sites (min, max) per bin class
    # Note that zero counts are not relevant for the plots b/c
    # they are not shown in the plots, i.e., for the minimum
    # we need to get the next lowest number.
    _counts = filedf[count_vals_col]
    min_count = _counts[_counts > 0].min()
    max_count = _counts[_counts > 0].max()
    minmax_counts = [min_count, max_count]

    # Prepare subset for plotting
    subsetdf = filedf[subsetcols].copy()
    subsetdf.columns = ['_'.join(c).strip() for c in subsetdf.columns]

    return filedf, subsetdf, minmax_counts, n_sites


def create_subsets_parquet_files(settings: dict, filepath_parquet_fullset: str, ix: int, varnames, site: str,
                                 igbp: str, origin: str, logging=logging) -> dict:
    """
    Processes the full flux data for a specific site to create a quality-controlled,
    seasonally-filtered, and year-balanced subset for subsequent analysis.

    The function performs a sequence of steps: loads data, selects required variables,
    filters for QC-flag 0 (measured data) and daytime records, identifies the 4 warmest months,
    balances the data across available years for these months, calculates derived
    variables (ET, NEP), converts all measured variables to Z-scores, saves the
    final subset to a Parquet file, generates a heatmap visualization, and returns
    comprehensive summary statistics.

    The core filtering step ensures that all 4 warmest months included in the subset
    have the *exact same number of available years* of data, using the latest years
    available to achieve this balance, which prevents monthly bias in long-term statistics.

    Args:
        settings (dict): A dictionary containing global settings, including output directory paths
                         (e.g., 'DIR_DATA_PROC_SUBSETS', 'DIR_DATA_PROC_SUBSETS_PLOTS').
        filepath_parquet_fullset (str): The file path to the complete, raw site dataset (Parquet format).
        ix (int): The index of the current site being processed (used for console logging).
        varnames (dict): A mapping dictionary where keys are generic variable types (e.g., 'nee_var',
                         'ta_var') and values are the specific column names in the input dataset.
                         (Variables must be present: NEE, LE, SWIN_POT, TA, VPD, SWC, and corresponding QC flags).
        site (str): The unique identifier for the flux tower site (e.g., 'AU-Cum'). This is used
                    in naming the output file.
        igbp (str): The IGBP classification code for the site (e.g., 'ENF'). Used for plot title/metadata.
        origin (str): The source of the data (e.g., 'FLUXNET', 'OZFLUX'). Used for plot title/metadata.

    Returns:
        dict: A dictionary containing comprehensive metadata and summary statistics for the
              generated subset, including date ranges, record counts, and min/max/mean/SD
              for both measured and Z-score variables. The dictionary also includes the
              file path to the generated Parquet subset.

    Raises:
        KeyError: If a required variable name from `varnames` is not found in the dataset
                  when loading or selecting columns.
    """
    print(f"\nLoading data for site #{ix + 1} {site} ...")

    # Load full dataset for this site
    sitedata = dv.load_parquet(filepath_parquet_fullset)

    # Make subset
    subset = sitedata[
        [
            varnames['nee_var'], varnames['nee_qc_var'],
            varnames['le_var'], varnames['le_qc_var'],
            varnames['gpp_var'],
            varnames['reco_var'],
            varnames['swinpot_var'], varnames['swin_var'],
            varnames['ta_var'], varnames['vpd_var'],
            varnames['swc_var'],
            # varnames['rh_var']
        ]
    ].copy()

    # Preparation
    subset['MONTH'] = subset.index.month
    subset['YEAR'] = subset.index.year

    # todo now testing with GPP
    # First, identify 4 months with highest GPP from full dataset
    gpp_avg = subset.groupby('MONTH')[varnames['gpp_var']].mean()
    gpp_top4 = gpp_avg.nlargest(4)
    gpp_top4 = gpp_top4.index.to_list()
    # # First, identify 4 warmest months from full dataset
    # ta_avg = subset.groupby('MONTH')[varnames['ta_var']].mean()
    # warmest4 = ta_avg.nlargest(4)
    # warmest4 = warmest4.index.to_list()

    # Now start to narrow down data

    # Keep directly measured NEE fluxes, no gap-filled flux data
    if varnames['nee_qc_var'] is not None:
        subset = subset.loc[subset[varnames['nee_qc_var']] == 0].copy()
    else:
        raise KeyError(f"Required variable '{varnames['nee_qc_var']}' not found in dataset.")

    # # Keep directly measured LE fluxes, no gap-filled flux data
    # # Does currently not work because of a bug in FLUXNET data where the QC flag
    # # indicates that ALL LE values are gapfilled.
    # if varnames['le_qc_var'] is not None:
    #     subset = subset.loc[subset[varnames['le_qc_var']] == 0].copy()
    # else:
    #     raise KeyError(f"Required variable '{varnames['le_qc_var']}' not found in dataset.")

    # Keep daytime records
    subset = subset.loc[subset[varnames['swinpot_var']] > 20].copy()

    # Keep 4 warmest months
    # Keep as much data as is needed to have the same number of available years
    # for each of the 4 warmest months, to avoid monthly bias.

    # Filter to only the 4 warmest months (after QC/daytime filters)
    gpp_top4_df = subset.loc[subset['MONTH'].isin(gpp_top4)].copy()

    # Count the number of unique years available for each of the 4 months
    # Group by month and count the number of unique years in each group
    month_year_counts = gpp_top4_df.groupby('MONTH')['YEAR'].nunique()

    # Find the minimum number of available years across all 4 months
    min_years = month_year_counts.min()

    # Identify the set of years needed to be kept for each month to achieve the balance.
    # Requires iteration over the months
    balanced_indices = []
    for month in gpp_top4:
        # Get all records for this specific month
        month_data = gpp_top4_df.loc[gpp_top4_df['MONTH'] == month].copy()

        # Find the years available for this month
        available_years = month_data['YEAR'].unique()

        # If a month has too many years, trim the oldest ones
        if len(available_years) > min_years:
            # Sort years and keep only the latest min_years, for example
            years_to_keep = sorted(available_years, reverse=True)[:min_years]
        else:
            years_to_keep = available_years

        # Filter the data for this month to keep only the selected years
        balanced_month_data = month_data.loc[month_data['YEAR'].isin(years_to_keep)]

        # Collect the indices of the balanced data
        balanced_indices.append(balanced_month_data.index)

    # Final subset: use the collected balanced indices
    all_balanced_indices = pd.DatetimeIndex([], name=subset.index.name)
    for index_list in balanced_indices:
        all_balanced_indices = all_balanced_indices.union(index_list)

    # Apply the final index filter
    subset = subset.loc[all_balanced_indices].copy()

    # Count if there are data from all 4 months
    if len(subset['MONTH'].unique()) != 4:
        logging.warning(f"Not all 4 high-GPP months are available for site {site}.")
        # logging.warning(f"Not all 4 warmest months are available for site {site}.")

    # Cleanup temporary columns
    subset = subset.drop(columns=['MONTH', 'YEAR'], errors='ignore')

    # Keep required cols
    subset = subset[
        [
            varnames['nee_var'],
            varnames['le_var'], varnames['gpp_var'], varnames['reco_var'],
            varnames['ta_var'],
            varnames['vpd_var'], varnames['swin_var'],
            varnames['swc_var'],
            # varnames['rh_var']
        ]
    ].copy()

    # Keep records where all vars available
    # It is possible that we lose the complete dataset here,
    # e.g. when SWC is available for some months but not for the warmest 4.
    subset_stats = subset.describe()
    subset = subset.dropna()
    if subset.empty:
        # Count if there are data from all 4 months
        logging.warning(f"{site} Skipped site because subset dataframe is empty after dropna(). "
                        f"Most likely one of the variables has no data in the selected time period. "
                        f"Variable counts: {subset_stats.loc['count']}")
        return dict()

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
        # varnames['rh_var']: "RH",
    }
    subset = subset.rename(columns=rename_dict, inplace=False)

    # Calculate ET from LE and TA
    subset['ET'] = dv.et_from_le(le=subset['LE'], ta=subset['TA'])

    # Convert NEE to NEP
    subset['NEP'] = subset['NEE'].multiply(-1)

    # Store originally measured values before z-score conversion
    subset_meas = subset.copy()

    # Convert z-scores, ignoring NaNs
    # z-scores are calculated from subset records
    subset = subset.apply(lambda x: zscore(x, nan_policy='omit'))
    subset = subset.add_suffix("_ZSCORE")

    # Merge measured and z-score data to one single dataframe
    subset = pd.concat([subset, subset_meas], axis=1)

    print(f"Records: {len(subset)}")

    # Save subset data with z-scores to parquet file
    outfilepath = dv.save_parquet(
        filename=f"{site}_subset_GPPhighest4_qc0_daytime",
        data=subset,
        outpath=Path(settings['DIR_DATA_PROC_SUBSETS']))
    print(f"Saved subset data (measured and z-scores) for {site} to file {outfilepath}.")

    # Save heatmap plot
    start = subset.index[0].year
    end = subset.index[-1].year
    outname = f"{site}_{igbp}_{origin}_SUBSET_{start}-{end}"
    save_subset_heatmap_plot(df=subset, outname=outname, site=site, igbp=igbp, sourcetxt=origin,
                             showplot=True, fluxvars=['SWIN', 'TA', 'VPD', 'SWC', 'NEP', 'ET', 'GPP', 'RECO'],
                             outpath=settings['DIR_DATA_PROC_SUBSETS_PLOTS'])

    # Calculate stats for subset
    date_first = subset.index[0]
    date_last = subset.index[-1]
    date_first_str = str(date_first.strftime('%d %b %Y'))
    date_last_str = str(date_last.strftime('%d %b %Y'))
    n_records = len(subset.index)
    n_years = (date_last.year - date_first.year) + 1

    # Collect info about subset
    subsetinfo = dict()
    subsetinfo['SITE'] = site
    subsetinfo['DATE_FIRST'] = date_first_str
    subsetinfo['DATE_LAST'] = date_last_str
    subsetinfo['N_YEARS'] = n_years
    subsetinfo['N_RECORDS'] = n_records

    # Calculate z-statistics for overview
    for var in subset.columns:

        subsetinfo[f'{var}_MIN'] = subset[var].min()
        subsetinfo[f'{var}_MAX'] = subset[var].max()

        if str(var).endswith("_ZSCORE"):
            measuredname = str(var).replace("_ZSCORE", "")
            mean = subset[measuredname].mean()
            sd = subset[measuredname].std()
            subsetinfo[f'{measuredname}_Z0'] = mean
            subsetinfo[f'{measuredname}_SD'] = sd
            subsetinfo[f'{measuredname}_Z+2'] = mean + (2 * sd)
            subsetinfo[f'{measuredname}_Z-2'] = mean - (2 * sd)
        else:
            continue

    # Store filepath to subset parquet file
    subsetinfo['_FILEPATH_PARQUET_SUBSET'] = outfilepath

    return subsetinfo


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


import pandas as pd


def _compare_years(primary_df, secondary_df, fluxvar):
    """
    Reconciles two pandas DataFrames by adjusting them based on a common year.
    """
    # 1. Check types FIRST before trying to access .year
    if not isinstance(primary_df.index, pd.DatetimeIndex) or not isinstance(secondary_df.index, pd.DatetimeIndex):
        raise TypeError("DataFrames must have a DatetimeIndex.")

    first_year_primary = primary_df.index.year.min()

    # Get the number of measured records for the specific year in each DataFrame
    n_measured_primary = (primary_df.loc[primary_df.index.year == first_year_primary, f'{fluxvar}_QC'] == 0).sum()
    n_measured_secondary = (
            secondary_df.loc[secondary_df.index.year == first_year_primary, f'{fluxvar}_QC'] == 0).sum()

    # Compare the number of records and truncate the appropriate DataFrame
    if n_measured_primary > n_measured_secondary:
        secondary_df = secondary_df[secondary_df.index.year != first_year_primary].copy()
    elif n_measured_secondary > n_measured_primary:
        primary_df = primary_df[primary_df.index.year != first_year_primary].copy()

    # 2. Safety check: raise early if primary_df was completely emptied by the truncation
    if primary_df.empty or secondary_df.empty:
        raise ValueError("One or both DataFrames were completely emptied by the truncation.")

    # 3. Determine which df is truly Primary (the one with the latest timestamp)
    if secondary_df.index.max() > primary_df.index.max():
        primary_df, secondary_df = secondary_df, primary_df

    # 4. Clip Secondary so it strictly ends before Primary starts
    secondary_df = secondary_df[secondary_df.index < primary_df.index.min()].copy()

    if secondary_df.empty:
        print("Secondary data was entirely superseded by Primary or is out of range.")

    return primary_df, secondary_df


def resample_to_lower_freq(higher, lower):
    resample_to = lower.index.freq

    # Resample all columns EXCEPT 'ORIGIN' (string) and apply the mean
    higher_numeric_cols = higher.drop(columns=['ORIGIN']).resample(resample_to, closed='left', label='right').mean()
    higher_numeric_cols.index.name = 'TIMESTAMP_END'

    # Resample ONLY the 'ORIGIN' column and apply the last value
    higher_origin_col = higher['ORIGIN'].resample(resample_to, closed='left',
                                                  label='right').last().to_frame()  # .to_frame() converts Series back to DataFrame
    higher_origin_col.index.name = 'TIMESTAMP_END'

    # Combine results and add middle timestamp
    higher_resampled = higher_numeric_cols.join(higher_origin_col)
    higher_resampled = insert_timestamp(data=higher_resampled, convention='middle',
                                        insert_as_first_col=True, verbose=True)
    higher_resampled = higher_resampled.set_index('TIMESTAMP_MIDDLE', inplace=False, drop=True)
    return higher_resampled


def create_parquet_files(datasets_df, data_nrows, settings, ix, sites_done,
                         site, showplot=False):
    if site in sites_done:
        return datasets_df, sites_done

    # todo testing
    if site != "US-xSB":
        return datasets_df, sites_done
    # todo testing

    # # todo testing
    # if ix + 1 < 298:
    #     return datasets_df, sites_done
    # # todo testing

    igbp = None
    merged_df = None
    sourcetxt = ""
    datasetinfo_updated = None  # Consolidates site info across multiple datasets (e.g. IGBP can be different)
    fluxvar = None  # Detected later

    # Found datasets for this site
    subset = datasets_df.loc[datasets_df['SITE'] == site].copy()

    print(f"\nMerging datasets for #{ix + 1} {site} ({len(subset)} datasets)...")

    subset = subset.sort_values(by='PRIORITY', ascending=True, inplace=False)
    subset = subset.reset_index(drop=True)
    for ix, datasetinfo in subset.iterrows():

        # Handle the highest priority dataset first
        # This section also handles sites with only one dataset
        if ix == 0:
            filetype = "FLUXNET-FULLSET-HR-CSV-60MIN" if '_FULLSET_HR_' in str(
                Path(datasetinfo['_FILEPATH']).name) else "FLUXNET-FULLSET-HH-CSV-30MIN"
            merged_df = readfile(filetype, datasetinfo['_FILEPATH'], data_nrows)
            merged_df['ORIGIN'] = datasetinfo['ORIGIN']
            sourcetxt += datasetinfo['ORIGIN']
            igbp = datasetinfo['IGBP']  # Use IGBP from highest priority
            datasetinfo_updated = datasetinfo.copy()  # Use info from highest priority

            # Set flux variable
            # This assumes that incoming_df also has this variable available
            allowed_fluxvars = ['NEE_VUT_50', 'NEE_vUT_USTAR50', 'NEE_CUT_50']
            fluxvar = None
            for af in allowed_fluxvars:
                if af in merged_df.columns:
                    fluxvar = af
                    break
            if fluxvar is None:
                raise ValueError(f"None of the allowed flux variables "
                                 f"found in primary dataset {datasetinfo['_FILEPATH']}.")
            # if fluxvar not in merged_df.columns:
            #     fluxvar = 'NEE_vUT_USTAR50'  # Found for JapanFlux2024 sites

        # Handle lower priority datasets
        else:
            filetype = "FLUXNET-FULLSET-HR-CSV-60MIN" if '_FULLSET_HR_' in str(
                Path(datasetinfo['_FILEPATH']).name) else "FLUXNET-FULLSET-HH-CSV-30MIN"
            incoming_df = readfile(filetype, datasetinfo['_FILEPATH'], data_nrows)

            # Additional datasets must have the same flux variable as the primary dataset
            if fluxvar not in incoming_df.columns:
                raise ValueError(f"Flux variable {fluxvar} from primary dataset not found in "
                                 f"additional dataset {datasetinfo['_FILEPATH']}.")

            incoming_df['ORIGIN'] = datasetinfo['ORIGIN']

            freq_merged = merged_df.index.freq
            freq_incoming = incoming_df.index.freq

            if freq_merged != freq_incoming:
                if freq_merged < freq_incoming:  # < means higher freq
                    merged_df = resample_to_lower_freq(higher=merged_df, lower=incoming_df)
                elif freq_merged > freq_incoming:  # > means lower freq
                    incoming_df = resample_to_lower_freq(higher=incoming_df, lower=merged_df)

            merged_df, incoming_df = _compare_years(primary_df=merged_df, secondary_df=incoming_df, fluxvar=fluxvar)
            merged_df = pd.concat([merged_df, incoming_df], axis=0)
            merged_df.index = pd.to_datetime(merged_df.index)
            merged_df = merged_df.sort_index()
            if merged_df.index.duplicated().sum() > 0:
                raise ValueError(f"Duplicate timestamps found in merged dataset for {site}.")
            merged_df.index.freq = pd.infer_freq(merged_df.index)
            sourcetxt += f"+{datasetinfo['ORIGIN']}"
            datasetinfo_updated['ORIGIN'] = sourcetxt
            datasetinfo_updated = datasetinfo_updated.fillna(datasetinfo)  # Fill gaps w/ lower priority

    # Save merged data to parquet file
    merged_df = merged_df.sort_index()
    start = merged_df.index[0].year
    end = merged_df.index[-1].year
    outname = f"{site}_{igbp}_{sourcetxt}_{start}-{end}"
    outfilepath = dv.save_parquet(filename=outname,
                                  data=merged_df,
                                  outpath=Path(settings['DIR_DATA_PROC_PARQUET']))

    # Save heatmap plot
    save_heatmap_plot(df=merged_df, outname=outname, site=site, igbp=igbp, sourcetxt=sourcetxt,
                      showplot=showplot, fluxvar=fluxvar,
                      outpath=settings['DIR_DATA_PROC_PARQUET_PLOTS'])

    # Add updated dataset info
    datasetinfo_updated['_FILEPATH_PARQUET'] = Path(outfilepath)  # Add filepath to parquet file
    datasetinfo_updated = datasetinfo_updated.to_frame().transpose()
    datasets_df = datasets_df[~datasets_df['SITE'].str.contains(site)]  # Remove old dataset info (often multiple)
    datasets_df = pd.concat([datasets_df, datasetinfo_updated], ignore_index=True)  # Add updated info at end of df

    sites_done.append(site)

    return datasets_df, sites_done


def save_subset_heatmap_plot(df: pd.DataFrame, outpath: str, outname: str, site: str,
                             igbp: str, sourcetxt: str, showplot: bool, fluxvars: list):
    # Heatmap plots
    outfile = Path(outpath) / outname
    print(f"Saving heatmap plot to {outfile} ...")

    fig = plt.figure(facecolor='white', figsize=(40, 10), dpi=72)
    gs = gridspec.GridSpec(1, 8)  # rows, cols
    # gs.update(wspace=0.7, hspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.07)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
    ax4 = fig.add_subplot(gs[0, 3], sharey=ax1)
    ax5 = fig.add_subplot(gs[0, 4], sharey=ax1)
    ax6 = fig.add_subplot(gs[0, 5], sharey=ax1)
    ax7 = fig.add_subplot(gs[0, 6], sharey=ax1)
    ax8 = fig.add_subplot(gs[0, 7], sharey=ax1)
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

    tickkwargs = dict(labeltop=False, labelbottom=True, labelright=False,
                      top=False, bottom=True, left=True, right=False)

    for ix, v in enumerate(fluxvars):
        ax = axes[ix]
        vmin = df[v].quantile(0.02)
        vmax = df[v].quantile(0.98)
        dv.heatmapdatetime(ax=ax, series=df[v], cb_digits_after_comma=1,
                           vmin=vmin, vmax=vmax, cb_extend='both').plot()
        ax.set_title(f"{v}", fontsize=20)

        if ix > 0:
            ax.set_ylabel("")

    fig.suptitle(f"{site} ({igbp}), {sourcetxt}", fontsize=24)
    fig.tight_layout()
    fig.savefig(outfile, dpi=72)
    if showplot:
        fig.show()


def save_heatmap_plot(df: pd.DataFrame, outpath: str, outname: str, site: str,
                      igbp: str, sourcetxt: str, showplot: bool, fluxvar: str):
    # Heatmap plots
    outfile = Path(outpath) / outname
    print(f"Saving heatmap plot to {outfile} ...")

    flux = df[fluxvar].copy()  # Gap-filled fluxes
    flux_qc = df.loc[df[f'{fluxvar}_QC'] == 0, fluxvar].copy()  # Measured fluxes

    fig = plt.figure(facecolor='white', figsize=(12, 12), dpi=72)
    gs = gridspec.GridSpec(1, 2)  # rows, cols
    gs.update(wspace=0.7, hspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.07)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax)

    plotkwargs = dict(cb_digits_after_comma=0, vmin=-20, vmax=20)
    if isinstance(df, pd.DataFrame):
        if not df.empty:
            dv.heatmapdatetime(ax=ax, series=flux, zlabel=f"{flux.name}", **plotkwargs).plot()
            dv.heatmapdatetime(ax=ax2, series=flux_qc, zlabel=f"{flux_qc.name}", **plotkwargs).plot()

    # Titles
    ax.set_title(f"{site} ({igbp})\n{sourcetxt}\nmerged data (gap-filled)")
    ax2.set_title(f"{site} ({igbp})\n{sourcetxt}\nmerged data (only measured)")

    tickkwargs = dict(labeltop=False, labelbottom=True, top=False, bottom=True,
                      left=True, right=False)
    ax.tick_params(labelleft=True, labelright=False, **tickkwargs)
    ax2.tick_params(labelleft=True, labelright=False, **tickkwargs)

    fig.savefig(outfile, dpi=72)
    if showplot:
        fig.show()


def read_settings_file(filepath_settings) -> dict:
    """Read start values from settings file as strings into dict, with same variable names as in file"""
    with open(filepath_settings, 'r', encoding='utf-8') as f:
        settings_dict = yaml.safe_load(f)
    return settings_dict
