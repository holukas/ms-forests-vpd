"""
Prepare the per-site input data for the XGBoost models.

Settings:
- VARIANT = "": subfolder of data/outputs/20_subsets/. Empty overwrites the
  main analysis subsets; set a name before rerunning with other settings.
- SWC_LAYER = "shallow": soil water layer, "shallow" or "deepest".
- N_WORKERS = 3: sites processed in parallel. With 1, the plots are shown.

Reads: data/outputs/10_datasets/17_datasets_info_parquet_vars_stats_usedsites_era5.csv
Writes, per site, a subset parquet file and a heatmap plot, plus
21_SUBSETS_parquet_vars_stats_subsets.csv and 21_warnings.log.

CD-Ygb reports VPD in Pa instead of hPa. The models use per-site z-scores, so
results are unaffected; its VPD mean in the Extended Data overview table was
corrected by hand.
"""
import logging
import re
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

import src.files as files
from src.common import deepest_swc_per_site, get_variable_names
from src.paths import data_path, load_settings, resolve_stored_path

# Run variant. An empty string writes to the baseline paths and overwrites the
# main analysis subsets. Any other value adds a folder level, e.g. "multilayer".
VARIANT = ""

# Which soil water layer to use. "shallow" keeps SWC_F_MDS_1, which is what the
# main analysis used. "deepest" swaps in the deepest layer per site that
# still holds at least 90 % of layer 1's records in the peak months, for the
# soil water sensitivity run. 128 of 208 sites move, the rest have no deeper
# layer or only gappy ones and stay on layer 1. Set VARIANT as well when using "deepest", or the deep subsets
# overwrite the main analysis ones.
SWC_LAYER = "shallow"

# How many sites to process at the same time. Memory is the limit here, not the
# number of cores. A site parquet file is compressed on disk and three to five
# times larger in memory, and diive copies the whole frame once while it checks
# the timestamps. The largest sites need about 3 GB per worker. Three workers
# failed nothing on 32 GB. Four ran out of memory on US-Ho1.
# With N_WORKERS = 1 the sites run one after the other and the plots are shown.
N_WORKERS = 3


class WarningCollector:
    """Collect warnings in a worker process.

    Stands in for the logging module; the parent process writes the messages
    to the log in site order after the run.
    """

    def __init__(self):
        self.messages = []

    def warning(self, msg):
        self.messages.append(str(msg))


def init_worker():
    """Switch matplotlib to a file-only backend in the worker process."""
    import matplotlib
    matplotlib.use('Agg', force=True)


def process_site(task: tuple) -> tuple:
    """Build the subset for one site.

    Returns the subset info, the collected warnings, and the site ID if the
    site failed. Errors are caught so one failing site does not stop the run.
    """
    ix, siteconfig, settings, showplot, deep_swc = task
    site = str(siteconfig['SITE'])
    collected = WarningCollector()

    try:
        varnames = get_variable_names(siteconfig)  # Variable names for this site
        if site in deep_swc:
            varnames['swc_var'] = deep_swc[site]['swc_var']
            varnames['swc_qc_var'] = deep_swc[site]['swc_qc_var']
        subsetinfo = files.create_subsets_parquet_files(
            site=site,
            igbp=siteconfig['IGBP'],
            origin=siteconfig['ORIGIN'],
            ix=int(ix),
            settings=settings,
            variant=VARIANT,
            showplot=showplot,
            filepath_parquet_fullset=str(resolve_stored_path(siteconfig['_FILEPATH_PARQUET'])),
            varnames=varnames,
            logging=collected
        )
    except Exception as error:
        collected.warning(f"{site} FAILED with {type(error).__name__}: {error}")
        return None, collected.messages, site

    # Some sites can come up empty if e.g. SWC is missing during 4 warmest months
    if not subsetinfo:
        return None, collected.messages, None

    # Which soil water layer went in. Without it a deep run cannot be paired
    # against the shallow one, and the layer per site cannot be reported.
    subsetinfo['SWC_VAR'] = varnames['swc_var']
    layer = re.search(r'_(\d+)$', varnames['swc_var'])
    subsetinfo['SWC_LAYER'] = int(layer.group(1)) if layer else -9999
    subsetinfo['LAT'] = siteconfig['LAT']
    subsetinfo['LON'] = siteconfig['LON']
    subsetinfo['ELEVATION'] = siteconfig['ELEVATION']
    subsetinfo['IGBP'] = siteconfig['IGBP']
    return subsetinfo, collected.messages, None


def main():
    # Load settings
    settings = load_settings()

    # Load datasets info
    infile = data_path("data/outputs/10_datasets/17_datasets_info_parquet_vars_stats_usedsites_era5.csv")
    datasets_df = pd.read_csv(infile)

    # Output folder for this run variant
    outdir = Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
    outdir.mkdir(parents=True, exist_ok=True)

    # Logger
    logging.basicConfig(
        filename=outdir / '21_warnings.log',
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Plots can only be shown when everything runs in this process
    showplot = N_WORKERS == 1

    if SWC_LAYER == "deepest":
        deep_swc = deepest_swc_per_site()
        print(f"Soil water: deepest usable layer, {len(deep_swc)} site(s) move off layer 1")
    elif SWC_LAYER == "shallow":
        deep_swc = {}
    else:
        raise ValueError(f"SWC_LAYER must be 'shallow' or 'deepest', not {SWC_LAYER!r}")

    tasks = []
    for ix, siteconfig in datasets_df.iterrows():
        # if ix < 144:
        #     continue
        # if siteconfig['SITE'] != "AT-Mmg":
        #     continue
        # if siteconfig['IGBP'] != "EBF":
        #     continue
        tasks.append((ix, siteconfig, settings, showplot, deep_swc))

    print(f"\n{'-' * 80}\nProcessing {len(tasks)} sites with {N_WORKERS} worker(s).\n{'-' * 80}")

    if N_WORKERS > 1:
        # chunksize=1 because the sites differ a lot in size.
        # maxtasksperchild=1 starts a fresh worker for every site. Without it a
        # worker keeps the memory of the largest site it has seen so far.
        with Pool(processes=N_WORKERS, initializer=init_worker, maxtasksperchild=1) as pool:
            results = pool.map(process_site, tasks, chunksize=1)
    else:
        results = [process_site(task) for task in tasks]

    # Write the warnings in site order, then collect the subset info
    rows = []
    failed = []
    for subsetinfo, messages, failed_site in results:
        for message in messages:
            logging.warning(message)
        if failed_site:
            failed.append(failed_site)
        if subsetinfo:
            rows.append(pd.DataFrame.from_dict(subsetinfo, orient='index').transpose())

    subsetinfo_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # Save to file
    outfile = outdir / '21_SUBSETS_parquet_vars_stats_subsets.csv'
    print(f"\n{'-' * 80}\nSaving info about {len(subsetinfo_df)} subsets to file {outfile.resolve()}.\n{'-' * 80}")
    subsetinfo_df.to_csv(outfile, index=False)

    if failed:
        print(f"\n{'!' * 80}")
        print(f"{len(failed)} site(s) failed and are missing from the info CSV: {', '.join(failed)}")
        print(f"See {outdir / '21_warnings.log'} for the reason. Rerun before you use this output.")
        print(f"{'!' * 80}")


if __name__ == '__main__':
    main()
