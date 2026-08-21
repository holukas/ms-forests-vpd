"""
Prepare input data for XGBoost models.

Writes one subset parquet file and one heatmap plot per site, plus an info CSV and
a warnings log for the run. Everything goes under data/outputs/20_subsets/<VARIANT>/.

VARIANT is empty by default, so the output goes straight to 20_subsets/. That folder
holds the subsets for the submitted figures. Set a name before you rerun with other
settings, or you overwrite them.

Sites are independent, so N_WORKERS of them run at the same time. Set N_WORKERS to 1
to run one site after the other and to see the plots on screen.

Note:
    One of the sites (CD-Ygb) had VPD in the wrong units. It seems that this site
    was the only site that recorded VPD in Pa instead of hPa. I found this issue
    after I ran the analyses. However, all analyses were run on the z-scores from
    each site. Since VPD from CD-Ygb was also transformed to z-scores, results
    are not affected by this issue. For the overview table in the Extended Data
    I manually corrected the reported VPD mean by dividing by 100 (Pa --> hPa).

"""
import logging
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

import src.files as files
from src.common import get_variable_names
from src.paths import data_path, load_settings, resolve_stored_path

# Run variant. An empty string writes to the baseline paths and overwrites the
# submitted subsets. Any other value adds a folder level, e.g. "multilayer".
VARIANT = ""

# How many sites to process at the same time. Each worker loads one full site
# parquet file, and the largest is about 0.5 GB on disk and several GB in memory.
# Four workers fit in 32 GB. Memory is the limit here, not the number of cores.
# With N_WORKERS = 1 the sites run one after the other and the plots are shown.
N_WORKERS = 4


class WarningCollector:
    """Collects warnings inside a worker process.

    Several processes cannot write to one log file safely. Each worker therefore
    keeps its warnings, and the parent process writes them in site order once the
    run is over. Stands in for the logging module, so files.py needs no change.
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
    """Build the subset for one site and return its info and its warnings."""
    ix, siteconfig, settings, showplot = task
    collected = WarningCollector()

    varnames = get_variable_names(siteconfig)  # Variable names for this site
    subsetinfo = files.create_subsets_parquet_files(
        site=str(siteconfig['SITE']),
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

    # Some sites can come up empty if e.g. SWC is missing during 4 warmest months
    if not subsetinfo:
        return None, collected.messages

    subsetinfo['LAT'] = siteconfig['LAT']
    subsetinfo['LON'] = siteconfig['LON']
    subsetinfo['ELEVATION'] = siteconfig['ELEVATION']
    subsetinfo['IGBP'] = siteconfig['IGBP']
    return subsetinfo, collected.messages


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

    tasks = []
    for ix, siteconfig in datasets_df.iterrows():
        # if ix < 144:
        #     continue
        # if siteconfig['SITE'] != "AT-Mmg":
        #     continue
        # if siteconfig['IGBP'] != "EBF":
        #     continue
        tasks.append((ix, siteconfig, settings, showplot))

    print(f"\n{'-' * 80}\nProcessing {len(tasks)} sites with {N_WORKERS} worker(s).\n{'-' * 80}")

    if N_WORKERS > 1:
        # chunksize=1 because the sites differ a lot in size
        with Pool(processes=N_WORKERS, initializer=init_worker) as pool:
            results = pool.map(process_site, tasks, chunksize=1)
    else:
        results = [process_site(task) for task in tasks]

    # Write the warnings in site order, then collect the subset info
    rows = []
    for subsetinfo, messages in results:
        for message in messages:
            logging.warning(message)
        if subsetinfo:
            rows.append(pd.DataFrame.from_dict(subsetinfo, orient='index').transpose())

    subsetinfo_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # Save to file
    outfile = outdir / '21_SUBSETS_parquet_vars_stats_subsets.csv'
    print(f"\n{'-' * 80}\nSaving info about {len(subsetinfo_df)} subsets to file {outfile.resolve()}.\n{'-' * 80}")
    subsetinfo_df.to_csv(outfile, index=False)


if __name__ == '__main__':
    main()
