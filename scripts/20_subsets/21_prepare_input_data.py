"""
Prepare input data for XGBoost models.

Note:
    One of the sites (CD-Ygb) had VPD in the wrong units. It seems that this site
    was the only site that recorded VPD in Pa instead of hPa. I found this issue
    after I ran the analyses. However, all analyses were run on the z-scores from
    each site. Since VPD from CD-Ygb was also transformed to z-scores, results
    are not affected by this issue. For the overview table in the Extended Data
    I manually corrected the reported VPD mean by dividing by 100 (Pa --> hPa).

"""
import logging
from pathlib import Path

import pandas as pd

import src.files as files
from src.common import get_variable_names

# Load datasets info
infile = Path('../../data/outputs/10_datasets/17_datasets_info_parquet_vars_stats_usedsites_era5.csv')
datasets_df = pd.read_csv(infile)

# Load settings
settings = files.read_settings_file("../../config/settings.yaml")

# Logger
OUTDIR = Path('../../data/outputs/20_subsets/')
outfile = OUTDIR / '21_warnings.log'
logging.basicConfig(
    filename=outfile,
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

_datasets_df = datasets_df.copy()
subsetinfo_df = pd.DataFrame()
counter = 0
for ix, siteconfig in _datasets_df.iterrows():
    # if ix < 144:
    #     continue
    # if siteconfig['SITE'] != "US-xUN":
    #     continue
    counter += 1
    varnames = get_variable_names(siteconfig)  # Variable names for this site
    subsetinfo = files.create_subsets_parquet_files(
        site=str(siteconfig['SITE']),
        igbp=siteconfig['IGBP'],
        origin=siteconfig['ORIGIN'],
        ix=int(ix),
        settings=settings,
        filepath_parquet_fullset=str(siteconfig['_FILEPATH_PARQUET']),
        varnames=varnames,
        logging=logging
    )

    # Some sites can come up empty if e.g. SWC is missing during 4 warmest months
    if not subsetinfo:
        continue

    subsetinfo['LAT'] = siteconfig['LAT']
    subsetinfo['LON'] = siteconfig['LON']
    subsetinfo['ELEVATION'] = siteconfig['ELEVATION']
    subsetinfo['IGBP'] = siteconfig['IGBP']
    subsetinfo['IGBP'] = siteconfig['IGBP']

    newrow = pd.DataFrame.from_dict(subsetinfo, orient='index').transpose()
    if counter == 1:
        subsetinfo_df = newrow
    else:
        subsetinfo_df = pd.concat([subsetinfo_df, newrow], ignore_index=True)

# Save to file
outfile = OUTDIR / '21_SUBSETS_parquet_vars_stats_subsets.csv'
print(f"\n{'-' * 80}\nSaving info about {len(subsetinfo_df)} subsets to file {outfile.resolve()}.\n{'-' * 80}")
subsetinfo_df.to_csv(outfile, index=False)
