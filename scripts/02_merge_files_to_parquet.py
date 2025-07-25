from pathlib import Path

import numpy as np
import pandas as pd

import src.files as files

# Load site info
settings = files.read_settings_file("../config/settings.yaml")
infile = Path(settings['DIR_DATA_OUT']) / "01_siteinfo.csv"
siteinfo_df = pd.read_csv(infile)
siteinfo_df = siteinfo_df.fillna(np.nan)

data_nrows = None  # for testing

siteinfo_df = files.create_parquet_files(siteinfo_df=siteinfo_df, data_nrows=data_nrows, settings=settings)
outfile = Path(settings['DIR_DATA_OUT']) / "02_siteinfo.csv"
siteinfo_df.to_csv(outfile, index=False)
print(f"Saved updated site info to file {outfile}.")
