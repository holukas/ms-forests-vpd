import pandas as pd

from diive.core.io.filereader import ReadFileType, search_files
from diive.core.io.files import save_parquet

# pd.options.display.width = None
# pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

# Settings
PATTERN = "ICOSETC_*_FLUXNET_HH_L2.csv"
FILETYPE = "FLUXNET-FULLSET-HH-CSV-30MIN"
SEARCHDIRS = r"L:\Sync\luhk_work\40 - DATA\Datasets\2024 - ICOS - Ecosystem final quality (L2) product in ETC-Archive format - release 2024-1\1-FLUXNET_HH_CSV"
OUTPATH = r"L:\Sync\luhk_work\40 - DATA\Datasets\2024 - ICOS - Ecosystem final quality (L2) product in ETC-Archive format - release 2024-1\2-FLUXNET_HH_PARQUET"

filelist = search_files(
    searchdirs=SEARCHDIRS,
    pattern=PATTERN)

for f in filelist:
    filename = f.name
    loaddatafile = ReadFileType(filetype=FILETYPE, filepath=f, data_nrows=None)
    data_df, metadata_df = loaddatafile.get_filedata()
    filepath = save_parquet(filename=filename, data=data_df, outpath=OUTPATH)
    print(f"Saved file {filename} to {filepath}.")
