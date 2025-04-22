from pathlib import Path

import numpy as np
import pandas as pd
from diive.core.io.filereader import ReadFileType
from diive.core.io.files import save_parquet

df = pd.read_csv('../OUT/01_siteinfo.csv')
df = df.fillna(np.nan)
# print(df)

data_nrows = None

_df = df.copy()
for ix, row in _df.iterrows():
    site = row['SITE']
    igbp = row['IGBP']
    origin = row['ORIGIN']
    filepath_icos = row['_FILEPATH_ICOS']
    filepath_fxn = row['_FILEPATH_FXN']
    filepath_amf = row['_FILEPATH_AMF']

    # Load data
    if origin == 'ICOS+FLUXNET':
        # todo deactivated for testing
        continue
        # load_icos = ReadFileType(filetype="FLUXNET-FULLSET-HH-CSV-30MIN", filepath=filepath_icos, data_nrows=data_nrows)
        # icosdf, _ = load_icos.get_filedata()
        # load_fxn = ReadFileType(filetype="FLUXNET-FULLSET-HH-CSV-30MIN", filepath=filepath_fxn, data_nrows=data_nrows)
        # fxndf, _ = load_fxn.get_filedata()
        #
        # # Keep records from FLUXNET data that are not in ICOS data
        # keeplocs = fxndf.index < icosdf.index[0]
        # fxndf = fxndf[keeplocs].copy()
        #
        # # Merge ICOS and FLUXNET data
        # merged_df = pd.concat([icosdf, fxndf], axis=0)
        # merged_df = merged_df.sort_index()
        # sourcetxt = "ICOS+FXN"

    elif origin == 'FLUXNET':
        # todo deactivated for testing
        continue
        # icosdf = None
        # load_fxn = ReadFileType(filetype="FLUXNET-FULLSET-HH-CSV-30MIN", filepath=filepath_fxn, data_nrows=data_nrows)
        # merged_df, _ = load_fxn.get_filedata()
        # sourcetxt = "FXN"

    elif origin == 'ICOS':
        # todo deactivated for testing
        continue
        # fxndf = None
        # load_icos = ReadFileType(filetype="FLUXNET-FULLSET-HH-CSV-30MIN", filepath=filepath_icos, data_nrows=data_nrows)
        # merged_df, _ = load_icos.get_filedata()
        # sourcetxt = "ICOS"

    elif origin == 'AMERIFLUX':
        fxndf = None
        filename = Path(filepath_amf).name
        # Some files are at 60MIN time resolution
        if "_FLUXNET_FULLSET_HH_" in filename:
            filetype = "FLUXNET-FULLSET-HH-CSV-30MIN"
        elif "_FLUXNET_FULLSET_HR_" in filename:
            filetype = "FLUXNET-FULLSET-HR-CSV-60MIN"
        load_amf = ReadFileType(filetype=filetype, filepath=filepath_amf, data_nrows=data_nrows)
        merged_df, _ = load_amf.get_filedata()
        sourcetxt = "AMERIFLUX"

    else:
        raise Exception("Unknown origin.")

    # Save merged data to parquet file
    start = merged_df.index[0].year
    end = merged_df.index[-1].year
    outpath = Path(r"F:\Sync\luhk_work\40 - DATA\Datasets\2025_FORESTS\2-parquet_merged")
    outfilepath = save_parquet(filename=f"DATA-MERGED_{site}_{igbp}_{sourcetxt}_{start}-{end}",
                               data=merged_df,
                               outpath=outpath)

    df.loc[ix, '_FILEPATH_PARQUET'] = Path(outfilepath)

df.to_csv("../OUT/02_siteinfo.csv", index=False)
