from pathlib import Path

import numpy as np
import pandas as pd
from diive.core.io.filereader import ReadFileType
from diive.core.io.files import save_parquet

df = pd.read_csv('../OUT/01_siteinfo.csv')
df = df.fillna(np.nan)
import diive as dv
# print(df)

data_nrows = None

_df = df.copy()
for ix, row in _df.iterrows():
    site = row['SITE']

    # if site != 'CH-Dav':
    #     continue

    igbp = row['IGBP']
    origin = row['ORIGIN']
    filepath_icos = row['_FILEPATH_ICOS']
    filepath_fxn = row['_FILEPATH_FXN']
    filepath_amf = row['_FILEPATH_AMF']

    # Load data
    if origin == 'ICOS+FLUXNET':
        load_icos = ReadFileType(filetype="FLUXNET-FULLSET-HH-CSV-30MIN", filepath=filepath_icos, data_nrows=data_nrows)
        icosdf, _ = load_icos.get_filedata()
        load_fxn = ReadFileType(filetype="FLUXNET-FULLSET-HH-CSV-30MIN", filepath=filepath_fxn, data_nrows=data_nrows)
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

    elif origin == 'FLUXNET':
        icosdf = None
        load_fxn = ReadFileType(filetype="FLUXNET-FULLSET-HH-CSV-30MIN", filepath=filepath_fxn, data_nrows=data_nrows)
        merged_df, _ = load_fxn.get_filedata()
        sourcetxt = "FXN"

    elif origin == 'ICOS':
        fxndf = None
        load_icos = ReadFileType(filetype="FLUXNET-FULLSET-HH-CSV-30MIN", filepath=filepath_icos, data_nrows=data_nrows)
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
        load_amf = ReadFileType(filetype=filetype, filepath=filepath_amf, data_nrows=data_nrows)
        merged_df, _ = load_amf.get_filedata()
        sourcetxt = "AMERIFLUX"

    else:
        raise Exception("Unknown origin.")

    # Save merged data to parquet file
    start = merged_df.index[0].year
    end = merged_df.index[-1].year
    outpath = Path(r"F:\Sync\luhk_work\40 - DATA\Datasets\2025_FORESTS\2-parquet_merged")
    outfilepath = save_parquet(filename=f"{site}_{igbp}_{sourcetxt}_{start}-{end}",
                               data=merged_df,
                               outpath=outpath)

    df.loc[ix, '_FILEPATH_PARQUET'] = Path(outfilepath)

df.to_csv("../OUT/02_siteinfo.csv", index=False)
