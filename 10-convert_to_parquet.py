from funcs.files import convert_datafiles_to_parquet

# Convert ICOS L2 half-hourly files to parquet
convert_datafiles_to_parquet(
    filepattern="ICOSETC_*_FLUXNET_HH_L2.csv",
    filetype="FLUXNET-FULLSET-HH-CSV-30MIN",
    searchdirs=r"L:\Sync\luhk_work\40 - DATA\Datasets\2024 - ICOS - Ecosystem final quality (L2) product in ETC-Archive format - release 2024-1\1-FLUXNET_HH_CSV",
    outpath=r"L:\Sync\luhk_work\40 - DATA\Datasets\2024 - ICOS - Ecosystem final quality (L2) product in ETC-Archive format - release 2024-1\2-FLUXNET_HH_PARQUET"
)

# Convert FLUXNET Warm Winter 2020 half-hourly files to parquet
convert_datafiles_to_parquet(
    filepattern="FLX_*_FLUXNET2015_FULLSET_HH_*.csv",
    filetype="FLUXNET-FULLSET-HH-CSV-30MIN",
    searchdirs=r"F:\Sync\luhk_work\40 - DATA\DATASETS\2022 - FLUXNET - Warm Winter 2020 - release 2022-1",
    outpath=r"F:\Sync\luhk_work\40 - DATA\DATASETS\2022 - FLUXNET - Warm Winter 2020 - release 2022-1\2-FLUXNET_HH_PARQUET"
)
