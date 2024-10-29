"""
Convert ICOS L2 and FLUXNET half-hourly CSV files to parquet
"""

from pathlib import Path

from funcs.files import convert_datafiles_to_parquet

# Input and output folder for each ecosystem
basedir = r"L:\Sync\luhk_work\40 - DATA\Datasets\2024 - FLUXNET ICOS FORESTS"
ecosystems = ['DBF', 'DNF', 'EBF', 'ENF', 'MF']

# Conversion to parquet
for ecosystem in ecosystems:
    searchdir = Path(basedir).joinpath(ecosystem) / "1-FLUXFILES_CSV"
    outpath = Path(basedir).joinpath(ecosystem) / "2-FLUXFILES_PARQUET"
    convert_datafiles_to_parquet(
        filepatterns=["ICOSETC_*_FLUXNET_HH_L2.csv", "FLX_*_FLUXNET2015_FULLSET_HH_*.csv"],
        filetype="FLUXNET-FULLSET-HH-CSV-30MIN",
        searchdir=str(searchdir),
        outpath=str(outpath)
    )
