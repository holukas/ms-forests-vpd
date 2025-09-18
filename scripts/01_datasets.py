from pathlib import Path

import numpy as np
import pandas as pd

import src.files as files
import src.sites as sites

# Settings
settings = files.read_settings_file("../config/settings.yaml")
ecosystems = settings['ECOSYSTEMS']

# JapanFlux2024
print("Collecting dataset info from JapanFlux2024...")
datasets_japanflux = sites.get_dataset_info_japanflux(
    pattern_dir=settings['PATTERN_DIR_JPF'],
    infofile=settings['INFOFILE_JPF'],
    pattern_file=settings['PATTERN_FILE_HH_JPF'],
    searchdir=settings['DIR_DATA_RAW_JPF'],
    origin='JAPANFLUX'
)

# ICOS
print("Collecting dataset info from ICOS...")
datasets_icos = sites.get_dataset_info_icos(
    pattern_dir=settings['PATTERN_DIR_ICOS'],
    pattern_file=settings['PATTERN_FILE_HH_ICOS'],
    searchdir=settings['DIR_DATA_RAW_ICOS']
)

# FLUXNET (from Carbon Portal)
print("Collecting dataset info from FLUXNET (data from Carbon Portal)...")
datasets_fxn_cp = sites.get_dataset_info_fxn_cp(
    pattern_dir=settings['PATTERN_DIR_FXN_CP'],
    infofile=settings['INFOFILE_FXN_CP'],
    pattern_file=settings['PATTERN_FILE_HH_FXN_CP'],
    searchdir=settings['DIR_DATA_RAW_FXN_CP']
)

# FLUXNET (from fluxnet.org)
print("Collecting dataset info from FLUXNET (data from fluxnet.org)...")
datasets_fxn_org = sites.get_dataset_info_fluxnet_ameriflux(
    pattern_dir=settings['PATTERN_DIR_FXN_ORG'],
    infofile=settings['INFOFILE_FXN_ORG'],
    pattern_file=settings['PATTERN_FILE_HH_FXN_ORG'],
    searchdir=settings['DIR_DATA_RAW_FXN_ORG'],
    origin='FLUXNET_ORG'
)

# AMERIFLUX
print("Collecting dataset info from AMERIFLUX...")
datasets_ameriflux = sites.get_dataset_info_fluxnet_ameriflux(
    pattern_dir=settings['PATTERN_DIR_AMF'],
    infofile=settings['INFOFILE_AMF'],
    pattern_file=settings['PATTERN_FILE_HH_AMF'],
    searchdir=settings['DIR_DATA_RAW_AMF'],
    origin='AMERIFLUX'
)

# Merge datasets info
datasets_df = pd.concat([datasets_fxn_cp, datasets_ameriflux, datasets_icos, datasets_fxn_org, datasets_japanflux],
                        axis=0, ignore_index=True)
datasets_df = datasets_df.reset_index(drop=True)
datasets_df = datasets_df.fillna(np.nan)
datasets_df = datasets_df.sort_values(by=['SITE'], inplace=False)
datasets_df = datasets_df[datasets_df['IGBP'].isin(ecosystems)].copy()

# Save to file
outfile = Path('../data/outputs/01_datasets.csv')
print(f"\n{'-' * 80}\nSaving info about {len(datasets_df)} datasets to file {outfile}.\n{'-' * 80}")
datasets_df.to_csv(outfile, index=False)
