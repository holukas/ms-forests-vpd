from pathlib import Path

import pandas as pd

import src.files as files
import src.sites as sites

# Settings
settings = files.read_settings_file("../../config/settings.yaml")
ecosystems = settings['ECOSYSTEMS']

# MULTIPLE NETWORKS (via fluxnet-shuttle)
print("Collecting dataset info from MULTIPLE NETWORKS (fluxnet-shuttle) ...")
datasets_shuttle = sites.get_dataset_info_fluxnet_shuttle(
    pattern_dir=settings['PATTERN_DIR_SHUTTLE'],
    infofile=settings['INFOFILE_SHUTTLE'],
    pattern_file=settings['PATTERN_FILE_HH_SHUTTLE'],
    searchdir=settings['DIR_DATA_RAW_SHUTTLE'],
    downloaded_via='SHUTTLE-CLI'
)

# JapanFlux2024
print("Collecting dataset info from JapanFlux2024...")
datasets_japanflux = sites.get_dataset_info_japanflux(
    pattern_dir=settings['PATTERN_DIR_JPF'],
    infofile=settings['INFOFILE_JPF'],
    pattern_file=settings['PATTERN_FILE_HH_JPF'],
    searchdir=settings['DIR_DATA_RAW_JPF'],
    downloaded_via='JAPANFLUX-URL'
)

# AMERIFLUX
print("Collecting dataset info from AMERIFLUX...")
datasets_ameriflux = sites.get_dataset_info_fluxnet_ameriflux(
    pattern_dir=settings['PATTERN_DIR_AMF'],
    infofile=settings['INFOFILE_AMF'],
    pattern_file=settings['PATTERN_FILE_HH_AMF'],
    searchdir=settings['DIR_DATA_RAW_AMF'],
    downloaded_via='AMERIFLUX'
)

# # ICOS (all sites part of fluxnet-shuttle downloads)
# print("Collecting dataset info from ICOS...")
# datasets_icos = sites.get_dataset_info_icos(
#     pattern_dir=settings['PATTERN_DIR_ICOS'],
#     pattern_file=settings['PATTERN_FILE_HH_ICOS'],
#     searchdir=settings['DIR_DATA_RAW_ICOS']
# )

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
    downloaded_via='FLUXNET_ORG'
)

# Merge datasets info
datasets_df = pd.concat([datasets_shuttle, datasets_japanflux, datasets_ameriflux,
                         datasets_fxn_cp, datasets_fxn_org], axis=0, ignore_index=True)
datasets_df = datasets_df.reset_index(drop=True)
# datasets_df = datasets_df.fillna("n.a.")
datasets_df = datasets_df.sort_values(by=['SITE'], inplace=False)
datasets_df = datasets_df[datasets_df['IGBP'].isin(ecosystems)].copy()

print(len(set(datasets_df['SITE'].tolist())))
# datasets_df['SITE'].loc[datasets_df['SITE'].duplicated()]

# Set dataset priority for merging of overlapping datasets, 1=top priority when datasets overlap
priority = {"SHUTTLE-CLI": 1, "ICOS": 2, "AMERIFLUX": 2, "JAPANFLUX-URL": 2, "FLUXNET_CP": 3, "FLUXNET_ORG": 4}

# Map the 'DOWNLOADED_VIA' column to the integer values
datasets_df['PRIORITY'] = datasets_df['DOWNLOADED_VIA'].map(priority)

# Save to file
outfile = Path('../../data/outputs/10_datasets/11_datasets_info.csv')
print(f"\n{'-' * 80}\nSaving info about {len(datasets_df)} datasets to file {outfile}.\n{'-' * 80}")
datasets_df.to_csv(outfile, index=False)
