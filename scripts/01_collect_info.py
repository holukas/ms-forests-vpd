from pathlib import Path

import src.files as files
import src.sites as sites

# Settings
settings = files.read_settings_file("../config/settings.yaml")

ecosystems = settings['ECOSYSTEMS']

# JapanFlux2024
print("Collecting site info from JapanFlux2024...")
site_info_japanflux = sites.get_site_info_japanflux(
    pattern_dir=settings['PATTERN_DIR_JPF'],
    infofile=settings['INFOFILE_JPF'],
    pattern_file=settings['PATTERN_FILE_HH_JPF'],
    searchdir=settings['DIR_DATA_RAW_JPF'],
    origin='JAPANFLUX'
)
site_info_japanflux = site_info_japanflux[site_info_japanflux['IGBP'].isin(ecosystems)].copy()
# print(site_info_japanflux)

# ICOS
print("Collecting site info from ICOS...")
site_info_icos = sites.get_site_info_icos(
    pattern_dir=settings['PATTERN_DIR_ICOS'],
    pattern_file=settings['PATTERN_FILE_HH_ICOS'],
    searchdir=settings['DIR_DATA_RAW_ICOS']
)
site_info_icos = site_info_icos[site_info_icos['IGBP'].isin(ecosystems)].copy()
# print(site_info_icos)

# FLUXNET (from Carbon Portal)
print("Collecting site info from FLUXNET (data from Carbon Portal)...")
site_info_fxn_cp = sites.get_site_info_fxn_cp(
    pattern_dir=settings['PATTERN_DIR_FXN_CP'],
    infofile=settings['INFOFILE_FXN_CP'],
    pattern_file=settings['PATTERN_FILE_HH_FXN_CP'],
    searchdir=settings['DIR_DATA_RAW_FXN_CP']
)
site_info_fxn_cp = site_info_fxn_cp[site_info_fxn_cp['IGBP'].isin(ecosystems)].copy()
# print(site_info_fxn_cp)

# FLUXNET (from fluxnet.org)
print("Collecting site info from FLUXNET (data from fluxnet.org)...")
site_info_fxn_org = sites.get_site_info_fluxnet_ameriflux(
    pattern_dir=settings['PATTERN_DIR_FXN_ORG'],
    infofile=settings['INFOFILE_FXN_ORG'],
    pattern_file=settings['PATTERN_FILE_HH_FXN_ORG'],
    searchdir=settings['DIR_DATA_RAW_FXN_ORG'],
    origin='FLUXNET_ORG'
)
site_info_fxn_org = site_info_fxn_org[site_info_fxn_org['IGBP'].isin(ecosystems)].copy()
# print(site_info_fluxnet)

# AMERIFLUX
print("Collecting site info from AMERIFLUX...")
site_info_ameriflux = sites.get_site_info_fluxnet_ameriflux(
    pattern_dir=settings['PATTERN_DIR_AMF'],
    infofile=settings['INFOFILE_AMF'],
    pattern_file=settings['PATTERN_FILE_HH_AMF'],
    searchdir=settings['DIR_DATA_RAW_AMF'],
    origin='AMERIFLUX'
)
site_info_ameriflux = sites.consolidate_duplicate_site_entries_amf(site_info_ameriflux)
site_info_ameriflux = site_info_ameriflux[site_info_ameriflux['IGBP'].isin(ecosystems)].copy()
# print(site_info_ameriflux)


# Merge site info
outfile = Path(settings['OUTFILE_SITEINFO'])
siteinfo_df = sites.merge_site_info(
    allsites_fxn_cp=site_info_fxn_cp,
    allsites_amf=site_info_ameriflux,
    allsites_icos=site_info_icos,
    allsites_fxn=site_info_fxn_org,
    allsites_jpf=site_info_japanflux)


# After combining all site info, check for duplicates
if siteinfo_df['SITE'].duplicated().sum() > 0:
    # Get all rows that are duplicates (including the first occurrence)
    duplicates = siteinfo_df[siteinfo_df['SITE'].duplicated(keep=False)]
    # Get the unique site names from the duplicated rows
    duplicate_site_names = duplicates['SITE'].unique()
    # Format the names for error message
    duplicate_sites_str = ", ".join(duplicate_site_names)
    # Raise the error with the names of the duplicate sites
    raise ValueError(f"Duplicate sites found in site info dataframe: {duplicate_sites_str}")

siteinfo_df.to_csv(outfile, index=False)
print(f"Saved site info to file {outfile}.")
