from pathlib import Path

import src.files as files
import src.sites as sites

# Settings
settings = files.read_settings_file("../config/settings.yaml")

ecosystems = settings['ECOSYSTEMS']

# ICOS
print("Collecting site info from ICOS...")
site_info_icos = sites.get_site_info_icos(
    pattern_icos=settings['PATTERN_DIR_ICOS'],
    # searchdir=TESTDIR  # todo testing
    searchdir_icos=settings['DIR_DATA_RAW_ICOS']
)
# print(site_info_icos)

# FLUXNET (from Carbon Portal)
print("Collecting site info from FLUXNET (data from Carbon Portal)...")
site_info_fxn_cp = sites.get_site_info_fxn_cp(
    pattern_fxn_cp=settings['PATTERN_DIR_FXN_CP'],
    infofile_fxn_cp=settings['INFOFILE_FXN_CP'],
    # searchdir=TESTDIR  # todo testing
    searchdir_cp=settings['DIR_DATA_RAW_FXN_CP']
)
# print(site_info_fxn_cp)

# FLUXNET (from fluxnet.org)
print("Collecting site info from FLUXNET (data from fluxnet.org)...")
site_info_fxn_org = sites.get_site_info_fluxnet_ameriflux(
    pattern=settings['PATTERN_DIR_FXN_ORG'],
    infofile=settings['INFOFILE_FXN_ORG'],
    # searchdir=TESTDIR  # todo testing
    searchdir=settings['DIR_DATA_RAW_FXN_ORG'],
    origin='FLUXNET_ORG'
)
site_info_fxn_org = site_info_fxn_org[site_info_fxn_org['IGBP'].isin(ecosystems)].copy()
# print(site_info_fluxnet)

# AMERIFLUX
print("Collecting site info from AMERIFLUX...")
site_info_ameriflux = sites.get_site_info_fluxnet_ameriflux(
    pattern=settings['PATTERN_DIR_AMF'],
    infofile=settings['INFOFILE_AMF'],
    # searchdir=TESTDIR  # todo testing
    searchdir=settings['DIR_DATA_RAW_AMF'],
    origin='AMERIFLUX'
)
# print(site_info_ameriflux)


# Merge site info
outfile = Path(settings['OUTFILE_SITEINFO'])
siteinfo_df = sites.merge_site_info(
    allsites_fxn_cp=site_info_fxn_cp,
    allsites_amf=site_info_ameriflux,
    allsites_icos=site_info_icos,
    allsites_fxn=site_info_fxn_org)

siteinfo_df.to_csv(outfile, index=False)
print(f"Saved site info to file {outfile}.")
