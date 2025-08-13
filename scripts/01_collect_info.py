from pathlib import Path

import src.files as files
import src.sites as sites

# Settings
settings = files.read_settings_file("../config/settings.yaml")

# # TODO testing dir ---
# TESTDIR = str(Path(r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\ms_co2_penalty\data\source\testing"))
# # TODO --- testing dir

ecosystems = settings['ECOSYSTEMS']

# AMERIFLUX
site_info_ameriflux = sites.get_site_info_ameriflux(
    pattern_amf=settings['PATTERN_DIR_AMF'],
    infofile_amf=settings['INFOFILE_AMF'],
    # searchdir=TESTDIR  # todo testing
    searchdir=settings['DIR_DATA_RAW_AMF']
)
# print(site_info_ameriflux)

# FLUXNET
site_info_fxn = sites.get_site_info_fxn(
    pattern_fxn=settings['PATTERN_DIR_FXN'],
    infofile_fxn=settings['INFOFILE_FXN'],
    # searchdir=TESTDIR  # todo testing
    searchdir=settings['DIR_DATA_RAW_FXN']
)
# print(site_info_fxn)

# ICOS
site_info_icos = sites.get_site_info_icos(
    pattern_icos=settings['PATTERN_DIR_ICOS'],
    # searchdir=TESTDIR  # todo testing
    searchdir=settings['DIR_DATA_RAW_ICOS']
)
# print(site_info_icos)

# Merge site info
outfile = Path(settings['OUTFILE_SITEINFO'])
siteinfo_df = sites.merge_site_info(
    allsites_fxn=site_info_fxn,
    allsites_amf=site_info_ameriflux,
    allsites_icos=site_info_icos,
    outfile=outfile)

siteinfo_df.to_csv(outfile, index=False)
print(f"Saved site info to file {outfile}.")
