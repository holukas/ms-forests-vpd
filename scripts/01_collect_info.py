from pathlib import Path

import src.files as files
import src.sites as sites

# Settings
settings = files.read_settings_file("../config/settings.yaml")
searchdir = str(Path(settings['BASEDIR']))

ecosystems = settings['ECOSYSTEMS']

# AMERIFLUX
site_info_ameriflux = sites.get_site_info_ameriflux(
    pattern_amf=settings['PATTERN_DIR_AMF'],
    infofile_amf=settings['INFOFILE_AMF'],
    searchdir=searchdir)
# print(site_info_ameriflux)

# FLUXNET
site_info_fxn = sites.get_site_info_fxn(
    pattern_fxn=settings['PATTERN_DIR_FXN'],
    infofile_fxn=settings['INFOFILE_FXN'],
    searchdir=searchdir)
# print(site_info_fxn)

# ICOS
site_info_icos = sites.get_site_info_icos(
    pattern_icos=settings['PATTERN_DIR_ICOS'],
    searchdir=searchdir)
# print(site_info_icos)

# Merge site info
site_info = sites.merge_site_info(site_info_fxn, site_info_ameriflux, site_info_icos)
print(site_info)
