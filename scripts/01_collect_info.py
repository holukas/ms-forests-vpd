from pathlib import Path

import src.files as files
import src.sites as sites

# Settings
settings = files.read_settings_file("../config/settings.yaml")
# TODO testing dir ---
searchdir = str(Path(settings['DIR_DATA_SOURCEFILES']) / 'icos')
# --- TODO testing dir

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
outfile = Path(settings['DIR_DATA_OUT_SITEINFO']) / '01_siteinfo.csv'
siteinfo_df = sites.merge_site_info(
    allsites_fxn=site_info_fxn,
    allsites_amf=site_info_ameriflux,
    allsites_icos=site_info_icos,
    outfile=outfile)

siteinfo_df.to_csv(outfile, index=False)
print(f"Saved site info to file {outfile}.")
