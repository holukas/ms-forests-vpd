from pathlib import Path

import numpy as np
import pandas as pd

import files
import sites

# Settings
settings = files.read_settings_file("settings.yaml")
filepatterns = [settings['PATTERN_FILE_HH_ICOS'], settings['PATTERN_FILE_HH_FXN']]
ecosystems = settings['ECOSYSTEMS']
searchdir = str(Path(settings['BASEDIR']))
pattern_fxn = settings['PATTERN_DIR_FXN']
pattern_icos = settings['PATTERN_DIR_ICOS']

# ---

# Get info for FLUXNET sites
fxn = sites.FluxnetIcosSiteList(searchdir=searchdir, identifiers=pattern_fxn, origin="FLUXNET")
fxn.run()
allsites_fxn = fxn.get_site_info()

# Read CSV with additional site info from EFDC / FLUXNET
infofile_fxn = r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\ms_co2_penalty\docs\SitesList_20240404.csv"
siteinfo_fxn = pd.read_csv(infofile_fxn)
siteinfo_fxn = siteinfo_fxn[['Site Code', 'IGBP Code', 'Site Latitude', 'Site Longitude']].copy()

# Add info to site df
allsites_fxn = allsites_fxn.merge(siteinfo_fxn, left_on='SITE', right_on='Site Code')
allsites_fxn = allsites_fxn.drop(columns='Site Code', inplace=False)

# Renaming
rename_dict = {
    'IGBP Code': 'IGBP',
    'Site Latitude': 'LAT',
    'Site Longitude': 'LON',
}
allsites_fxn = allsites_fxn.rename(columns=rename_dict, inplace=False)
allsites_fxn = allsites_fxn.sort_values(by=['SITE'], ascending=True, inplace=False)

# ---------

# Get info for ICOS sites
icos = sites.FluxnetIcosSiteList(searchdir=searchdir, identifiers=pattern_icos, origin='ICOS')
icos.run()
allsites_icos = icos.get_site_info()

# Read CSV with additional site info from ICOS
_allsites_icos = allsites_icos.copy()
for ix, row in _allsites_icos.iterrows():
    site = row['SITE']

    # Get info from ICOS file
    infofile_icos = f"ICOSETC_{site}_SITEINFO_L2.csv"
    infofile_icos_path = Path(row['_DIRPATH_ICOS']) / infofile_icos
    info_icos = pd.read_csv(infofile_icos_path)

    # Check if sites match
    checksite = str(list(set(info_icos['SITE_ID'].tolist()))[0])
    if site != checksite:
        raise Exception("Site does not match.")

    elev_icos = info_icos[info_icos['VARIABLE'] == 'LOCATION_ELEV']['DATAVALUE'].iloc[0]
    lon_icos = info_icos[info_icos['VARIABLE'] == 'LOCATION_LONG']['DATAVALUE'].iloc[0]
    lat_icos = info_icos[info_icos['VARIABLE'] == 'LOCATION_LAT']['DATAVALUE'].iloc[0]
    igbp_icos = info_icos[info_icos['VARIABLE'] == 'IGBP']['DATAVALUE'].iloc[0]

    allsites_icos.loc[allsites_icos['SITE'] == site, 'ELEVATION'] = elev_icos
    allsites_icos.loc[allsites_icos['SITE'] == site, 'LON'] = lon_icos
    allsites_icos.loc[allsites_icos['SITE'] == site, 'LAT'] = lat_icos
    allsites_icos.loc[allsites_icos['SITE'] == site, 'IGBP'] = igbp_icos

# print(allsites_icos)

# ---------

allsites = pd.concat([allsites_fxn, allsites_icos], axis=0, ignore_index=True)
allsites = allsites.reset_index(drop=True)
allsites = allsites.fillna(np.nan)
# print(allsites)

allsites_combined = pd.DataFrame(columns=allsites.columns)

uniq_sites = list(set(allsites['SITE'].tolist()))

for ix, u in enumerate(uniq_sites):
    row = None
    _df = allsites.loc[allsites['SITE'] == u, :]
    n_records = len(_df)

    # Both ICOS and FLUXNET data available
    if n_records == 2:
        row = _df.loc[_df['ORIGIN'] == 'ICOS']
        row = row.set_index('SITE', drop=False)  # Set index for .fillna()
        row_fxn = _df.loc[_df['ORIGIN'] == 'FLUXNET']
        row_fxn = row_fxn.set_index('SITE', drop=False)
        row = row.fillna(row_fxn)
        row['ORIGIN'] = 'ICOS+FLUXNET'

    # Only 1 available
    elif n_records == 1:
        if not _df.loc[_df['ORIGIN'] == 'ICOS'].empty:
            row = _df.loc[_df['ORIGIN'] == 'ICOS']
            # row = row.set_index('SITE', drop=False)
        elif not _df.loc[_df['ORIGIN'] == 'FLUXNET'].empty:
            row = _df.loc[_df['ORIGIN'] == 'FLUXNET']
            # row = row.set_index('SITE', drop=False)

    else:
        raise Exception(f"{n_records} entries not allowed, only 1 or 2.")

    allsites_combined = pd.concat([allsites_combined, row], axis=0, ignore_index=True)


cols = [c for c in allsites_combined.columns if not str(c).startswith('_')]
auxcols = [cols.append(c) for c in allsites_combined.columns if str(c).startswith('_')]
allsites_combined = allsites_combined[cols]
allsites_combined = allsites_combined.sort_values(by='SITE', inplace=False, ascending=True, ignore_index=True)

allsites_combined.to_csv('../OUT/01_siteinfo.csv', index=False)

print(allsites_combined)
