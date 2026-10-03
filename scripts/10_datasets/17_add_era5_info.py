"""
Add 1991-2020 mean annual temperature (MAT) and precipitation (MAP) to the site table.

The values describe the sites (Fig. 1b, Supplementary Table 1) and enter no analysis.

MAT: mean of the yearly TA_ERA in the ERA5 file of the flux product (ERA5 scaled to the
site sensor by ONEFlux). Sites without such a file (FLUXNET2015 downloads) use
ERA5-Land at the site coordinates (scripts 16c/16d).

MAP: mean of the yearly P_ERA in the same file, except where it is not usable. Then
ERA5-Land precipitation is used:
- no site precipitation: ONEFlux scales ERA precipitation by the ratio of the summed
  site to the summed ERA precipitation (oneflux/downscaling/gapfilling.py). Without
  site data the factor is set to 1, and P_ERA is about 3.6 times too high. Marker:
  ERA_SLOPE of P is -9999 in the BIF (Shuttle) or AUXMETEO file (AmeriFlux).
- no ERA5 file in the flux product.
- P_ERA differs from the MAP reported for the site by more than 25% and ERA5-Land is
  closer to it. P_ERA follows the site rain gauge, and gauges can miss part of the
  precipitation (e.g. snow).

Site overrides: config/site_climate_source_override.csv sets single values to ERA5-Land
by hand, each with its reason.

Site-reported MAT and MAP serve only for these checks and are not written to the table.
Precedence: config/site_climate_reported_manual.csv (web and literature values with
source), then the GRP_CLIM_AVG/CLIM_AVG rows of the site BIF, then
config/AMERIFLUX_site_display_climate_20261003.csv (AmeriFlux site API).

Reads:
- data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv
- the ERA5 YY, BIF and AUXMETEO files in each site folder (_DIRPATH)
- data/outputs/10_datasets/16_ERA5_climate_1991-2020_Copernicus/<SITE>/<SITE>_era5_1991-2020_yearly.csv
Writes, in data/outputs/10_datasets/:
- 17_datasets_info_parquet_vars_stats_usedsites_era5.csv, with the added columns
  ERA5_MAT_1991_2020, ERA5_MAP_1991_2020, ERA5_MAT_SOURCE, ERA5_MAP_SOURCE,
  ERA5_MAT_REASON and ERA5_MAP_REASON (column names kept for the scripts that read them)
- a time-stamped log 17_add_era5_info_*.log with all values and decisions
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from src.paths import data_path, repo_path, resolve_stored_path

YEARS = range(1991, 2021)
MAP_MAX_DEVIATION = 0.25  # P_ERA against the site-reported MAP, relative

# Setup logging
output_dir = data_path("data/outputs/10_datasets")
output_dir.mkdir(parents=True, exist_ok=True)
log_file = output_dir / f"17_add_era5_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        self.closed = False

    def write(self, message):
        self.terminal.write(message)
        if not self.closed:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        if not self.closed:
            self.log.flush()

    def close(self):
        if not self.closed:
            self.log.close()
            self.closed = True


sys.stdout = Logger(log_file)
print("=" * 100)
print("SITE CLIMATE, MAT AND MAP 1991-2020")
print("=" * 100)
print(f"\nLog file: {log_file}\n")

dir_era5land = data_path("data/outputs/10_datasets/16_ERA5_climate_1991-2020_Copernicus")


def read_era5land(site_id):
    """30-year MAT and MAP from the ERA5-Land yearly file. Raises on gaps."""
    filepath = Path(dir_era5land) / site_id / f"{site_id}_era5_1991-2020_yearly.csv"
    if not filepath.is_file():
        raise FileNotFoundError(f"ERA5-Land file not found for {site_id}: {filepath}")
    df = pd.read_csv(filepath)
    years = pd.to_datetime(df['valid_time']).dt.year
    if sorted(years) != list(YEARS):
        raise Exception(f"{site_id}: expected the years 1991 to 2020, found {sorted(years)}")
    if df[['TA_degC', 'PRECIP_TOT_mm']].isna().any().any():
        raise Exception(f"{site_id}: missing ERA5-Land values (sea cell?) in {filepath}")
    if (df['PRECIP_TOT_mm'] <= 0).any():
        raise Exception(f"{site_id}: a year without precipitation in {filepath}")
    return df['TA_degC'].mean(), df['PRECIP_TOT_mm'].mean()


def read_fluxnet_era(site_dir, site_id):
    """30-year mean TA_ERA and P_ERA from the ERA5 YY file of the flux product."""
    matches = list(site_dir.glob(f"*_{site_id}_FLUXNET_ERA5_YY_*.csv"))
    if len(matches) != 1:
        raise Exception(f"{site_id}: expected one ERA5 YY file in {site_dir}, found {len(matches)}")
    df = pd.read_csv(matches[0])
    df = df.loc[df['TIMESTAMP'].between(YEARS[0], YEARS[-1])]
    if len(df) != len(YEARS) or (df[['TA_ERA', 'P_ERA']] == -9999).any().any():
        raise Exception(f"{site_id}: incomplete 1991-2020 record in {matches[0]}")
    return df['TA_ERA'].mean(), df['P_ERA'].mean()


def read_bif_rows(site_dir):
    """Rows (group, variable, value) of the site BIF, empty if there is none."""
    rows = []
    for f in site_dir.glob("*_BIF_*.csv"):
        if 'VARINFO' in f.name:
            continue
        for line in open(f, encoding='latin-1'):
            p = line.strip().split(',')
            if len(p) >= 5:
                rows.append((p[1], p[2], p[3], p[4]))  # group id, group, variable, value
    return rows


def p_era_fitted(site_dir, bif_rows):
    """False if ONEFlux had no site precipitation to scale P_ERA (ERA_SLOPE of P is -9999)."""
    groups = {}
    for gid, grp, var, val in bif_rows:
        if grp == 'GRP_ERA_DOWN':
            groups.setdefault(gid, {})[var] = val
    slopes = [g.get('ERA_SLOPE') for g in groups.values() if g.get('ERA_VARIABLE') == 'P']
    for f in site_dir.glob("*_FLUXNET_AUXMETEO_*.csv"):
        aux = pd.read_csv(f, dtype=str)
        slopes += aux.loc[(aux['VARIABLE'] == 'P') & (aux['PARAMETER'] == 'ERA_SLOPE'), 'VALUE'].tolist()
    if len(slopes) != 1:
        raise Exception(f"{site_dir.name}: expected one ERA_SLOPE for P, found {slopes}")
    return float(slopes[0]) != -9999


def to_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if value not in (0, -9999) else np.nan  # 0 marks an empty field


# Site-reported climate, used only to check P_ERA
manual = pd.read_csv(repo_path("config/site_climate_reported_manual.csv")).set_index('SITE')
amf = pd.read_csv(repo_path("config/AMERIFLUX_site_display_climate_20261003.csv")).set_index('SITE_ID')
overrides = pd.read_csv(repo_path("config/site_climate_source_override.csv"))
overrides = {(r.SITE, r.VARIABLE): (r.SOURCE, r.REASON) for r in overrides.itertuples()}


def site_reported(site_id, bif_rows, var):
    """Site-reported MAT or MAP and its source, by the precedence in the docstring."""
    if site_id in manual.index and not np.isnan(to_float(manual.at[site_id, var])):
        return to_float(manual.at[site_id, var]), 'manual'
    bif = [to_float(val) for _, grp, v, val in bif_rows if grp in ('GRP_CLIM_AVG', 'CLIM_AVG') and v == var]
    if bif and not np.isnan(bif[0]):
        return bif[0], 'BIF'
    if site_id in amf.index and not np.isnan(to_float(amf.at[site_id, var])):
        return to_float(amf.at[site_id, var]), 'AmeriFlux'
    return np.nan, ''


infile = data_path("data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv")
datasets_df = pd.read_csv(infile)

print("Calculating 30-year MAT and MAP for each site...")
rows = []
for index, row in datasets_df.iterrows():
    site_id = row['SITE']
    mat_e5l, map_e5l = read_era5land(site_id)
    site_dir = resolve_stored_path(row['_DIRPATH'])
    has_fluxnet = row['DOWNLOADED_VIA'] in ['SHUTTLE-CLI', 'AMERIFLUX']
    bif_rows = read_bif_rows(site_dir) if has_fluxnet else []
    map_site, map_site_src = site_reported(site_id, bif_rows, 'MAP')

    if not has_fluxnet:
        mat, mat_src = mat_e5l, 'ERA5-Land'
        map_, map_src, reason = map_e5l, 'ERA5-Land', 'no ERA5 file in the flux product'
        mat_fxn = map_fxn = np.nan
    else:
        mat_fxn, map_fxn = read_fluxnet_era(site_dir, site_id)
        mat, mat_src = mat_fxn, 'FLUXNET TA_ERA'
        map_, map_src, reason = map_fxn, 'FLUXNET P_ERA', ''
        if not p_era_fitted(site_dir, bif_rows):
            map_, map_src, reason = map_e5l, 'ERA5-Land', 'no site precipitation, P_ERA not scaled'
        elif not np.isnan(map_site):
            dev_fxn = abs(map_fxn - map_site) / map_site
            dev_e5l = abs(map_e5l - map_site) / map_site
            if dev_fxn > MAP_MAX_DEVIATION and dev_e5l < dev_fxn:
                map_, map_src = map_e5l, 'ERA5-Land'
                reason = (f"P_ERA {dev_fxn:.0%} off the site-reported MAP ({map_site_src}), "
                          f"ERA5-Land {dev_e5l:.0%}")

    mat_reason = '' if has_fluxnet else 'no ERA5 file in the flux product'

    # Hand-set sources from the override file
    for var in ['MAT', 'MAP']:
        if (site_id, var) not in overrides:
            continue
        src, why = overrides[(site_id, var)]
        if src != 'ERA5-Land':
            raise Exception(f"{site_id}: unknown override source {src}")
        if var == 'MAT':
            mat, mat_src, mat_reason = mat_e5l, src, f"override: {why}"
        else:
            map_, map_src, reason = map_e5l, src, f"override: {why}"

    datasets_df.at[index, 'ERA5_MAT_1991_2020'] = round(mat, 3)
    datasets_df.at[index, 'ERA5_MAP_1991_2020'] = round(map_, 3)
    datasets_df.at[index, 'ERA5_MAT_SOURCE'] = mat_src
    datasets_df.at[index, 'ERA5_MAP_SOURCE'] = map_src
    datasets_df.at[index, 'ERA5_MAT_REASON'] = mat_reason
    datasets_df.at[index, 'ERA5_MAP_REASON'] = reason
    rows.append({'SITE': site_id, 'MAT_FLUXNET': mat_fxn, 'MAT_ERA5L': mat_e5l, 'MAT': mat,
                 'MAP_FLUXNET': map_fxn, 'MAP_ERA5L': map_e5l, 'MAP_SITE': map_site,
                 'MAP_SITE_SOURCE': map_site_src, 'MAP': map_, 'MAP_SOURCE': map_src, 'REASON': reason, 'MAT_SOURCE': mat_src,
                 'MAT_REASON': mat_reason})

# All values and decisions, for the log
log_df = pd.DataFrame(rows).set_index('SITE')
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_colwidth', 80)
print(f"\n{'-' * 100}\nALL SITES (MAT in degC, MAP in mm/year)\n{'-' * 100}")
print(log_df.round(1).to_string())
print(f"\n{'-' * 100}\nSOURCES\n{'-' * 100}")
print(datasets_df['ERA5_MAT_SOURCE'].value_counts().to_string())
print(datasets_df['ERA5_MAP_SOURCE'].value_counts().to_string())
print(datasets_df.loc[datasets_df['ERA5_MAP_REASON'] != '', 'ERA5_MAP_REASON']
      .str.replace(r'P_ERA \d+%.*', 'P_ERA off the site-reported MAP', regex=True).value_counts().to_string())
print(f"MAT overrides: {(datasets_df['ERA5_MAT_REASON'] != '').sum()}")

print(f"\n[OK] All {len(datasets_df)} sites have MAT and MAP")
print(f"  MAT range: {datasets_df['ERA5_MAT_1991_2020'].min():.1f} to {datasets_df['ERA5_MAT_1991_2020'].max():.1f} degC")
print(f"  MAP range: {datasets_df['ERA5_MAP_1991_2020'].min():.1f} to {datasets_df['ERA5_MAP_1991_2020'].max():.1f} mm/year")

# Save to file
datasets_df = datasets_df.sort_values(by=['SITE']).reset_index(drop=True)
outfile = data_path("data/outputs/10_datasets/17_datasets_info_parquet_vars_stats_usedsites_era5.csv")
print(f"\n{'-' * 80}\nSaving info about {len(datasets_df)} datasets to file {outfile}.\n{'-' * 80}")
datasets_df.to_csv(outfile, index=False)
print(f"Log file: {log_file}")

# Close log file
sys.stdout.close()
