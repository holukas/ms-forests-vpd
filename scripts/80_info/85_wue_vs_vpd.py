"""
Water use efficiency (GPP/ET) against VPD, a check of the stomatal reading of the by-flux figure (script 58).

If stomatal closure drives the GPP turnover, GPP per unit ET should fall across the whole
VPD range. This is the measured relationship, not a SHAP attribution, so it cannot separate
stomatal control from anything that covaries with VPD.

- Per VPD bin, mean GPP over mean ET (per-record ratios explode where ET is near zero),
  in micromol CO2 per mmol H2O.
- Each site is scaled by its own overall value; sites are weighted equally and bins need
  half the sites, as in scripts 47, 52, 54 and 58.
- Per site, the slope of log water use efficiency on log VPD: -1 means no stomatal
  adjustment, about -0.5 matches optimal stomatal theory.
- The printed split at the GPP turnover uses a hard-coded 0.18 sigma.

Reads: 30_shap/NEP_ZSCORE/conditional/{SITE}_shap-conditional_NEP_ZSCORE.parquet,
21_SUBSETS_parquet_vars_stats_subsets.csv.
Writes to 80_info/NEP_ZSCORE/: 85_INFO_WueVsVpd_perSite.csv,
85_INFO_WueVsVpd_acrossSites.csv, 85_INFO_WueVsVpd_exponents.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.paths import data_path, load_settings

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""

BINSIZE = 0.1           # sigma, the bin width stages 41 and 42 use
MIN_RECORDS_PER_BIN = 10
MIN_ET = 0.01           # mm h-1, below this the ratio is not a water use efficiency
MMOL_PER_MM_H = 1000 / 18.015 / 3.6   # mm H2O h-1 to mmol H2O m-2 s-1, 15.42

shap_type = 'conditional' if CONDITIONAL else 'interventional'
settings = load_settings()
dir_shap = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type / VARIANT
dir_out = Path(settings['DIR_INFO_OUT']) / FLUX
dir_out.mkdir(parents=True, exist_ok=True)

subsets = pd.read_csv(data_path('data/outputs/20_subsets/'
                                '21_SUBSETS_parquet_vars_stats_subsets.csv'))
EDGES = np.round(np.arange(-8, 10, BINSIZE), 1)


def site_curve(site):
    """Water use efficiency per VPD bin for one site, plus the site's own overall value."""
    fp = dir_shap / f"{site}_shap-{shap_type}_{FLUX}.parquet"
    if not fp.is_file():
        return None
    d = pd.read_parquet(fp, columns=['GPP', 'ET', 'VPD_ZSCORE']).dropna()
    if d.empty:
        return None

    site_wue = d['GPP'].mean() / (d['ET'].mean() * MMOL_PER_MM_H)
    if not np.isfinite(site_wue) or site_wue <= 0:
        return None

    d = d.assign(bin=np.round(EDGES[np.digitize(d['VPD_ZSCORE'], EDGES) - 1], 1))
    g = d.groupby('bin').agg(gpp=('GPP', 'mean'), et=('ET', 'mean'), n=('GPP', 'size'))
    g = g.loc[(g['n'] >= MIN_RECORDS_PER_BIN) & (g['et'] > MIN_ET)]
    if g.empty:
        return None

    g['wue'] = g['gpp'] / (g['et'] * MMOL_PER_MM_H)
    g['wue_relative'] = g['wue'] / site_wue
    return g.reset_index().assign(SITE=site, site_wue=site_wue)


rows = []
for ix, site in enumerate(subsets['SITE']):
    out = site_curve(site)
    if out is None:
        print(f"  no usable records for {site}, skipping")
        continue
    rows.append(out)
    if (ix + 1) % 25 == 0:
        print(f"  {ix + 1} sites")

per_site = pd.concat(rows, ignore_index=True)
n_sites = per_site['SITE'].nunique()
print(f"\n{n_sites} sites, {len(per_site)} site bins")
print(f"site water use efficiency, micromol CO2 per mmol H2O: "
      f"median {per_site.groupby('SITE')['site_wue'].first().median():.2f}, "
      f"range {per_site['site_wue'].min():.2f} to {per_site['site_wue'].max():.2f}")

# Across sites: equal weight per site, and only bins that at least half the sites reach.
across = per_site.groupby('bin').agg(
    n_sites=('SITE', 'nunique'),
    wue_relative=('wue_relative', 'median'),
    q25=('wue_relative', lambda s: s.quantile(0.25)),
    q75=('wue_relative', lambda s: s.quantile(0.75)),
    wue_absolute=('wue', 'median')).reset_index()
across = across.loc[across['n_sites'] >= np.ceil(n_sites / 2)].reset_index(drop=True)

per_site.to_csv(dir_out / '85_INFO_WueVsVpd_perSite.csv', index=False)
across.to_csv(dir_out / '85_INFO_WueVsVpd_acrossSites.csv', index=False)

# Does it fall across the whole range, or only past the GPP turnover at 0.18 sigma?
x, y = across['bin'].to_numpy(), across['wue_relative'].to_numpy()
rho, p = stats.spearmanr(x, y)
below = x <= 0.18
print(f"\nVPD range kept: {x.min():.1f} to {x.max():.1f} sigma, {len(x)} bins")
print(f"relative water use efficiency: {y[0]:.3f} at {x[0]:.1f}, {y[-1]:.3f} at {x[-1]:.1f}")
print(f"Spearman over the whole range: rho {rho:.3f}, p {p:.2g}")
for label, mask in [('below the GPP turnover, up to 0.18 sigma', below),
                    ('above it', ~below)]:
    if mask.sum() > 2:
        r, pp = stats.spearmanr(x[mask], y[mask])
        slope = np.polyfit(x[mask], y[mask], 1)[0]
        print(f"  {label}: {mask.sum()} bins, rho {r:.3f}, p {pp:.2g}, "
              f"slope {slope:+.3f} per sigma")

# The exponent, fitted per site in physical units. The bins are in sigma, so each site's bin
# center goes back to its own absolute VPD with that site's mean and standard deviation.
# CD-Ygb records VPD in Pa where every other site uses hPa, the same conversion as in stage 47.
z0, sd = subsets.set_index('SITE')['VPD_Z0'].copy(), subsets.set_index('SITE')['VPD_SD'].copy()
z0.loc['CD-Ygb'] /= 100
sd.loc['CD-Ygb'] /= 100
per_site['vpd_hpa'] = per_site['SITE'].map(z0) + per_site['bin'] * per_site['SITE'].map(sd)

fits = []
usable = per_site.loc[(per_site['vpd_hpa'] > 0) & (per_site['wue'] > 0)]
for site, g in usable.groupby('SITE'):
    if len(g) < 8:
        continue
    slope, _ = np.polyfit(np.log(g['vpd_hpa']), np.log(g['wue']), 1)
    fits.append({'SITE': site, 'exponent': slope, 'n_bins': len(g)})
fits = pd.DataFrame(fits)
fits.to_csv(dir_out / '85_INFO_WueVsVpd_exponents.csv', index=False)

q25, q50, q75 = fits['exponent'].quantile([0.25, 0.5, 0.75])
print(f"log-log exponent against VPD, {len(fits)} sites: "
      f"median {q50:.3f}, IQR {q25:.3f} to {q75:.3f}")
print(f"  steeper than -1, the no-adjustment case: {(fits['exponent'] < -1).sum()}")
print(f"  between -1 and -0.5: {int(fits['exponent'].between(-1, -0.5).sum())}")
print(f"  shallower than -0.5: {(fits['exponent'] > -0.5).sum()}")
print(f"  against -0.5, p {stats.ttest_1samp(fits['exponent'], -0.5).pvalue:.2g}; "
      f"against -1.0, p {stats.ttest_1samp(fits['exponent'], -1.0).pvalue:.2g}")

print(f"Saved {dir_out / '85_INFO_WueVsVpd_perSite.csv'}")
print(f"Saved {dir_out / '85_INFO_WueVsVpd_acrossSites.csv'}")
print(f"Saved {dir_out / '85_INFO_WueVsVpd_exponents.csv'}")
