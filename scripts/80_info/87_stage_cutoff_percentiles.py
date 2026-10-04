"""
Where the fixed stage cutoffs fall in each site's empirical distribution of the drivers.

The stages use three cutoffs a, b and c on site-standardized drivers, the standard-normal
quantiles that mark the middle 25 %, the middle 52.5 % and the outer 10 % on each side of a Gaussian.
Standardizing a skewed variable does not make it Gaussian, so at a given site the cutoffs
need not land on those percentiles. For every site and driver, the script reports the share
of records at or below -c, -b, -a, a, b and c next to the Gaussian share. The cutoffs are
read from the defaults of `src.stages.stage_2`, so they follow the stage definitions.

Reads: 30_shap/NEP_ZSCORE/conditional/{site}_shap-conditional_NEP_ZSCORE.parquet
Writes:
- 80_info/NEP_ZSCORE/87_INFO_StageCutoffPercentiles_NEP_ZSCORE.csv: one row per site and driver
- 80_info/NEP_ZSCORE/87_INFO_StageCutoffPercentiles_summary.csv: median and quartiles across
  sites per driver and cutoff, with the Gaussian share
"""
import glob
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm

import src.stages as stg
from src.paths import load_settings

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""                       # empty is the main analysis
DRIVERS = ['TA_ZSCORE', 'VPD_ZSCORE', 'SWC_ZSCORE', 'SWIN_ZSCORE']

shap_type = 'conditional' if CONDITIONAL else 'interventional'
settings = load_settings()
dir_out = Path(settings['DIR_INFO_OUT']) / FLUX
dir_out.mkdir(parents=True, exist_ok=True)

# The cut-offs as the stage functions carry them.
_defaults = inspect.signature(stg.stage_2).parameters
A, B, C = (_defaults[k].default for k in ('a', 'b', 'c'))
CUTS = {'-c': -C, '-b': -B, '-a': -A, 'a': A, 'b': B, 'c': C}
GAUSS = {k: norm.cdf(v) for k, v in CUTS.items()}

d = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type / VARIANT
files = sorted(glob.glob(str(d / f'*_shap-{shap_type}_{FLUX}.parquet')))

rows = []
for f in files:
    site = Path(f).name.split('_')[0]
    t = pq.read_table(f, columns=DRIVERS).to_pandas()
    for drv in DRIVERS:
        z = t[drv].dropna().to_numpy(float)
        row = dict(site=site, driver=drv, n=len(z))
        for k, v in CUTS.items():
            row[f'share_below_{k}'] = float(np.mean(z <= v))
        rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(dir_out / f'87_INFO_StageCutoffPercentiles_{FLUX}.csv', index=False)

# Summary: per driver and cut-off, the spread across sites next to the Gaussian share.
summary = []
for drv, g in out.groupby('driver'):
    for k in CUTS:
        s = g[f'share_below_{k}']
        summary.append(dict(driver=drv, cutoff=k, sigma=CUTS[k], gaussian=GAUSS[k],
                            sites=len(s), q25=s.quantile(0.25), median=s.median(),
                            q75=s.quantile(0.75), median_minus_gaussian=s.median() - GAUSS[k]))
summary = pd.DataFrame(summary)
summary.to_csv(dir_out / '87_INFO_StageCutoffPercentiles_summary.csv', index=False)

pd.set_option('display.width', 200)
print(f"{len(files)} sites; cut-offs a={A:.4f}, b={B:.4f}, c={C:.4f} sigma\n")
for drv, g in summary.groupby('driver', sort=False):
    print(drv)
    print(g[['cutoff', 'sigma', 'gaussian', 'q25', 'median', 'q75', 'median_minus_gaussian']]
          .to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print()
worst = summary.loc[summary['median_minus_gaussian'].abs().idxmax()]
print(f"largest departure of a site median from the Gaussian share: {worst['driver']} at "
      f"{worst['cutoff']}, {worst['median']:.3f} against {worst['gaussian']:.3f} "
      f"({100 * worst['median_minus_gaussian']:+.1f} points)")
print(f"\nSaved {dir_out / f'87_INFO_StageCutoffPercentiles_{FLUX}.csv'} and "
      f"87_INFO_StageCutoffPercentiles_summary.csv")
