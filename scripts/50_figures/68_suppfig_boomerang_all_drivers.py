"""
PLANNED SUPPLEMENTARY FIGURE, not yet adopted.

Like 67, but every co-driver is held nearly constant, not only air temperature.

Script 67 splits the records into classes of TA and reads the NEP response to VPD inside
each. That removes temperature but leaves soil water and radiation free to travel with
VPD, so its slope is a total response and cannot be attributed to VPD alone. Here the
records are binned jointly by TA, SM and SW, so within a cell all three are nearly
constant and the step from the first VPD bin to the second is close to a pure VPD
contrast. It is the measured counterpart of what conditional SHAP does, without a model.

The cells stay well populated despite the three dimensions: at 10 classes per driver, 956
of 1000 cells hold at least 400 records per VPD bin and together cover 99 % of the record.

Reads the per-site files written by stage 31, which hold the measured values alongside the
SHAP values, so the records are exactly those the models saw.
"""
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import data_path, load_settings

SHOW_PLOT = False

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""
SITE_SUBSET = ""

# Classes per co-driver. 10 gives 1000 cells and keeps them populated.
N_CLASSES = 10
# VPD bins inside each cell. Two keeps the contrast clean.
N_VPD_BINS = 2
# A cell is dropped if either of its VPD bins holds fewer records than this.
MIN_PER_BIN = 100
# Bootstrap replicates over sites. Each rebuilds the whole binning, so this is the cost.
N_BOOT = 100
# Reuse the saved numbers unless something that affects them changed.
RECOMPUTE = False

XVAR, YVAR = 'VPD_ZSCORE', 'NEP_ZSCORE'
COND_VARS = ['TA_ZSCORE', 'SWC_ZSCORE', 'SWIN_ZSCORE']
AGG = 'median'
FIT_RANGE = (-1.5, 2.5)
AX_LABELS_FONTSIZE = 12

shap_type = 'conditional' if CONDITIONAL else 'interventional'
settings = load_settings()
indir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type / VARIANT
outdir = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT / SITE_SUBSET
agg = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT / SITE_SUBSET
outdir.mkdir(parents=True, exist_ok=True)

files = sorted(indir.glob(f'*_shap-{shap_type}_{FLUX}.parquet'))
print(f"Reading {len(files)} site files")
frames = []
for fp in files:
    part = pd.read_parquet(fp, columns=[XVAR, YVAR] + COND_VARS)
    part['SITE'] = fp.name.split('_shap-')[0]
    frames.append(part)
df = pd.concat(frames, axis=0).dropna()
igbp_map = (pd.read_csv(data_path('data/outputs/20_subsets/'
                                  '21_SUBSETS_parquet_vars_stats_subsets.csv'))
            .set_index('SITE')['IGBP'])
df['IGBP'] = df['SITE'].map(igbp_map)
print(f"{len(df):,} records from {df['SITE'].nunique()} sites")

PANELS = [('Global forests', None), ('ENF', 'ENF'), ('DBF', 'DBF'),
          ('EBF', 'EBF'), ('MF', 'MF')]


def bin_jointly(data):
    """Quantile cells over the three co-drivers, then VPD bins inside each cell."""
    work = data[[XVAR, YVAR] + COND_VARS].copy()
    cell = np.zeros(len(work), dtype=np.int64)
    for var in COND_VARS:
        code = pd.qcut(work[var], N_CLASSES, labels=False, duplicates='drop')
        cell = cell * N_CLASSES + code.to_numpy()
    work['_cell'] = cell
    work['_x'] = work.groupby('_cell', observed=True)[XVAR].transform(
        lambda v: pd.qcut(v, N_VPD_BINS, labels=False, duplicates='drop'))
    grouped = work.groupby(['_cell', '_x'], observed=True)
    out = grouped.agg(**{XVAR: (XVAR, AGG), YVAR: (YVAR, AGG),
                         'COUNTS': (YVAR, 'size')}).reset_index()
    # Both bins of a cell must survive, otherwise there is no contrast to read.
    keep = out.groupby('_cell')['COUNTS'].transform('min') >= MIN_PER_BIN
    full = out.groupby('_cell')['_x'].transform('size') == N_VPD_BINS
    return out.loc[keep & full]


def segment_slopes(res):
    """One slope per cell, with the VPD interval it was measured over."""
    by_cell = res.sort_values(['_cell', XVAR]).groupby('_cell')
    out = pd.DataFrame({
        'vpd_start': by_cell[XVAR].first(),
        'vpd_end': by_cell[XVAR].last(),
        'dy': by_cell[YVAR].last() - by_cell[YVAR].first(),
    })
    out['dx'] = out['vpd_end'] - out['vpd_start']
    out = out.loc[out['dx'] > 0]
    out['slope'] = out['dy'] / out['dx']
    out['vpd'] = (out['vpd_start'] + out['vpd_end']) / 2
    out['weight'] = 1.0 / out['dx']
    return out


def zero_crossing(x, y, w):
    """VPD where a width weighted quadratic fit of slope against VPD reaches zero."""
    if len(x) < 5:
        return float('nan')
    coef = np.polyfit(x, y, 2, w=np.sqrt(w))
    roots = np.roots(coef)
    roots = np.real(roots[np.abs(np.imag(roots)) < 1e-9])
    inside = roots[(roots >= x.min()) & (roots <= x.max())]
    return float(min(inside, key=abs)) if len(inside) else float('nan')


def threshold_from(res):
    sl = segment_slopes(res)
    sl = sl.loc[sl['vpd'].between(*FIT_RANGE)]
    return zero_crossing(sl['vpd'].values, sl['slope'].values, sl['weight'].values)


def measured_threshold(data, rng_seed=42):
    """Threshold with a bootstrap interval, resampling sites rather than cells.

    Cells share sites, so resampling cells would give an interval far too narrow. The
    unit that can be resampled honestly is the site.
    """
    point = threshold_from(bin_jointly(data))
    codes, sites = pd.factorize(data['SITE'].values)
    order = np.argsort(codes, kind='stable')
    bounds = np.searchsorted(codes[order], np.arange(len(sites) + 1))
    positions = [order[bounds[i]:bounds[i + 1]] for i in range(len(sites))]
    rng_bs = np.random.default_rng(rng_seed)
    boot = []
    for _ in range(N_BOOT):
        pick = rng_bs.integers(0, len(sites), len(sites))
        rows = np.concatenate([positions[i] for i in pick])
        val = threshold_from(bin_jointly(data.take(rows)))
        if np.isfinite(val):
            boot.append(val)
    boot = np.asarray(boot)
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan))
    return point, lo, hi


conv = pd.read_csv(agg / '48_Threshold_zscore_to_kPa.csv').set_index('group')
MODEL_Z = {'Global forests': float(conv.loc['ALL SITES', 'z'])}
for g in ('ENF', 'DBF', 'EBF', 'MF'):
    MODEL_Z[g] = float(conv.loc[g, 'z'])

outfile = outdir / (f'68_PLANNED-SUPPFIG_Boomerang_AllDrivers_{YVAR}_vs_{XVAR}'
                    f'_by{N_CLASSES}cubed_perBiome.png')
cache_binned = Path(str(outfile).replace('.png', '_DATA.csv'))
cache_thresholds = Path(str(outfile).replace('.png', '_THRESHOLDS.csv'))
cached = (not RECOMPUTE) and cache_binned.exists() and cache_thresholds.exists()

all_results = {}
if cached:
    print(f"reusing {cache_binned.name}, set RECOMPUTE = True to redo the numbers")
    stored = pd.read_csv(cache_binned)
    for title, _ in PANELS:
        all_results[title] = stored.loc[stored['panel'] == title].drop(columns='panel')
    thresholds = pd.read_csv(cache_thresholds).set_index('panel')
else:
    for title, igbp in PANELS:
        subset = df if igbp is None else df.loc[df['IGBP'] == igbp]
        res = bin_jointly(subset)
        print(f"{title}: {len(subset):,} records, {res['_cell'].nunique()} cells kept, "
              f"{100 * res['COUNTS'].sum() / len(subset):.1f} % of records")
        all_results[title] = res
    thresholds = None

slope_frames = {t: segment_slopes(r) for t, r in all_results.items()}
_all = np.concatenate([sf['slope'].values for sf in slope_frames.values()])
norm = mcolors.TwoSlopeNorm(vmin=np.nanpercentile(_all[_all < 0], 2), vcenter=0.0,
                            vmax=np.nanpercentile(_all[_all > 0], 98))
cmap = mcolors.LinearSegmentedColormap.from_list(
    'red_yellow_blue', ['#67001f', '#d6604d', '#f4a582', '#ffe08a', '#ffd23f',
                        '#ffe08a', '#92c5de', '#4393c3', '#053061'])

fig = plt.figure(figsize=(13.5, 7.4), dpi=150, constrained_layout=True)
fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.03, hspace=0.04)
gs = fig.add_gridspec(3, 4, width_ratios=[1.2, 1.2, 1, 1], height_ratios=[1, 1, 0.075])
axes = [fig.add_subplot(gs[0:2, 0:2]),
        fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])]
cax = fig.add_subplot(gs[2, 1:3])

summary = []
for ax_i, (letter, (title, igbp)) in enumerate(zip('abcde', PANELS)):
    ax = axes[ax_i]
    res, sl = all_results[title], slope_frames[title]
    for cell, grp in res.groupby('_cell'):
        if cell not in sl.index:
            continue
        grp = grp.sort_values(XVAR)
        ax.plot(grp[XVAR], grp[YVAR], color=cmap(norm(sl.loc[cell, 'slope'])),
                lw=0.8, alpha=0.85, zorder=3)
    ax.axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.5, zorder=1)

    if thresholds is not None:
        row = thresholds.loc[title]
        point, lo, hi = row['measured'], row['ci_lower'], row['ci_upper']
    else:
        subset = df if igbp is None else df.loc[df['IGBP'] == igbp]
        point, lo, hi = measured_threshold(subset)
    model = MODEL_Z[title]
    summary.append({'panel': title, 'measured': point, 'ci_lower': lo, 'ci_upper': hi,
                    'model': model, 'cells': int(res['_cell'].nunique())})

    if np.isfinite(lo) and np.isfinite(hi):
        ax.axvspan(lo, hi, color='black', alpha=0.10, zorder=1)
    ax.axvline(point, color='black', lw=1.5, zorder=4)
    ax.axvline(model, color='#D55E00', lw=1.5, linestyle='--', zorder=4)

    n_sites = (df['SITE'].nunique() if igbp is None
               else df.loc[df['IGBP'] == igbp, 'SITE'].nunique())
    ax.set_title(f'{letter} | {title}', fontsize=AX_LABELS_FONTSIZE, loc='left',
                 fontweight='bold')
    ax.text(0.025, 0.025,
            f'measured {point:.2f} [{lo:.2f}, {hi:.2f}]' + chr(10)
            + f'model {model:.2f}' + chr(10)
            + f'{n_sites} sites, {res["_cell"].nunique()} cells',
            transform=ax.transAxes, fontsize=AX_LABELS_FONTSIZE * 0.62, va='bottom',
            color='#333333', linespacing=1.25)

    ax.set_xlim(-1.8, 3.2)
    ax.set_ylim(-1.15, 0.55)
    if ax_i == 0 or ax_i >= 3:
        ax.set_xlabel(r'VPD ($\sigma$)', fontsize=AX_LABELS_FONTSIZE * 0.9)
    else:
        ax.set_xticklabels([])
    if ax_i in (0, 1, 3):
        ax.set_ylabel(r'NEP ($\sigma$)', fontsize=AX_LABELS_FONTSIZE * 0.9)
    else:
        ax.set_yticklabels([])
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                  orientation='horizontal')
cb.set_label(r'$\Delta$NEP / $\Delta$VPD within a TA, SM and SW cell ($\sigma$/$\sigma$).'
             '   Blue: VPD goes with more uptake.   Red: VPD goes with less.',
             fontsize=AX_LABELS_FONTSIZE * 0.85)
cb.ax.tick_params(labelsize=AX_LABELS_FONTSIZE * 0.8)

fig.suptitle('PLANNED supplementary figure, not yet adopted   '
             f'(TA, SM and SW each in {N_CLASSES} classes, {N_CLASSES ** 3} cells, '
             f'{N_VPD_BINS} VPD bins per cell, measured values.' + chr(10) +
             'Black line and band: measured threshold with bootstrap interval over sites.'
             '   Orange line: model threshold for that group.)',
             fontsize=AX_LABELS_FONTSIZE * 0.8, color='#D55E00', fontweight='bold',
             x=0.01, ha='left')

fig.savefig(outfile, dpi=300, facecolor='white', bbox_inches='tight')
if not cached:
    pd.concat([r.assign(panel=k) for k, r in all_results.items()]).to_csv(
        cache_binned, index=False)
    pd.DataFrame(summary).to_csv(cache_thresholds, index=False)
print(pd.DataFrame(summary).round(3).to_string(index=False))
print(f"Saved to {outfile}")
if SHOW_PLOT:
    plt.show()
