"""
PLANNED SUPPLEMENTARY FIGURE, not yet adopted.

The measured counterpart to Figure 4. Everything else in the paper attributes NEP to its
drivers through the model, so a reader has to take the attribution on trust. This figure
uses the measured half-hourly values directly: NEP against VPD, stratified into classes of
air temperature, with no model in between.

It cannot separate drivers, which is the reason the SHAP analysis exists, so it supports
the model result rather than replacing it. Say so in the caption.

Method: diive's StratifiedAnalysis (called SortingBinsMethod in earlier versions). Records
are split into N_TA_CLASSES quantile classes of TA, then within each class into N_VPD_BINS
bins of VPD, and the median NEP of each bin is plotted.

N_TA_CLASSES is the interesting knob. Few classes give a readable figure. Many classes
make TA nearly constant within a class, which isolates the VPD response more sharply at
the cost of a legend nobody can read. Both are legitimate, so the value is set here rather
than hard-coded.

Reads the per-site files written by stage 31, which hold the measured values alongside the
SHAP values, so the records are exactly those the models saw.
"""
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from diive.analysis.decoupling import StratifiedAnalysis

from src.paths import data_path, load_settings

SHOW_PLOT = False

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""
SITE_SUBSET = ""

# Number of TA classes. Raising it holds TA more nearly constant within a class, which
# sharpens the VPD response. A legend listing every class stops being readable well before
# this value, so the classes are shown as a colour bar instead.
N_TA_CLASSES = 1000
# Number of VPD bins within each TA class. Two is the boomerang proper: within a class TA
# is nearly constant, so the step from the first bin to the second is almost purely a
# change in VPD, and each class contributes one segment. More bins give a curve per class
# and a turnover can be read off it, but the segment is no longer a clean VPD contrast.
N_VPD_BINS = 2
# Draw the slope panel. Off leaves the boomerang on its own.
SHOW_SLOPE_PANEL = False

# Recompute the binning and the bootstrap, or reuse what the last run wrote. The bootstrap
# rebuilds the binning 200 times per panel and takes minutes, while the layout takes
# seconds, so a layout change should not pay for the statistics again. Set to True after
# changing anything that affects the numbers: the class counts, the fit range, the
# bootstrap, or the input data.
RECOMPUTE = False

XVAR, YVAR, ZVAR = 'VPD_ZSCORE', 'NEP_ZSCORE', 'TA_ZSCORE'
AGG = 'median'
AX_LABELS_FONTSIZE = 12

shap_type = 'conditional' if CONDITIONAL else 'interventional'
settings = load_settings()
indir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type / VARIANT
outdir = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT / SITE_SUBSET
outdir.mkdir(parents=True, exist_ok=True)

# Pool every site. The z-scores are per site, so pooling them is the same operation the
# rest of the analysis performs.
files = sorted(indir.glob(f'*_shap-{shap_type}_{FLUX}.parquet'))
print(f"Reading {len(files)} site files from {indir}")
frames = []
for fp in files:
    site = fp.name.split('_shap-')[0]
    part = pd.read_parquet(fp, columns=[XVAR, YVAR, ZVAR])
    part['SITE'] = site
    frames.append(part)
df = pd.concat(frames, axis=0)
print(f"{len(df)} records from {df['SITE'].nunique()} sites")

# Forest type per site. The stage 31 files do not carry it, the stage 21 table does.
igbp_map = (pd.read_csv(data_path('data/outputs/20_subsets/'
                                  '21_SUBSETS_parquet_vars_stats_subsets.csv'))
            .set_index('SITE')['IGBP'])
df['IGBP'] = df['SITE'].map(igbp_map)

# One panel for all sites, then one per forest type, ordered by size.
PANELS = [('Global forests', None), ('ENF', 'ENF'), ('DBF', 'DBF'),
          ('EBF', 'EBF'), ('MF', 'MF')]

def bin_locally(data, n_z, n_x, aggfunc):
    """Same computation as diive's StratifiedAnalysis, without its 120 class limit.

    Quantile bins of z, then quantile bins of x inside each, then aggregate y. diive is
    used below the limit so the two routes stay comparable; this path only exists for
    class counts diive refuses.
    """
    work = data[[ZVAR, XVAR, YVAR]].dropna().copy()
    work['_z'] = pd.qcut(work[ZVAR], n_z, labels=False, duplicates='drop')
    work['_x'] = work.groupby('_z', observed=True)[XVAR].transform(
        lambda v: pd.qcut(v, n_x, labels=False, duplicates='drop'))
    grouped = work.groupby(['_z', '_x'], observed=True)
    out = grouped.agg(**{XVAR: (XVAR, aggfunc), YVAR: (YVAR, aggfunc),
                         f'{YVAR}_COUNTS': (YVAR, 'size')}).reset_index()
    out['z_bin_label'] = out['_z'].map(
        work.groupby('_z', observed=True)[ZVAR].agg(aggfunc))
    return out.drop(columns=['_z', '_x'])


def binned_for(data):
    """Bin one subset, through diive where it is willing and locally otherwise."""
    if N_TA_CLASSES <= 120:
        sba = StratifiedAnalysis(df=data, zvar=ZVAR, xvar=XVAR, yvar=YVAR,
                                 n_bins_z=N_TA_CLASSES, n_bins_x=N_VPD_BINS, agg=AGG)
        sba.calcbins()
        return sba.results
    return bin_locally(data, N_TA_CLASSES, N_VPD_BINS, AGG)


# Model thresholds per group, from what script 58 wrote. Each panel is compared with the
# threshold for its own forest type, which is what makes this the measured counterpart of
# the SHAP response curve rather than a loose analogue.
conv = pd.read_csv(outdir / '58_Threshold_zscore_to_kPa.csv').set_index('group')
MODEL_Z = {'Global forests': float(conv.loc['ALL SITES', 'z'])}
for g in ('ENF', 'DBF', 'EBF', 'MF'):
    MODEL_Z[g] = float(conv.loc[g, 'z'])

# The fit is restricted to the VPD range where the classes are dense. Beyond it a handful
# of extreme classes drag a polynomial around and the curve turns back upwards, which the
# data do not support.
FIT_RANGE = (-1.5, 2.5)
# Each replicate rebuilds the binning over a resampled network, which is far more work
# than resampling classes, so the count is lower. 200 is enough for a 95 % interval.
N_BOOT = 200


def segment_slopes(res):
    """One slope per TA class, with the VPD interval the slope was measured over."""
    res = res.copy()
    res['z_bin_label'] = res['z_bin_label'].astype(float)
    by_class = res.sort_values(['z_bin_label', XVAR]).groupby('z_bin_label')
    out = pd.DataFrame({
        'vpd_start': by_class[XVAR].first(),
        'vpd_end': by_class[XVAR].last(),
        'dy': by_class[YVAR].last() - by_class[YVAR].first(),
    })
    out['dx'] = out['vpd_end'] - out['vpd_start']
    out = out.loc[out['dx'] > 0]
    out['slope'] = out['dy'] / out['dx']
    out['vpd'] = (out['vpd_start'] + out['vpd_end']) / 2
    out['weight'] = 1.0 / out['dx']
    return out


def zero_crossing(x, y, w):
    """VPD where a width-weighted quadratic fit of slope against VPD reaches zero."""
    if len(x) < 5:
        return float('nan')
    coef = np.polyfit(x, y, 2, w=np.sqrt(w))
    roots = np.roots(coef)
    roots = np.real(roots[np.abs(np.imag(roots)) < 1e-9])
    inside = roots[(roots >= x.min()) & (roots <= x.max())]
    return float(min(inside, key=abs)) if len(inside) else float('nan')


def threshold_from(res):
    """Point estimate of the threshold from one binned result."""
    sl = segment_slopes(res)
    sl = sl.loc[sl['vpd'].between(*FIT_RANGE)]
    return zero_crossing(sl['vpd'].values, sl['slope'].values, sl['weight'].values)


def measured_threshold(data, rng_seed=42, n_boot=N_BOOT):
    """Threshold with a bootstrap interval, resampling sites rather than classes.

    Neighbouring temperature classes share sites and overlap in VPD, so they are nowhere
    near independent and a bootstrap over classes gives an interval far too narrow. The
    unit that can be resampled honestly is the site, so each replicate rebuilds the whole
    binning from a resampled network.
    """
    point = threshold_from(binned_for(data))

    # Row positions per site, computed once. Building a replicate is then one index
    # concatenation and one take, instead of concatenating 208 dataframe slices, which
    # is roughly three seconds of the four a replicate used to cost.
    codes, sites = pd.factorize(data['SITE'].values)
    order = np.argsort(codes, kind='stable')
    bounds = np.searchsorted(codes[order], np.arange(len(sites) + 1))
    positions = [order[bounds[i]:bounds[i + 1]] for i in range(len(sites))]

    rng_bs = np.random.default_rng(rng_seed)
    boot = []
    for _ in range(n_boot):
        pick = rng_bs.integers(0, len(sites), len(sites))
        # A site drawn twice contributes its rows twice, which is what a bootstrap over
        # sites means.
        rows = np.concatenate([positions[i] for i in pick])
        val = threshold_from(binned_for(data.take(rows)))
        if np.isfinite(val):
            boot.append(val)
    boot = np.asarray(boot)
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan))
    return point, lo, hi, len(boot)


outfile = outdir / (f'67_PLANNED-SUPPFIG_Boomerang_Measured_{YVAR}_vs_{XVAR}'
                    f'_by{N_TA_CLASSES}x{ZVAR}_perBiome.png')
cache_binned = Path(str(outfile).replace('.png', '_DATA.csv'))
cache_thresholds = Path(str(outfile).replace('.png', '_THRESHOLDS.csv'))
cached = (not RECOMPUTE) and cache_binned.exists() and cache_thresholds.exists()

all_results, all_slopes = {}, {}
if cached:
    print(f"reusing {cache_binned.name}, set RECOMPUTE = True to redo the numbers")
    stored = pd.read_csv(cache_binned)
    for title, _ in PANELS:
        all_results[title] = stored.loc[stored['panel'] == title].drop(columns='panel')
    thresholds = pd.read_csv(cache_thresholds).set_index('panel')
else:
    for title, igbp in PANELS:
        subset = df if igbp is None else df.loc[df['IGBP'] == igbp]
        print(f"binning {title}: {len(subset):,} records")
        all_results[title] = binned_for(subset)
    thresholds = None

# Colour by slope, not by temperature. The sign of the slope is the message: blue where
# rising VPD goes with rising NEP, red where it goes with falling NEP.
slope_frames = {t: segment_slopes(r) for t, r in all_results.items()}
_all_slopes = np.concatenate([sf['slope'].values for sf in slope_frames.values()])
# The two sides of zero have very different ranges: slopes reach about +1.5 but only
# about -0.5. A symmetric scale would leave the whole suppression arm pale, so each side
# is scaled to its own range and zero stays the centre colour.
s_lo = np.nanpercentile(_all_slopes[_all_slopes < 0], 2)
s_hi = np.nanpercentile(_all_slopes[_all_slopes > 0], 98)
norm = mcolors.TwoSlopeNorm(vmin=s_lo, vcenter=0.0, vmax=s_hi)
# Red through yellow to blue, with yellow at zero slope. A white centre disappears
# against the panel, and zero slope is what the threshold is defined by, so it should not
# be the least visible colour in the figure.
cmap = mcolors.LinearSegmentedColormap.from_list(
    'red_yellow_blue', ['#67001f', '#d6604d', '#f4a582', '#ffe08a', '#ffd23f',
                        '#ffe08a', '#92c5de', '#4393c3', '#053061'])

# Panel a takes the left two thirds and the full height, the four forest types stack in
# two rows on the right. Panel a is then much the largest, the four small panels share
# their axes with each other, and the colour bar gets a strip of its own under everything.
fig = plt.figure(figsize=(13.5, 7.4), dpi=150, constrained_layout=True)
fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.03, hspace=0.04)
gs = fig.add_gridspec(3, 4, width_ratios=[1.2, 1.2, 1, 1],
                      height_ratios=[1, 1, 0.075])
ax_main = fig.add_subplot(gs[0:2, 0:2])
ax_small = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]),
            fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])]
axes = [ax_main] + ax_small
cax = fig.add_subplot(gs[2, 1:3])

summary = []
for ax_i, (letter, (title, igbp)) in enumerate(zip('abcde', PANELS)):
    ax = axes[ax_i]
    res = all_results[title].copy()
    res['z_bin_label'] = res['z_bin_label'].astype(float)
    sl = slope_frames[title]

    for label, grp in res.groupby('z_bin_label'):
        if label not in sl.index:
            continue
        grp = grp.sort_values(XVAR)
        ax.plot(grp[XVAR], grp[YVAR], color=cmap(norm(sl.loc[label, 'slope'])),
                lw=0.8, alpha=0.85, zorder=3)

    ax.axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.5, zorder=1)

    if thresholds is not None:
        row = thresholds.loc[title]
        point, lo, hi = row['measured'], row['ci_lower'], row['ci_upper']
    else:
        subset_df = df if igbp is None else df.loc[df['IGBP'] == igbp]
        point, lo, hi, _ = measured_threshold(subset_df)
    model = MODEL_Z[title]
    summary.append({'panel': title, 'measured': point, 'ci_lower': lo, 'ci_upper': hi,
                    'model': model})

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
            + f'{n_sites} sites, {int(res[f"{YVAR}_COUNTS"].sum()):,} half-hours',
            transform=ax.transAxes, fontsize=AX_LABELS_FONTSIZE * 0.62, va='bottom',
            color='#333333', linespacing=1.25)

    # The classes run past VPD 4 but only a handful of segments reach there, so the axis
    # stops where the data are still dense and the panels stay comparable.
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
    ax.tick_params(labelsize=AX_LABELS_FONTSIZE * 0.85)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                  orientation='horizontal')
cb.set_label(r'$\Delta$NEP / $\Delta$VPD within a TA class ($\sigma$/$\sigma$).   '
             r'Blue: VPD goes with more uptake.   Red: VPD goes with less.',
             fontsize=AX_LABELS_FONTSIZE * 0.85)
cb.ax.tick_params(labelsize=AX_LABELS_FONTSIZE * 0.8)

fig.suptitle('PLANNED supplementary figure, not yet adopted   '
             f'({N_TA_CLASSES} TA classes x {N_VPD_BINS} VPD bins, measured values.'
             + chr(10) +
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
