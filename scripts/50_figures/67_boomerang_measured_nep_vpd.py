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
N_TA_CLASSES = 100
# Number of VPD bins within each TA class. Two is the boomerang proper: within a class TA
# is nearly constant, so the step from the first bin to the second is almost purely a
# change in VPD, and each class contributes one segment. More bins give a curve per class
# and a turnover can be read off it, but the segment is no longer a clean VPD contrast.
N_VPD_BINS = 2

XVAR, YVAR, ZVAR = 'VPD_ZSCORE', 'NEP_ZSCORE', 'TA_ZSCORE'
AGG = 'median'
AX_LABELS_FONTSIZE = 12

shap_type = 'conditional' if CONDITIONAL else 'standard'
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

sba = StratifiedAnalysis(df=df, zvar=ZVAR, xvar=XVAR, yvar=YVAR,
                         n_bins_z=N_TA_CLASSES, n_bins_x=N_VPD_BINS, agg=AGG)
sba.calcbins()

# Drawn here rather than with diive's own plot, so the styling matches Figures 1 to 4.
results = sba.results.copy()
results['z_bin_label'] = results['z_bin_label'].astype(float)

# Two panels when each class is a single segment: the boomerang itself, and the slope of
# every segment against its temperature class, which is where the sign change is legible.
two_panel = N_VPD_BINS < 4
if two_panel:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), dpi=150,
                                  gridspec_kw={'width_ratios': [1.25, 1]})
else:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax2 = None

norm = mcolors.Normalize(vmin=results['z_bin_label'].min(), vmax=results['z_bin_label'].max())
cmap = plt.get_cmap('RdYlBu_r')

for label, grp in results.groupby('z_bin_label'):
    grp = grp.sort_values(XVAR)
    ax.plot(grp[XVAR], grp[YVAR], color=cmap(norm(label)), lw=1.1, alpha=0.85, zorder=3)

ax.axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.5, zorder=1)

# The model threshold, for comparison. Read from what script 58 wrote rather than typed in.
conv = pd.read_csv(outdir / '58_Threshold_zscore_to_kPa.csv').set_index('group')
model_z = float(conv.loc['ALL SITES', 'z'])

# With more than a few bins per class each class traces a curve and the flux maximum can
# be read off it. With two bins there is no curve to find a maximum in, only a segment, so
# the summary is the share of classes whose segment points downwards instead.
extra_info = []
if N_VPD_BINS >= 4:
    peaks = (results.loc[results.groupby('z_bin_label')[YVAR].idxmax()]
             .sort_values('z_bin_label'))
    turnover = peaks[XVAR].median()
    ax.scatter(peaks[XVAR], peaks[YVAR], s=9, facecolor='none', edgecolor='black',
               linewidths=0.6, alpha=0.7, zorder=4)
    ax.axvline(turnover, color='black', lw=1.4, zorder=2)
    ax.text(turnover, ax.get_ylim()[1], rf' turnover {turnover:.2f}$\sigma$', ha='left',
            va='top', fontsize=AX_LABELS_FONTSIZE * 0.85, fontweight='bold')
else:
    slopes = results.sort_values(['z_bin_label', XVAR]).groupby('z_bin_label').agg(
        dx=(XVAR, lambda v: v.iloc[-1] - v.iloc[0]),
        dy=(YVAR, lambda v: v.iloc[-1] - v.iloc[0]))
    slopes['slope'] = slopes['dy'] / slopes['dx']
    # Each class also sits at a characteristic VPD, so the sign change can be expressed as
    # a VPD as well as a temperature. That is the quantity comparable with the model
    # threshold, which is a VPD.
    slopes['vpd'] = (results.sort_values(['z_bin_label', XVAR])
                     .groupby('z_bin_label')[XVAR].mean())
    down = int((slopes['slope'] < 0).sum())
    extra_info.append(f'NEP falls with VPD in {down} of {len(slopes)} TA classes')

    # Second panel: the slope of each segment against its temperature class. The sign
    # change is the same result as the turnover in the left panel, read off directly.
    ta = slopes.index.astype(float)
    ax2.scatter(ta, slopes['slope'], s=22, c=[cmap(norm(v)) for v in ta],
                edgecolor='none', zorder=3)
    ax2.axhline(0, color='black', lw=0.9, linestyle='--', alpha=0.6, zorder=1)

    # Where the slope crosses zero, by linear interpolation between the two classes that
    # straddle it. Reported rather than fitted, since the crossing is what matters.
    sign = slopes['slope'].values
    cross = None
    for i in range(len(sign) - 1):
        if sign[i] > 0 >= sign[i + 1]:
            f = sign[i] / (sign[i] - sign[i + 1])
            cross = ta[i] + f * (ta[i + 1] - ta[i])
            break
    if cross is not None:
        vpd_at_cross = np.interp(cross, ta, slopes['vpd'].values)
        ax2.axvline(cross, color='black', lw=1.3, zorder=2)
        ax2.text(cross, ax2.get_ylim()[1],
                 rf' sign change at TA {cross:.2f}$\sigma$' + chr(10)
                 + rf' VPD {vpd_at_cross:.2f}$\sigma$ there',
                 ha='left', va='top', fontsize=AX_LABELS_FONTSIZE * 0.85,
                 fontweight='bold')
        extra_info.append(f'slope changes sign at TA {cross:.2f} sigma, '
                          f'VPD {vpd_at_cross:.2f} sigma')

    # The model threshold is a VPD, so it is placed on this panel through the temperature
    # class that sits at that VPD.
    ta_at_model = np.interp(model_z, slopes['vpd'].values, ta)
    ax2.axvline(ta_at_model, color='#D55E00', lw=1.3, linestyle='--', zorder=2)
    ax2.text(ta_at_model, ax2.get_ylim()[0],
             rf'model threshold VPD {model_z:.2f}$\sigma$ ', ha='right', va='bottom',
             rotation=90, fontsize=AX_LABELS_FONTSIZE * 0.85, color='#D55E00')

    # Top axis: the VPD each temperature class sits at, so the panel can be read in VPD.
    ax2top = ax2.secondary_xaxis(
        'top', functions=(lambda t: np.interp(t, ta, slopes['vpd'].values),
                          lambda v: np.interp(v, slopes['vpd'].values, ta)))
    ax2top.set_xlabel(r'VPD of that class ($\sigma$)', fontsize=AX_LABELS_FONTSIZE)
    ax2top.tick_params(labelsize=AX_LABELS_FONTSIZE * 0.9)

    ax2.set_xlabel(r'TA class median ($\sigma$)', fontsize=AX_LABELS_FONTSIZE)
    ax2.set_ylabel(r'$\Delta$NEP / $\Delta$VPD within class ($\sigma$/$\sigma$)',
                   fontsize=AX_LABELS_FONTSIZE)
    ax2.tick_params(labelsize=AX_LABELS_FONTSIZE)
    for sp in ('top', 'right'):
        ax2.spines[sp].set_visible(False)
    ax2.set_title('b', fontsize=AX_LABELS_FONTSIZE * 1.1, loc='left', fontweight='bold')

ax.axvline(model_z, color='#D55E00', lw=1.4, linestyle='--', zorder=2)
ax.text(model_z, ax.get_ylim()[0], rf'model threshold {model_z:.2f}$\sigma$ ', ha='right',
        va='bottom', rotation=90, fontsize=AX_LABELS_FONTSIZE * 0.85, color='#D55E00')

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=ax, pad=0.02)
cb.set_label(r'TA class median ($\sigma$)', fontsize=AX_LABELS_FONTSIZE)
cb.ax.tick_params(labelsize=AX_LABELS_FONTSIZE * 0.85)

ax.set_xlabel(r'VPD ($\sigma$)', fontsize=AX_LABELS_FONTSIZE)
ax.set_ylabel(r'NEP ($\sigma$)', fontsize=AX_LABELS_FONTSIZE)
ax.tick_params(labelsize=AX_LABELS_FONTSIZE)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)

counts = results[f'{YVAR}_COUNTS']
info_lines = [
    f'{int(counts.sum()):,} measured half-hours from {len(files)} sites',
    f'{N_TA_CLASSES} TA classes x {N_VPD_BINS} VPD bins = {len(results):,} points',
    f'median {int(counts.median()):,} values per point '
    f'(range {int(counts.min()):,} to {int(counts.max()):,})',
] + extra_info
ax.text(0.02, 0.03, chr(10).join(info_lines), transform=ax.transAxes,
        fontsize=AX_LABELS_FONTSIZE * 0.75, va='bottom', color='#444444')

ax.set_title(('a' if two_panel else '')
             + '   PLANNED supplementary figure, not yet adopted',
             fontsize=AX_LABELS_FONTSIZE, loc='left', color='#D55E00', fontweight='bold')
fig.tight_layout()

outfile = outdir / (f'67_PLANNED-SUPPFIG_Boomerang_Measured_{YVAR}_vs_{XVAR}'
                    f'_by{N_TA_CLASSES}x{ZVAR}.png')
fig.savefig(outfile, dpi=300, facecolor='white', bbox_inches='tight')

results.to_csv(str(outfile).replace('.png', '_DATA.csv'))
print(f"Saved to {outfile}")
if SHOW_PLOT:
    plt.show()
