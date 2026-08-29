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
N_TA_CLASSES = 50
# Number of VPD bins within each TA class.
N_VPD_BINS = 20

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

fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

norm = mcolors.Normalize(vmin=results['z_bin_label'].min(), vmax=results['z_bin_label'].max())
cmap = plt.get_cmap('RdYlBu_r')

for label, grp in results.groupby('z_bin_label'):
    grp = grp.sort_values(XVAR)
    ax.plot(grp[XVAR], grp[YVAR], color=cmap(norm(label)), lw=1.1, alpha=0.85, zorder=3)

ax.axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.5, zorder=1)
ax.axvline(0, color='black', lw=0.8, linestyle=':', alpha=0.4, zorder=1)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=ax, pad=0.02)
cb.set_label(r'TA class median ($\sigma$)', fontsize=AX_LABELS_FONTSIZE)
cb.ax.tick_params(labelsize=AX_LABELS_FONTSIZE * 0.85)

ax.set_xlabel(r'VPD ($\sigma$)', fontsize=AX_LABELS_FONTSIZE)
ax.set_ylabel(r'NEP ($\sigma$)', fontsize=AX_LABELS_FONTSIZE)
ax.tick_params(labelsize=AX_LABELS_FONTSIZE)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)

n_records = int(results[f'{YVAR}_COUNTS'].sum())
ax.text(0.02, 0.03,
        f'{n_records:,} measured half-hours, {len(files)} sites' + chr(10) +
        f'median NEP in {N_VPD_BINS} VPD bins, {N_TA_CLASSES} TA classes',
        transform=ax.transAxes, fontsize=AX_LABELS_FONTSIZE * 0.8, va='bottom',
        color='#444444')

ax.set_title('PLANNED supplementary figure, not yet adopted',
             fontsize=AX_LABELS_FONTSIZE, loc='left', color='#D55E00', fontweight='bold')
fig.tight_layout()

outfile = outdir / (f'67_PLANNED-SUPPFIG_Boomerang_Measured_{YVAR}_vs_{XVAR}'
                    f'_by{N_TA_CLASSES}x{ZVAR}.png')
fig.savefig(outfile, dpi=300, facecolor='white', bbox_inches='tight')

results.to_csv(str(outfile).replace('.png', '_DATA.csv'))
print(f"Saved to {outfile}")
if SHOW_PLOT:
    plt.show()
