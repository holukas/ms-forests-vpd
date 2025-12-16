"""
Flame plot.
"""
from pathlib import Path

import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from diive.core.plotting.styles import LightTheme as theme

import src.files as files
import src.plot as plot
from src.common import findpoi

# Settings & variables
FLUX, xvar, yvar, zvar = 'NEP_ZSCORE', 'TA_ZSCORE', 'VPD_ZSCORE', 'VPD_ZSCORE'
aggfunc, CONDITIONAL = 'median', True
n_sites_min, n_sites_used, cb_digits, area_size = 1, 171, 1, 25  # TODO act min 30
cmap, igbps = 'RdYlBu', ['DBF', 'MF', 'EBF']  # TODO act ENF
igbps_n_sites = [87, 56, 14, 14]
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Latin Modern Roman'] + plt.rcParams['font.serif']

# Labels & Columns
xlabel, ylabel = f'{xvar} (z-score)', f'{yvar} (z-score)'
zlabel = f'Impact of {zvar} on {FLUX} (SHAP {aggfunc} z-score)'
cols = {k: (f"BIN_{k}", aggfunc) for k in [xvar, yvar]}
z_col, z_cnt = (f"{zvar}_SHAPVALS", aggfunc), (f"{zvar}_SHAPVALS", "count")

# Paths & Settings
shap_type = 'conditional' if CONDITIONAL else 'standard'
settings = files.read_settings_file("../../config/settings.yaml")
dir_res = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type


def load_data(suffix, site_filter=None):
    """
    Loads parquet, flattens cols, optionally filters by index.
    suffix: 'Sites' (for main plot, prefix 42) or 'IGBP-X' (for subplots, prefix 43)
    """
    # Select 42 for 'Sites' and 43 for IGBP
    prefix = "42" if suffix == 'Sites' else "43"
    filename = f"{prefix}_SHAPVALUES-{shap_type}_AggregatedAcross{suffix}_BIN-{xvar}+BIN-{yvar}+{FLUX}.parquet"
    fp = dir_res / filename
    df = dv.load_parquet(fp, sanitize_timestamp=False, output_middle_timestamp=False)

    # Filter to match the index of the main plot (for IGBP subplots)
    if site_filter is not None:
        df = df[df.index.isin(site_filter)].copy()

    # Apply count threshold (n_sites_min for main, 1 for IGBP)
    threshold = n_sites_min if suffix == 'Sites' else 1
    mask = df[z_cnt] >= threshold
    df = df[mask].copy()

    # Prepare subset for plotting
    sub = df[[cols[xvar], cols[yvar], z_col]].copy()
    sub.columns = ['_'.join(c).strip() for c in sub.columns]

    return df, sub, df[z_cnt].min()


def plot_markers(ax, df, annotate=False):
    """Finds min/max regions, plots markers, and optionally adds arrows."""
    piv = df.pivot(index=f'BIN_{xvar}_median', columns=f'BIN_{yvar}_median', values=f'{zvar}_SHAPVALS_median')
    locs = {m: findpoi(df=piv, k=area_size, agg='mean', what=m)[0] for m in ['max', 'min']}

    for m_type, (lx, ly) in locs.items():
        x, y = lx + 0.05, ly + 0.05
        marker = '+' if m_type == 'max' else '_'
        ax.scatter(x, y, c='k', marker=marker, lw=3, s=650, zorder=100, alpha=0.5)
        ax.scatter(x, y, facecolors='none', edgecolors='k', marker='o', lw=3, s=650, zorder=100, alpha=0.5)

        if annotate:
            txt, y_off = (f'highest {FLUX} increase', 1.5) if m_type == 'max' else (f'highest {FLUX} decrease', 0.6)
            tx_pos = (x - 3, y + y_off) if m_type == 'max' else (x - 1.1, y + y_off)
            ha = 'left' if m_type == 'max' else 'center'
            ax.annotate(txt, xy=(x, y), xytext=tx_pos, arrowprops=dict(arrowstyle="->", color='k', lw=3, shrinkB=15),
                        fontsize=16, color='black', ha=ha, va='center', zorder=100)
    print(f"Max: {locs['max']}, Min: {locs['min']}")


def style_ax(ax, title):
    ax.text(0.03 if 'All' in title else 0.1, 0.98 if 'All' in title else 1, title, transform=ax.transAxes,
            size=theme.AX_LABELS_FONTSIZE, ha='left', va='bottom' if 'All' in title else 'top',
            zorder=99, backgroundcolor='white')
    ax.axhline(0, c='k', ls='--', lw=1, zorder=100)
    ax.axvline(0, c='k', ls='--', lw=1, zorder=100)
    ax.set_aspect('equal')


# Plotting
fig = plt.figure(figsize=(20.7, 9), dpi=150, facecolor="white")
gs = gridspec.GridSpec(2, 4)
ax_all = fig.add_subplot(gs[0:2, 0:2])
axes_sub = [fig.add_subplot(gs[r, c], sharex=ax_all, sharey=ax_all) for r, c in [(0, 2), (0, 3), (1, 2), (1, 3)]]

# Main plot, plot across all sites ---
raw_all, df_all, n_min = load_data('Sites')
vmin, vmax = df_all.iloc[:, 2].min(), df_all.iloc[:, 2].max()
plot.flameplot(df=df_all, fig=fig, ax=ax_all, cmap=cmap, title=None, cb_digits_after_comma=cb_digits,
               xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, cb_extend='both')

# Main Layout & Annotations
ax_all.set_ylim(df_all.iloc[:, 1].min() * 1.15, df_all.iloc[:, 1].max() * 1.05)
ax_all.set_xlim(df_all.iloc[:, 0].min() * 1.15, df_all.iloc[:, 0].max() * 1.15)
style_ax(ax_all, f"(a) All sites (n={n_sites_used}, min. {n_min})")
t_params = dict(size=theme.AX_LABELS_FONTSIZE, color='k', zorder=100)
texts = [(2, 0.1, r"$\uparrow$ dry", 'left', 'bottom'), (2, -0.1, r"$\downarrow$ humid", 'left', 'top'),
         (-0.1, 3.5, r"$\leftarrow$ cool", 'right', 'center'), (0.1, 3.5, r"warm $\rightarrow$", 'left', 'center')]
for x, y, s, h, v in texts:
    ax_all.text(x, y, s, ha=h, va=v, **t_params)
plot_markers(ax_all, df_all, annotate=True)

# Subplots (IGBP) ---
configs = zip(axes_sub, igbps, igbps_n_sites, [" ", " ", xlabel, xlabel], [ylabel, " ", ylabel, " "],
              ['b', 'c', 'd', 'e'])
for ax, igbp, n_site, xl, yl, ltr in configs:
    # Filter using index from main dataset (keeplocs logic)
    _, df_sub, n_sub_min = load_data(f"IGBP-{igbp}", site_filter=raw_all.index)
    plot.flameplot(df=df_sub, fig=fig, ax=ax, cmap=cmap, title=None, show_colormap=False,
                   vmin=vmin, vmax=vmax, xlabel=xl, ylabel=yl)
    style_ax(ax, f"({ltr}) {igbp} (n={n_site}, min. {n_sub_min})")
    plot_markers(ax, df_sub, annotate=False)

fig.tight_layout()
gs.update(wspace=.2)
fig.show()
