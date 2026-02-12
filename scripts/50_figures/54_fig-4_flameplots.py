import string
from pathlib import Path

import diive as dv
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib import ticker

import src.files as files
from src.plot import plot_markers

# --- SETTINGS ---
# Each inner list represents one row
plotvars_rows = [
    ['NEP_ZSCORE', 'BIN_TA_ZSCORE', 'BIN_VPD_ZSCORE', 'VPD_ZSCORE_SHAPVALS', 'BIN-TA_ZSCORE', 'BIN-VPD_ZSCORE'],
    ['NEP_ZSCORE', 'BIN_SWC_ZSCORE', 'BIN_VPD_ZSCORE', 'VPD_ZSCORE_SHAPVALS', 'BIN-SWC_ZSCORE', 'BIN-VPD_ZSCORE'],
    ['NEP_ZSCORE', 'BIN_TA_ZSCORE', 'BIN_SWC_ZSCORE', 'VPD_ZSCORE_SHAPVALS', 'BIN-TA_ZSCORE', 'BIN-SWC_ZSCORE'],
]

# Shared plotting constants
aggfunc, CONDITIONAL = 'mean', True
n_sites_min, cb_digits, area_size = 30, 1, 50
cmap, igbps = 'RdYlBu', ['ENF', 'DBF', 'MF', 'EBF']
AX_LABELS_FONTSIZE = 18

beautify = {
    "NEP_ZSCORE": "NEP",
    "ET_ZSCORE": "ET",
    "BIN_TA_ZSCORE": "Air temperature",
    "BIN_VPD_ZSCORE": "Vapor pressure deficit",
    "BIN_SWC_ZSCORE": "Soil moisture",
    "SWC_ZSCORE_SHAPVALS": "Soil moisture",
    "VPD_ZSCORE_SHAPVALS": "Vapor pressure deficit",
    "SWC_ZSCORE": "Soil moisture", "SWIN_ZSCORE": "SWIN",
    "ENF": "Evergreen needleleaf forests",
    "DBF": "Deciduous broadleaf forests",
    "MF": "Mixed forests",
    "EBF": "Evergreen broadleaf forests"
}

# Figure layout
fig = plt.figure(figsize=(22, 15), dpi=150, facecolor="white")
gs = mpl.gridspec.GridSpec(3, 6, width_ratios=[1, 1, 1, 1, 1, 0.1])

# Create 2D axes list: axes_grid[row][col]
axes_grid = [[fig.add_subplot(gs[r, c]) for c in range(5)] for r in range(3)]
cax = fig.add_subplot(gs[1, 5])  # Colorbar

# Main loop (rows)
for row_idx, plotvars in enumerate(plotvars_rows):
    # Extract variables for this row
    FLUX = plotvars[0]
    xvar, yvar, zvar = plotvars[1], plotvars[2], plotvars[3]
    x_in_filename, y_in_filename = plotvars[4], plotvars[5]

    # Labels & Column logic
    xlabel = rf'{beautify[xvar]} ($\sigma$)'
    ylabel = rf'{beautify[yvar]} ($\sigma$)'
    zlabel = rf'{beautify[zvar]} effect ($\sigma$)'

    # Bins always use 'median', b/c using 'mean' results in floating point errors
    xagg = 'median' if str(xvar).startswith('BIN_') else aggfunc
    yagg = 'median' if str(xvar).startswith('BIN_') else aggfunc
    xcol, ycol, zcol = (f"{xvar}", xagg), (f"{yvar}", yagg), (f"{zvar}", aggfunc)
    count_vals_col = (f"{zvar}", "count")

    # Paths
    shap_type = 'conditional' if CONDITIONAL else 'standard'
    settings = files.read_settings_file("../../config/settings.yaml")
    dir_res = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type

    # Pre-calculate for scaling z-values (colors)
    allsites_df, allsites_subset_df, _ = files.load_data(
        suffix='Sites', shap_type=shap_type, dir_res=dir_res, flux=FLUX,
        count_vals_col=count_vals_col, n_sites_min=n_sites_min,
        subsetcols=[xcol, ycol, zcol], site_filter=None,
        x_in_filename=x_in_filename, y_in_filename=y_in_filename, aggfunc=aggfunc
    )
    absmax = np.max([allsites_subset_df.iloc[:, 2].abs().min(), allsites_subset_df.iloc[:, 2].abs().max()])

    # Inner loop (columns)
    panel_igbp = [None] + igbps
    for col_idx, ax in enumerate(axes_grid[row_idx]):
        igbp = panel_igbp[col_idx]

        # Load Data
        if igbp is None:
            df_to_plot = allsites_subset_df
            # title_suffix = "All sites"
        else:
            _, df_to_plot, n_info = files.load_data(
                suffix=f"IGBP-{igbp}", shap_type=shap_type, dir_res=dir_res, flux=FLUX,
                count_vals_col=count_vals_col, n_sites_min=n_sites_min,
                subsetcols=[xcol, ycol, zcol], site_filter=allsites_df.index,
                x_in_filename=x_in_filename, y_in_filename=y_in_filename, aggfunc=aggfunc)
            # title_suffix = f"{igbp} (n={n_info[1]})"
            print(n_info)

        # Flameplot (heatmap)
        hm = dv.heatmapxyz(
            ax=ax, x=df_to_plot.iloc[:, 0], y=df_to_plot.iloc[:, 1], z=df_to_plot.iloc[:, 2],
            # xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
            cmap=cmap, vmin=-absmax, vmax=absmax, color_bad='white',
            show_colormap=False, show_grid=False)
        hm.plot()

        # Headers for the very top row only
        if row_idx == 0:
            coltitle = igbp if 1 <= col_idx <= 4 else "Global forests"
            trans = transforms.blended_transform_factory(ax.transAxes, fig.transFigure)
            fig.text(0.5, 0.99, coltitle, transform=trans,
                     fontsize=AX_LABELS_FONTSIZE * 1.2, ha='center', va='top', weight='bold')


        # Panel letters
        letter_idx = row_idx * 5 + col_idx
        letter = string.ascii_lowercase[letter_idx]
        ax.text(0.05, 1.05, f"{letter}", transform=ax.transAxes, zorder=99,
                size=AX_LABELS_FONTSIZE * 1.2, weight='bold', ha='left', va='top')

        # Zero lines
        ax.axhline(0, c='k', ls='--', lw=1, zorder=99)
        ax.axvline(0, c='k', ls='--', lw=1, zorder=99)

        # Marker and annotations
        plot_markers(ax, df_to_plot, xvals=f'{xvar}_{xagg}', yvals=f'{yvar}_{yagg}',
                     zvals=f'{zvar}_{aggfunc}', flux_txt=beautify[FLUX], annotate=False,
                     ax_labels_fontsize=AX_LABELS_FONTSIZE, area_size=area_size)

        # Labels
        ax.set_xlabel(xlabel, fontsize=AX_LABELS_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=AX_LABELS_FONTSIZE)
        ax.yaxis.label.set_visible(col_idx == 0)

        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1) if (col_idx == 0) else ax.spines['left'].set_visible(False)

        # Ticks
        # Show y-ticks only for the first column
        ax.tick_params(axis='y', which='major', width=1, length=5, labelsize=AX_LABELS_FONTSIZE, left=(col_idx == 0))
        ax.tick_params(axis='x', which='major', width=1, length=5, labelsize=AX_LABELS_FONTSIZE)

        # Ticklabels (x)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(f"{f'%.0f'}"))
        ax.tick_params(axis='x', labelbottom=True)

        # Ticklabels (y)
        if col_idx == 0:
            # Show y-labels only on the first column
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(f"{f'%.0f'}"))
            ax.tick_params(axis='y', labelleft=True)
        else:
            ax.tick_params(axis='y', labelleft=False)  # Hide labels

        # Appearance
        ax.set_aspect('equal')
        ax.grid(False)

# --- COLORBAR (Unified for all rows) ---
# Note: This uses the absmax from the LAST row processed.
# If rows have different scales, you might need two colorbars.
def cb_formatter(x, pos):
    """Custom format: 0 as '0', others as '0.1f'"""
    if np.isclose(x, 0, atol=1e-5):
        return "0"
    return f"{x:.1f}"

norm = mpl.colors.Normalize(vmin=-absmax, vmax=absmax)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, cax=cax, extend='both')
cb.ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(cb_formatter))
cb.set_label(zlabel, size=AX_LABELS_FONTSIZE, labelpad=20)
cb.ax.tick_params(labelsize=AX_LABELS_FONTSIZE)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for the super-title
gs.update(wspace=0.15, hspace=0.3)
plt.show()
