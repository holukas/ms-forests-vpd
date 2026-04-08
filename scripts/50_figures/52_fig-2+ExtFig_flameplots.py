import string
from pathlib import Path

import diive as dv
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib import ticker

import src.files as files
import src.plot as plot

# --- SETTINGS ---
# Each inner list represents one row
# Order: explained flux, x-bins, y-bins, z-colors, x in filename, y in filename, colormap for row,
# show colormap for row (if False shows one overall colormap for all rows)

# # Figure 3
# # SHAP values heatmaps
# # 1: Physical drivers (atmosphere, drivers)
# # 2: Supply limitation (soil, constraints; supply vs. demand)
# # 3: Physiological response (plant, response)
# plotvars_rows = [
#     ['NEP_ZSCORE', 'BIN_TA_ZSCORE', 'BIN_VPD_ZSCORE', 'VPD_ZSCORE_SHAPVALS',
#      'BIN-TA_ZSCORE', 'BIN-VPD_ZSCORE', 'RdYlBu', False],
#     ['NEP_ZSCORE', 'BIN_SWC_ZSCORE', 'BIN_VPD_ZSCORE', 'VPD_ZSCORE_SHAPVALS',
#      'BIN-SWC_ZSCORE', 'BIN-VPD_ZSCORE', 'RdYlBu', False],
#     ['NEP_ZSCORE', 'BIN_ET_ZSCORE', 'BIN_VPD_ZSCORE', 'VPD_ZSCORE_SHAPVALS',
#      'BIN-ET_ZSCORE', 'BIN-VPD_ZSCORE', 'RdYlBu', False],
# ]
# figsize = (19, 13)
# figure_info = [2, f'ShapValues-{plotvars_rows[0][3]}']
# show_only_max_marker = False


# Figure 4
# Flux heatmaps
plotvars_rows = [
    ['NEP_ZSCORE', 'BIN_SWC_ZSCORE', 'BIN_VPD_ZSCORE', 'NEP_ZSCORE',
     'BIN-SWC_ZSCORE', 'BIN-VPD_ZSCORE', 'RdYlBu', True],
    ['NEP_ZSCORE', 'BIN_SWC_ZSCORE', 'BIN_VPD_ZSCORE', 'GPP_ZSCORE',
     'BIN-SWC_ZSCORE', 'BIN-VPD_ZSCORE', 'BrBG', True],
    ['NEP_ZSCORE', 'BIN_SWC_ZSCORE', 'BIN_VPD_ZSCORE', 'RECO_ZSCORE',
     'BIN-SWC_ZSCORE', 'BIN-VPD_ZSCORE', 'coolwarm', True],
    ['NEP_ZSCORE', 'BIN_SWC_ZSCORE', 'BIN_VPD_ZSCORE', 'ET_ZSCORE',
     'BIN-SWC_ZSCORE', 'BIN-VPD_ZSCORE', 'RdBu', True],
]
figsize = (19, 13 / 3 * 4)
figure_info = ["ExtFig2", f'Fluxes-{plotvars_rows[0][3]}']
show_only_max_marker = True

# Shared plotting constants
aggfunc, CONDITIONAL = 'mean', True
cb_digits, area_size = 1, 25
igbps = ['ENF', 'DBF', 'MF', 'EBF']
facecolor = 'white'
# facecolor = '#faf9f6'
AX_LABELS_FONTSIZE = 18

beautify = {
    "GPP_ZSCORE": "GPP",
    "RECO_ZSCORE": "RECO",
    "NEP_ZSCORE": "NEP",
    "ET_ZSCORE": "ET",
    "VPD_ZSCORE": "VPD",
    "BIN_ET_ZSCORE": "ET",  # Evapotranspiration
    "BIN_TA_ZSCORE": "TA",  # Air temperature
    "TA_ZSCORE_SHAPVALS": "TA",
    "BIN_VPD_ZSCORE": "VPD",
    "BIN_SWC_ZSCORE": "SM",
    "SWC_ZSCORE_SHAPVALS": "SM",
    "SWC_ZSCORE": "SM",  # Soil moisture
    "VPD_ZSCORE_SHAPVALS": "VPD",
    "SWIN_ZSCORE": "SWIN",
    "ENF": "Evergreen needleleaf forests",
    "DBF": "Deciduous broadleaf forests",
    "MF": "Mixed forests",
    "EBF": "Evergreen broadleaf forests"
}

# Figure layout
n_rows = len(plotvars_rows)
fig = plt.figure(figsize=figsize, dpi=150, facecolor="white")
gs = mpl.gridspec.GridSpec(n_rows, 6, width_ratios=[1, 1, 1, 1, 1, 0.1])

# Create 2D axes list: axes_grid[row][col]
axes_grid = [[fig.add_subplot(gs[r, c]) for c in range(5)] for r in range(n_rows)]
# cax = fig.add_subplot(gs[1, 5])  # Colorbar
show_row_colormap = False

# Main loop (rows)
for row_idx, plotvars in enumerate(plotvars_rows):
    # Extract variables for this row
    FLUX = plotvars[0]
    xvar, yvar, zvar = plotvars[1], plotvars[2], plotvars[3]
    x_in_filename, y_in_filename = plotvars[4], plotvars[5]
    cmap = plotvars[6]
    show_row_colormap = plotvars[7]

    # Labels & Column logic
    xlabel = rf'{beautify[xvar]} ($\sigma$)'
    ylabel = rf'{beautify[yvar]} ($\sigma$)'
    zlabel = rf'{beautify[zvar]} effect on NEP ($\sigma$)' if '_SHAPVALS' in zvar else rf'{beautify[zvar]} ($\sigma$)'

    # Bins always use 'median', b/c using 'mean' results in floating point errors
    xagg = 'median' if str(xvar).startswith('BIN_') else aggfunc
    yagg = 'median' if str(xvar).startswith('BIN_') else aggfunc
    xcol, ycol, zcol = (f"{xvar}", xagg), (f"{yvar}", yagg), (f"{zvar}", aggfunc)
    count_vals_col = (f"{zvar}", "count")

    # Paths
    shap_type = 'conditional' if CONDITIONAL else 'standard'
    settings = files.read_settings_file("../../config/settings.yaml")
    dir_res = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type

    # todo check Pre-calculate for scaling z-values (colors)
    allsites_df, allsites_subset_df, minmax_counts, n_sites = files.load_data(
        suffix='Sites', shap_type=shap_type, dir_res=dir_res, flux=FLUX,
        count_vals_col=count_vals_col,
        subsetcols=[xcol, ycol, zcol], site_filter=None,
        x_in_filename=x_in_filename, y_in_filename=y_in_filename, aggfunc=aggfunc
    )

    if show_row_colormap:
        absmax = np.max(allsites_subset_df.iloc[:, 2].abs())
        cax = fig.add_subplot(gs[row_idx, 5])
        plot.create_colormap(fig=fig, ax=cax, cmap=cmap, label=zlabel, absmax=absmax, labelsize=AX_LABELS_FONTSIZE)
    else:
        if row_idx == 0:
            absmax = np.max([allsites_subset_df.iloc[:, 2].abs().min(), allsites_subset_df.iloc[:, 2].abs().max()])
            cax = fig.add_subplot(gs[1, 5])

    # Inner loop (columns)
    panel_igbp = [None] + igbps
    for col_idx, ax in enumerate(axes_grid[row_idx]):
        igbp = panel_igbp[col_idx]

        # Load Data
        if igbp is None:
            df_to_plot = allsites_subset_df
        else:
            _, df_to_plot, minmax_counts, n_sites = files.load_data(
                suffix=f"IGBP-{igbp}", shap_type=shap_type, dir_res=dir_res, flux=FLUX,
                count_vals_col=count_vals_col,
                subsetcols=[xcol, ycol, zcol], site_filter=allsites_df.index,
                x_in_filename=x_in_filename, y_in_filename=y_in_filename, aggfunc=aggfunc)

        # Flameplot (heatmap)
        hm = dv.heatmapxyz(
            ax=ax, x=df_to_plot.iloc[:, 0], y=df_to_plot.iloc[:, 1], z=df_to_plot.iloc[:, 2],
            cmap=cmap, vmin=-absmax, vmax=absmax, color_bad=facecolor,
            show_colormap=False, show_grid=False)
        hm.plot()

        # Headers for the very top row only
        if row_idx == 0:
            coltitle = igbp if 1 <= col_idx <= 4 else "Global forests"
            trans = transforms.blended_transform_factory(ax.transAxes, fig.transFigure)
            fig.text(0.5, 0.99, coltitle, transform=trans,
                     fontsize=AX_LABELS_FONTSIZE * 1.2, ha='center', va='top', weight='bold')
            fig.text(0.5, 0.965, f"n={n_sites} (min. {minmax_counts[0]})", transform=trans,
                     fontsize=AX_LABELS_FONTSIZE, ha='center', va='top', weight='normal')

        # Panel letters
        letter_idx = row_idx * 5 + col_idx
        letter = string.ascii_lowercase[letter_idx]
        ax.text(0.05, 1.05, f"{letter}", transform=ax.transAxes, zorder=99,
                size=AX_LABELS_FONTSIZE * 1.2, weight='bold', ha='left', va='top')

        # Zero lines
        ax.axhline(0, c='k', ls='--', lw=1, zorder=99)
        ax.axvline(0, c='k', ls='--', lw=1, zorder=99)

        # Marker and annotations
        plot.plot_markers(ax, df_to_plot, xvals=f'{xvar}_{xagg}', yvals=f'{yvar}_{yagg}',
                          zvals=f'{zvar}_{aggfunc}', flux_txt=beautify[FLUX], annotate=False,
                          ax_labels_fontsize=AX_LABELS_FONTSIZE, area_size=area_size,
                          show_only_max_marker=show_only_max_marker)

        # Labels
        ax.set_xlabel(xlabel, fontsize=AX_LABELS_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=AX_LABELS_FONTSIZE)
        ax.yaxis.label.set_visible(col_idx == 0)

        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin * 1.1, xmax * 1.15)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin * 1.1, ymax * 1.1)

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
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
            ax.tick_params(axis='y', labelleft=True)
        else:
            ax.tick_params(axis='y', labelleft=False)  # Hide labels

        # Appearance
        ax.set_facecolor(facecolor)
        ax.set_aspect('equal')
        ax.grid(False)

if not show_row_colormap:
    plot.create_colormap(fig=fig, ax=cax, cmap=cmap, label=zlabel, absmax=absmax, labelsize=AX_LABELS_FONTSIZE)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for the super-title
gs.update(wspace=0.1, hspace=0.3)

# Save fig
FLUX = plotvars[0]
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type
outfilepath = dir_out / f'52_FIG-{figure_info[0]}_FlamePlots{figure_info[1]}_{FLUX}.png'
print(f"Saved to {outfilepath}")
plt.savefig(outfilepath, bbox_inches='tight', dpi=300)

plt.show()
