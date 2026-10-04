"""
Binned grid figures of SHAP effects and fluxes, three display items from one switch.

FIGURE at the top of the file selects the output:
- `"fig2"`: Figure 2, the VPD effect on NEP over the TA, SM and ET grids
- `"driver-effects"`: Supplementary Fig. 1, the TA and SM effects on the same grids
- `"fluxes"` (default): Supplementary Fig. 7, NEP, GPP, RECO and ET over the SM by VPD grid

Reads the stage 42 aggregation. VARIANT and SITE_SUBSET select a sensitivity run.
Writes the figure and `<figure name>_DATA.csv`, one row per grid cell and panel.
"""
import string
from pathlib import Path

import diive as dv
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from matplotlib import ticker

import src.files as files
import src.plot as plot
from src.paths import load_settings

# Open the figure in a window after saving. False by default so a script can run
# unattended: matplotlib picks the interactive TkAgg backend here, and plt.show()
# then blocks until the window is closed by hand.
SHOW_PLOT = False

# --- SETTINGS ---
# Run variant. An empty string reads the results of the main analysis and
# writes to the baseline plot folder. Any other value reads the matching variant
# folder and writes the figures next to it, so a sensitivity run cannot overwrite a
# figure of the main analysis. The aggregation must have run with the same value.
VARIANT = ""
# Site subset. An empty string reads the aggregation over every site.
# "deeper-only" reads the run restricted to the 128 sites whose soil water comes
# from below layer 1, and writes the figures next to it. The value has to match
# the one the aggregation ran with.
SITE_SUBSET = ""

# Which figure to draw. The three share every line of plotting code below and
# differ only in what colors the grid, so one script draws all of them.
#   "fig2"            Figure 2: the VPD effect on NEP, mapped on TA, SM and ET.
#   "fluxes"          Supplementary Fig. 7: the fluxes themselves on the SM by VPD grid.
#   "driver-effects"  Supplementary Fig. 1: the TA effect on the TA by VPD grid and the
#                     SM effect on the SM by VPD grid. The same SHAP campaign that
#                     gave the VPD column gave these, so nothing is recomputed.
FIGURE = "fluxes"

# Each inner list is one row of panels.
# Order: explained flux, x-bins, y-bins, z-colors, x in filename, y in filename,
# colormap for row, show colormap for row (False shares one colormap over all rows).
FIGURES = {
    # Rows: 1 physical drivers (atmosphere), 2 supply limitation (soil),
    # 3 physiological response (plant).
    "fig2": dict(
        plotvars_rows=[
            ['NEP_ZSCORE', 'BIN_TA_ZSCORE', 'BIN_VPD_ZSCORE', 'VPD_ZSCORE_SHAPVALS',
             'BIN-TA_ZSCORE', 'BIN-VPD_ZSCORE', 'RdYlBu', False],
            ['NEP_ZSCORE', 'BIN_SWC_ZSCORE', 'BIN_VPD_ZSCORE', 'VPD_ZSCORE_SHAPVALS',
             'BIN-SWC_ZSCORE', 'BIN-VPD_ZSCORE', 'RdYlBu', False],
            ['NEP_ZSCORE', 'BIN_ET_ZSCORE', 'BIN_VPD_ZSCORE', 'VPD_ZSCORE_SHAPVALS',
             'BIN-ET_ZSCORE', 'BIN-VPD_ZSCORE', 'RdYlBu', False],
        ],
        figsize=(19, 13),
        figure_info=['FIG-2', 'ShapValues-VPD_ZSCORE_SHAPVALS'],
        show_only_max_marker=False,
    ),
    "fluxes": dict(
        plotvars_rows=[
            ['NEP_ZSCORE', 'BIN_SWC_ZSCORE', 'BIN_VPD_ZSCORE', 'NEP_ZSCORE',
             'BIN-SWC_ZSCORE', 'BIN-VPD_ZSCORE', 'RdYlBu', True],
            ['NEP_ZSCORE', 'BIN_SWC_ZSCORE', 'BIN_VPD_ZSCORE', 'GPP_ZSCORE',
             'BIN-SWC_ZSCORE', 'BIN-VPD_ZSCORE', 'BrBG', True],
            ['NEP_ZSCORE', 'BIN_SWC_ZSCORE', 'BIN_VPD_ZSCORE', 'RECO_ZSCORE',
             'BIN-SWC_ZSCORE', 'BIN-VPD_ZSCORE', 'coolwarm', True],
            ['NEP_ZSCORE', 'BIN_SWC_ZSCORE', 'BIN_VPD_ZSCORE', 'ET_ZSCORE',
             'BIN-SWC_ZSCORE', 'BIN-VPD_ZSCORE', 'RdBu', True],
        ],
        figsize=(19, 13 / 3 * 4),
        figure_info=["SUPPFIG-7", 'Fluxes-NEP_ZSCORE'],
        show_only_max_marker=True,
        # One tick step for the four colorbars, so that the fluxes read on the same scale.
        # Chosen by range, the NEP bar (about 0.7) would get 0.2 and the others 0.5.
        colorbar_step=0.5,
    ),
    # Same colormap as Figure 2, so blue is a positive effect on NEP in both. Each
    # row gets its own colorbar because the two effects differ in size.
    "driver-effects": dict(
        plotvars_rows=[
            ['NEP_ZSCORE', 'BIN_TA_ZSCORE', 'BIN_VPD_ZSCORE', 'TA_ZSCORE_SHAPVALS',
             'BIN-TA_ZSCORE', 'BIN-VPD_ZSCORE', 'RdYlBu', True],
            ['NEP_ZSCORE', 'BIN_SWC_ZSCORE', 'BIN_VPD_ZSCORE', 'SWC_ZSCORE_SHAPVALS',
             'BIN-SWC_ZSCORE', 'BIN-VPD_ZSCORE', 'RdYlBu', True],
        ],
        figsize=(19, 13 / 3 * 2),
        figure_info=["SUPPFIG-1", 'DriverEffects-TA+SM'],
        show_only_max_marker=False,
    ),
}

plotvars_rows = FIGURES[FIGURE]["plotvars_rows"]
figsize = FIGURES[FIGURE]["figsize"]
figure_info = FIGURES[FIGURE]["figure_info"]
show_only_max_marker = FIGURES[FIGURE]["show_only_max_marker"]
colorbar_step = FIGURES[FIGURE].get("colorbar_step")   # None: chosen per colorbar by range

# Header geometry, in figure fractions. The gap between the two header lines and
# the room the panels leave for them are set in inches and divided by the figure
# height, so a two-row figure gets the same clearance as the three-row one. Fixed
# fractions put the site count on top of the column title in the short figure.
HEADER_TOP = 0.99
HEADER_GAP = 0.40 / figsize[1]
HEADER_ROOM = 0.75 / figsize[1]

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
plotted = []   # the values of every panel, for the data file
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
    zlabel = rf'{beautify[zvar]} effect on daytime NEP ($\sigma$)' if '_SHAPVALS' in zvar else rf'{beautify[zvar]} ($\sigma$)'

    # Bin coordinates are read as the median, which is exact; a mean of equal values
    # carries floating point noise into the labels. Both axes are always bin columns here.
    xagg = 'median' if str(xvar).startswith('BIN_') else aggfunc
    yagg = 'median' if str(yvar).startswith('BIN_') else aggfunc
    xcol, ycol, zcol = (f"{xvar}", xagg), (f"{yvar}", yagg), (f"{zvar}", aggfunc)
    count_vals_col = (f"{zvar}", "count")

    # Paths
    shap_type = 'conditional' if CONDITIONAL else 'interventional'
    settings = load_settings()
    dir_res = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT / SITE_SUBSET

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
        plot.create_colormap(fig=fig, ax=cax, cmap=cmap, label=zlabel, absmax=absmax, labelsize=AX_LABELS_FONTSIZE,
                             step=colorbar_step)
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
        hm = dv.plotting.HeatmapXYZ(
            x=df_to_plot.iloc[:, 0], y=df_to_plot.iloc[:, 1], z=df_to_plot.iloc[:, 2])
        hm.plot(
            ax=ax, format_style=plot.heatmap_style(show_grid=False),
            cmap=cmap, vmin=-absmax, vmax=absmax, color_bad=facecolor,
            show_colormap=False)

        # Headers for the very top row only
        if row_idx == 0:
            coltitle = igbp if 1 <= col_idx <= 4 else "All sites"
            trans = transforms.blended_transform_factory(ax.transAxes, fig.transFigure)
            fig.text(0.5, HEADER_TOP, coltitle, transform=trans,
                     fontsize=AX_LABELS_FONTSIZE * 1.2, ha='center', va='top', weight='bold')
            fig.text(0.5, HEADER_TOP - HEADER_GAP, f"n={n_sites} (min. {minmax_counts[0]})", transform=trans,
                     fontsize=AX_LABELS_FONTSIZE, ha='center', va='top', weight='normal')

        # Panel letters
        letter_idx = row_idx * 5 + col_idx
        letter = string.ascii_lowercase[letter_idx]
        _p = df_to_plot.iloc[:, :3].copy()
        _p.columns = ['x', 'y', 'z']
        _p.insert(0, 'panel', letter)
        _p.insert(1, 'group', igbp or 'All sites')
        _p.insert(2, 'x_var', xvar)
        _p.insert(3, 'y_var', yvar)
        _p.insert(4, 'z_var', f'{zvar}_{aggfunc}')
        plotted.append(_p)
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

plt.tight_layout(rect=[0, 0, 1, 1 - HEADER_ROOM])  # Leave room for the column headers
gs.update(wspace=0.1, hspace=0.3)

# Save fig
FLUX = plotvars[0]

dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT / SITE_SUBSET
dir_out.mkdir(parents=True, exist_ok=True)
outfilepath = dir_out / f'52_{figure_info[0]}_FlamePlots{figure_info[1]}_{FLUX}.png'
print(f"Saved to {outfilepath}")
plt.savefig(outfilepath, bbox_inches='tight', dpi=300)
# The plotted values, one row per grid cell and panel.
pd.concat(plotted).to_csv(outfilepath.with_name(outfilepath.stem + '_DATA.csv'), index=False)

if SHOW_PLOT:

    plt.show()