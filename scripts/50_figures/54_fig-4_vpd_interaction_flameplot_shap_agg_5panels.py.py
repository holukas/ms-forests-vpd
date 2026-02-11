from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import string

import src.files as files
import src.plot as plot
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
AX_LABELS_FONTSIZE = 16

beautify = {
    "NEP_ZSCORE": "NEP", "ET_ZSCORE": "ET", "BIN_TA_ZSCORE": "Air temperature",
    "BIN_VPD_ZSCORE": "VPD", "BIN_SWC_ZSCORE": "Soil moisture",
    "SWC_ZSCORE_SHAPVALS": "Soil moisture", "VPD_ZSCORE_SHAPVALS": "VPD",
    "SWC_ZSCORE": "Soil moisture", "SWIN_ZSCORE": "SWIN",
    "ENF": "Evergreen needleleaf forests", "DBF": "Deciduous broadleaf forests",
    "MF": "Mixed forests", "EBF": "Evergreen broadleaf forests"
}

# --- FIGURE LAYOUT ---
# Update: 2 rows, 5 panels each + 1 colorbar on the right
fig = plt.figure(figsize=(22, 15), dpi=150, facecolor="white")
gs = mpl.gridspec.GridSpec(3, 6, width_ratios=[1, 1, 1, 1, 1, 0.1])

# Create 2D axes list: axes_grid[row][col]
axes_grid = [[fig.add_subplot(gs[r, c]) for c in range(5)] for r in range(3)]
cax = fig.add_subplot(gs[:, 5])  # Colorbar spans both rows

# --- MAIN NESTED LOOP ---
for row_idx, plotvars in enumerate(plotvars_rows):
    # Extract variables for this row
    FLUX = plotvars[0]
    xvar, yvar, zvar = plotvars[1], plotvars[2], plotvars[3]
    x_in_filename, y_in_filename = plotvars[4], plotvars[5]

    # Labels & Column logic
    xlabel = rf'{beautify[xvar]} ($\sigma$)'
    ylabel = rf'{beautify[yvar]} ($\sigma$)'
    zlabel = rf'{beautify[zvar]} effect ($\sigma$)'

    xagg = 'median' if str(xvar).startswith('BIN_') else aggfunc
    yagg = 'median' if str(xvar).startswith('BIN_') else aggfunc
    xcol, ycol, zcol = (f"{xvar}", xagg), (f"{yvar}", yagg), (f"{zvar}", aggfunc)
    count_vals_col = (f"{zvar}", "count")

    # Paths
    shap_type = 'conditional' if CONDITIONAL else 'standard'
    settings = files.read_settings_file("../../config/settings.yaml")
    dir_res = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type

    # --- PRE-CALCULATE SCALING FOR THIS ROW ---
    allsites_df, allsites_subset_df, _ = files.load_data(
        suffix='Sites', shap_type=shap_type, dir_res=dir_res, flux=FLUX,
        count_vals_col=count_vals_col, n_sites_min=n_sites_min,
        subsetcols=[xcol, ycol, zcol], site_filter=None,
        x_in_filename=x_in_filename, y_in_filename=y_in_filename, aggfunc=aggfunc
    )
    absmax = np.max([allsites_subset_df.iloc[:, 2].abs().min(), allsites_subset_df.iloc[:, 2].abs().max()])

    # --- INNER COLUMN LOOP ---
    panel_igbp = [None] + igbps
    for col_idx, ax in enumerate(axes_grid[row_idx]):
        igbp = panel_igbp[col_idx]

        # Load Data
        if igbp is None:
            df_to_plot = allsites_subset_df
            title_suffix = "All sites"
        else:
            _, df_to_plot, n_info = files.load_data(
                suffix=f"IGBP-{igbp}", shap_type=shap_type, dir_res=dir_res, flux=FLUX,
                count_vals_col=count_vals_col, n_sites_min=n_sites_min,
                subsetcols=[xcol, ycol, zcol], site_filter=allsites_df.index,
                x_in_filename=x_in_filename, y_in_filename=y_in_filename, aggfunc=aggfunc)
            title_suffix = f"{igbp} (n={n_info[1]})"

        # Plot
        plot.flameplot(df=df_to_plot, fig=fig, ax=ax, cmap=cmap, title=None,
                       show_colormap=False, vmin=-absmax, vmax=absmax,
                       xlabel=xlabel if row_idx == 1 else "",  # Only X-label on bottom row
                       ylabel=ylabel if col_idx == 0 else "")  # Only Y-label on first col

        # Headers for the very top row only
        if row_idx == 0:
            coltitle = beautify[igbp] if 1 <= col_idx <= 4 else "Global forests"
            trans = transforms.blended_transform_factory(ax.transAxes, fig.transFigure)
            fig.text(0.5, 0.99, coltitle, transform=trans,
                     fontsize=AX_LABELS_FONTSIZE * 1.2, ha='center', va='top', weight='normal')

        # Panel Annotation (a, b, c...)
        letter_idx = row_idx * 5 + col_idx
        letter = string.ascii_lowercase[letter_idx]
        ax.text(0.05, 1.05, f"({letter}) {title_suffix}", transform=ax.transAxes,
                size=AX_LABELS_FONTSIZE * 1.2, weight='bold', ha='left', va='bottom')

        # Visual Guides
        ax.axhline(0, c='k', ls='--', lw=1, zorder=99)
        ax.axvline(0, c='k', ls='--', lw=1, zorder=99)
        ax.set_aspect('equal')

        plot_markers(ax, df_to_plot, xvals=f'{xvar}_{xagg}', yvals=f'{yvar}_{yagg}',
                     zvals=f'{zvar}_{aggfunc}', flux_txt=beautify[FLUX], annotate=False,
                     ax_labels_fontsize=AX_LABELS_FONTSIZE, area_size=area_size)

        # Format Ticks
        plot.format(ax=ax, fontsize=AX_LABELS_FONTSIZE,
                    showyticklabels=(col_idx == 0),
                    showxticklabels=(row_idx == 1),
                    xtickdigits=0, ytickdigits=0,
                    showbottomspine=True, showleftspine=(col_idx == 0))

# --- COLORBAR (Unified for both rows) ---
# Note: This uses the absmax from the LAST row processed.
# If rows have different scales, you might need two colorbars.
norm = mpl.colors.Normalize(vmin=-absmax, vmax=absmax)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, cax=cax, extend='both')
cb.set_label(zlabel, size=AX_LABELS_FONTSIZE, labelpad=20)
cb.ax.tick_params(labelsize=AX_LABELS_FONTSIZE)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for the super-title
gs.update(wspace=0.15, hspace=0.3)
plt.show()