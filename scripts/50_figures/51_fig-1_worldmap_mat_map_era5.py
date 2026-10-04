"""
Figure 1: site map and the climate space of the sites.

Reads `17_datasets_info_parquet_vars_stats_usedsites_era5.csv` for the ERA5 mean annual
temperature and precipitation and the stage 21 subsets table for the site counts.
Writes `51_FIG-1_WorldMap_MAT_MAP_ERA5.png` and `_DATA.csv` with site, forest type, position,
climate and the source of each MAT and MAP value.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from src.paths import data_path, repo_path

# Open the figure in a window after saving. False by default so a script can run
# unattended: matplotlib picks the interactive TkAgg backend here, and plt.show()
# then blocks until the window is closed by hand.
SHOW_PLOT = False

AX_LABELS_FONTSIZE = 15

# ---------------------------------------------------------
# 1. DATA PREPARATION
# ---------------------------------------------------------
infile = data_path("data/outputs/10_datasets/17_datasets_info_parquet_vars_stats_usedsites_era5.csv")
datasets_df = pd.read_csv(infile)

datasets_df = datasets_df.loc[datasets_df['SWC_AVG'] != '-MISSING-'].copy()
datasets_df = datasets_df.loc[datasets_df['IGBP'] != 'DNF'].copy()

# Special case: site US-xBN was not used in analyses b/c
# no soil water data in months of interest (i.e., it has
# soil water during other time periods and therefore was
# not kicked out before)
datasets_df = datasets_df.loc[datasets_df['SITE'] != 'US-xBN'].copy()

n_sites_total = len(datasets_df)

# Sites and record length per forest type, taken from the subsets that entered the models
# so the counts match the analysis rather than the raw site collection.
subsets_df = pd.read_csv(data_path("data/outputs/20_subsets/21_SUBSETS_parquet_vars_stats_subsets.csv"))
counts = subsets_df.groupby('IGBP').agg(n_sites=('SITE', 'size'), n_years=('N_YEARS', 'sum'))
n_sites_counted = counts['n_sites'].sum()
n_years_counted = counts['n_years'].sum()
IGBP_ORDER = ['ENF', 'DBF', 'MF', 'EBF']  # the order used in all other figures and tables

valid_df = datasets_df.dropna(subset=['ERA5_MAT_1991_2020', 'ERA5_MAP_1991_2020']).copy()
n_sites_climate = len(valid_df)

shapefile_path = str(repo_path("data/worldmap/ne_10m_admin_0_countries.shp"))
world = gpd.read_file(shapefile_path)

gdf = gpd.GeoDataFrame(
    datasets_df,
    geometry=gpd.points_from_xy(datasets_df.LON, datasets_df.LAT),
    crs="EPSG:4326"
)

# ---------------------------------------------------------
# 2. DESIGN & STYLING (Nature Standard)
# ---------------------------------------------------------
# Force sans-serif font (Arial/Helvetica)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Map background colors
OCEAN_COL = '#d4e8f5'
LAND_COL = '#f0ede8'
BORDER_COL = '#b8b4ad'

# Okabe-Ito colorblind-safe palette
igbp_markers = {
    'ENF': {'marker': '^', 'markersize': 130, 'color': '#009E73', 'edgecolor': '#005f45'},
    'DBF': {'marker': 'o', 'markersize': 110, 'color': '#D55E00', 'edgecolor': '#803800'},
    'MF': {'marker': 'v', 'markersize': 130, 'color': '#E69F00', 'edgecolor': '#8a5f00'},
    'EBF': {'marker': 's', 'markersize': 110, 'color': '#56B4E9', 'edgecolor': '#336c8b'},
    'DNF': {'marker': 'X', 'markersize': 130, 'color': '#CC79A7', 'edgecolor': '#7a4864'},
}

labels = dict(
    ENF='ENF (', DBF='DBF (', MF='MF (', EBF='EBF (', DNF='DNF ('
)

# Isolation Calculation
mat_z = (valid_df['ERA5_MAT_1991_2020'] - valid_df['ERA5_MAT_1991_2020'].mean()) / valid_df['ERA5_MAT_1991_2020'].std()
map_z = (valid_df['ERA5_MAP_1991_2020'] - valid_df['ERA5_MAP_1991_2020'].mean()) / valid_df['ERA5_MAP_1991_2020'].std()

coords = np.column_stack((mat_z, map_z))
dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
np.fill_diagonal(dist_matrix, np.inf)

valid_df['Isolation_Score'] = dist_matrix.min(axis=1)
# isolated_sites = valid_df.nlargest(3, 'Isolation_Score')

# ---------------------------------------------------------
# 3. SET UP GRIDSPEC LAYOUT
# ---------------------------------------------------------
# The map keeps its geographic aspect ratio, so its height follows from the panel width
# and the figure height has to match, otherwise the map floats in white space. The legend
# gets its own row rather than being pushed below the axis with a negative offset, and the
# climate panel spans both rows so it keeps the full height.
fig = plt.figure(figsize=(17, 6.0), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1])

ax_world = fig.add_subplot(gs[0, 0])
ax_climate = fig.add_subplot(gs[0, 1])

# The map holds a geographic aspect ratio, so it is shrunk inside its cell. Anchoring both
# panels to the top of their cell puts the two axes tops, and with them the two panel
# letters, on the same line.
ax_world.set_anchor('N')
ax_climate.set_anchor('N')


# ---------------------------------------------------------
# 4. HELPER FUNCTION TO PLOT MAPS & ADD PANEL LETTERS
# ---------------------------------------------------------
def panel_label(ax, letter, title, gap_points=7, title_gap_points=16):
    """Put the panel letter and its title above the top left corner of the axes.

    Offsets are in points, so every panel gets the same gap whatever its height.
    """
    ax.annotate(letter, xy=(0, 1), xycoords='axes fraction',
                xytext=(0, gap_points), textcoords='offset points',
                fontsize=AX_LABELS_FONTSIZE * 1.2, fontweight='bold',
                va='bottom', ha='left', color='#000000', annotation_clip=False)
    ax.annotate(title, xy=(0, 1), xycoords='axes fraction',
                xytext=(title_gap_points, gap_points), textcoords='offset points',
                fontsize=AX_LABELS_FONTSIZE * 1.2,
                va='bottom', ha='left', color='#555555', annotation_clip=False)



def plot_map_region(ax, extent=None, title="", letter="", is_main_map=False):
    world.plot(ax=ax, color=LAND_COL, edgecolor=BORDER_COL, linewidth=0.6)

    for igbp_class, styles in igbp_markers.items():
        subset = gdf[gdf['IGBP'] == igbp_class]
        if not subset.empty:
            label = f"{labels[igbp_class]}n={len(subset)})" if is_main_map else None
            msize = styles['markersize'] * 0.7 if is_main_map else styles['markersize'] * 1.1

            subset.plot(ax=ax, marker=styles['marker'], color=styles['color'],
                        edgecolor=styles['edgecolor'], linewidth=0.8, markersize=msize,
                        alpha=0.9, label=label, zorder=3)

    ax.set_facecolor(OCEAN_COL)

    # Restore spines (borders)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)

    # Add Ticks and Axis Labels
    ax.tick_params(axis='both', which='major', labelsize=AX_LABELS_FONTSIZE, colors='#444444')
    ax.set_xlabel('Longitude (°)', fontsize=AX_LABELS_FONTSIZE, color='#222222')
    ax.set_ylabel('Latitude (°)', fontsize=AX_LABELS_FONTSIZE, color='#222222')

    # Reduce clutter by limiting the maximum number of ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    if extent:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    # Panel letter and title, placed a fixed number of points above the axes rather than a
    # fraction of its height. The two panels have different heights, so a fraction would put
    # the letters at different heights on the page.
    panel_label(ax, letter, title)


# ---------------------------------------------------------
# 5. PLOT THE MAPS (Left Panels)
# ---------------------------------------------------------
# Global map adjusted to align with the aspect ratio of the subpanels below
plot_map_region(ax_world, extent=[-180, 180, -60, 85], title=f'Global forest sites in this study (n={n_sites_total})',
                letter='a', is_main_map=True)

# Legend under the map. Short labels only: the site-year counts and the percentages go in
# the figure caption instead.
# Explicit handles, so the legend follows IGBP_ORDER and not the drawing order.
legend_handles = [
    Line2D([], [], linestyle='none', marker=igbp_markers[igbp]['marker'],
           markerfacecolor=igbp_markers[igbp]['color'],
           markeredgecolor=igbp_markers[igbp]['edgecolor'], markeredgewidth=0.8,
           markersize=11,
           label=f"{igbp} (n={counts.loc[igbp, 'n_sites']})")
    for igbp in IGBP_ORDER
]
# Anchored to the map rather than given its own row. The map is aspect-locked, so its cell
# is taller than the map itself, and the leftover height should sit under the legend as an
# ordinary margin rather than as a hole between the map and the legend.
ax_world.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=4,
                fontsize=AX_LABELS_FONTSIZE, frameon=False, columnspacing=2.5,
                handletextpad=0.6)

# ---------------------------------------------------------
# 6. PLOTTING THE CLIMATE SPACE (Right Panel)
# ---------------------------------------------------------
ax_climate.grid(True, color='#F0F0F0', linestyle='-', linewidth=1, zorder=0)
ax_climate.spines['top'].set_visible(False)
ax_climate.spines['right'].set_visible(False)
ax_climate.spines['left'].set_color('#888888')
ax_climate.spines['bottom'].set_color('#888888')
ax_climate.tick_params(colors='#555555', labelsize=AX_LABELS_FONTSIZE)

for igbp_class, group_df in valid_df.groupby('IGBP'):
    if igbp_class in igbp_markers:
        style = igbp_markers[igbp_class]
        ax_climate.scatter(
            group_df['ERA5_MAT_1991_2020'], group_df['ERA5_MAP_1991_2020'],
            marker=style['marker'], s=style['markersize'] * 1.2,
            c=style['color'], edgecolors=style['edgecolor'],
            linewidths=1, alpha=0.85, zorder=3
        )

# # Annotation for the most isolated sites
# import matplotlib.patheffects as pe
# for _, row in isolated_sites.iterrows():
#     ax_climate.annotate(
#         row['SITE'], (row['ERA5_MAT_1991_2020'], row['ERA5_MAP_1991_2020']),
#         xytext=(7, 7), textcoords='offset points', fontsize=AX_LABELS_FONTSIZE, fontweight='bold', color='#222222',
#         path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=5
#     )

ax_climate.set_xlabel('MAT (°C)', fontsize=AX_LABELS_FONTSIZE, color='#222222')
ax_climate.set_ylabel('MAP (mm)', fontsize=AX_LABELS_FONTSIZE, color='#222222')
ax_climate.set_facecolor('white')

panel_label(ax_climate, 'b', f'Bioclimatic distribution (n={n_sites_climate})')

# ---------------------------------------------------------
# 7. SAVE AND SHOW
# ---------------------------------------------------------
out_plot = data_path("data/outputs/50_plots/NEP_ZSCORE/conditional/51_FIG-1_WorldMap_MAT_MAP_ERA5.png")
out_plot.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(out_plot, dpi=300, bbox_inches='tight', facecolor='white')
# The plotted values: position, forest type and climate of every site, with the source
# of each MAT and MAP value and the reason where it is not the flux product (script 17).
datasets_df[['SITE', 'IGBP', 'LAT', 'LON', 'ERA5_MAT_1991_2020', 'ERA5_MAP_1991_2020',
             'ERA5_MAT_SOURCE', 'ERA5_MAT_REASON', 'ERA5_MAP_SOURCE', 'ERA5_MAP_REASON']].to_csv(
    out_plot.with_name(out_plot.stem + '_DATA.csv'), index=False)
print(f"Plot saved successfully to {out_plot}")
if SHOW_PLOT:
    plt.show()