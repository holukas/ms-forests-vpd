from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.paths import data_path, repo_path

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

# Map background colours
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
fig = plt.figure(figsize=(21 * 0.9, 10 * 0.9), constrained_layout=True)
# Added fig.add_gridspec to fix the UserWarning
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], height_ratios=[1.6, 1])

ax_world = fig.add_subplot(gs[0, :2])
ax_usa = fig.add_subplot(gs[1, 0])
ax_eu = fig.add_subplot(gs[1, 1])
ax_climate = fig.add_subplot(gs[:, 2])


# ---------------------------------------------------------
# 4. HELPER FUNCTION TO PLOT MAPS & ADD PANEL LETTERS
# ---------------------------------------------------------
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

    # Adjust title/letter height to sit comfortably above the new ticks/borders
    # Set a smaller relative offset for the wide main map so the physical gap matches
    title_x_offset = 0.025 if is_main_map else 0.06

    # Adjust title/letter height to sit comfortably above the new ticks/borders
    ax.text(0.0, 1.05, letter, transform=ax.transAxes, fontsize=AX_LABELS_FONTSIZE * 1.2, fontweight='bold',
            va='bottom', ha='left', color='#000000')
    ax.text(title_x_offset, 1.05, title, transform=ax.transAxes, fontsize=AX_LABELS_FONTSIZE * 1.2,
            va='bottom', ha='left', color='#555555')


# ---------------------------------------------------------
# 5. PLOT THE MAPS (Left Panels)
# ---------------------------------------------------------
# Global map adjusted to align with the aspect ratio of the subpanels below
plot_map_region(ax_world, extent=[-180, 180, -60, 85], title=f'Global forest sites in this study (n={n_sites_total})',
                letter='a', is_main_map=True)

# Because we added an x-axis label, the legend needs to be pushed slightly further down
ax_world.legend(loc='lower center', bbox_to_anchor=(0.5, -0.28), ncol=5, fontsize=AX_LABELS_FONTSIZE, frameon=False,
                columnspacing=1.5)

# MATHEMATICAL ALIGNMENT:
# To make panels b and c the EXACT same height and width, their coordinate spans must be identical.
# Both are now set to span exactly 64 degrees of Longitude and 32 degrees of Latitude.
extent_us = [-128, -64, 22, 58]  # 64 lon span, 36 lat span
extent_eu = [-10, 54, 36, 72]  # 64 lon span, 36 lat span

plot_map_region(ax_usa, extent=extent_us, title="Contiguous US", letter='b')
plot_map_region(ax_eu, extent=extent_eu, title="Europe", letter='c')

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

# Fixed the transform argument to ax_climate.transAxes
# Fixed the transform argument to ax_climate.transAxes AND standardized the y-height to 1.05
ax_climate.text(0.0, 1.02, 'd', transform=ax_climate.transAxes, fontsize=AX_LABELS_FONTSIZE * 1.2, fontweight='bold',
                va='bottom', ha='left', color='#000000')
ax_climate.text(0.06, 1.02, f'Bioclimatic distribution (n={n_sites_climate})', transform=ax_climate.transAxes,
                fontsize=AX_LABELS_FONTSIZE * 1.2, va='bottom', ha='left', color='#555555')

# ---------------------------------------------------------
# 7. SAVE AND SHOW
# ---------------------------------------------------------
out_plot = data_path("data/outputs/50_plots/NEP_ZSCORE/conditional/51_FIG-1_18_WorldMap_MAT_MAP_ERA5.png")
out_plot.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(out_plot, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Plot saved successfully to {out_plot}")
plt.show()
