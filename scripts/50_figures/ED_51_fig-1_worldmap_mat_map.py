from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. DATA PREPARATION
# ---------------------------------------------------------
infile = Path('../../data/outputs/10_datasets/17_datasets_info_parquet_vars_stats_usedsites_era5.csv')
datasets_df = pd.read_csv(infile)

# Filter dataset based on your rules
datasets_df = datasets_df.loc[datasets_df['SWC_AVG'] != '-MISSING-'].copy()
datasets_df = datasets_df.loc[datasets_df['IGBP'] != 'DNF'].copy()
n_sites_total = len(datasets_df)

# Prepare dataframe specifically for the ERA5 Climate Space plot
valid_df = datasets_df.dropna(subset=['ERA5_MAT_1991_2020', 'ERA5_MAP_1991_2020']).copy()
n_sites_climate = len(valid_df)

# Prepare Map Geometries
shapefile_path = str(Path(r"../../data/worldmap/ne_10m_admin_0_countries.shp"))
world = gpd.read_file(shapefile_path)

gdf = gpd.GeoDataFrame(
    datasets_df,
    geometry=gpd.points_from_xy(datasets_df.LON, datasets_df.LAT),
    crs="EPSG:4326"
)

# Shared Styling Dictionaries
igbp_markers = {
    'ENF': {'marker': '^', 'markersize': 50, 'color': '#76FF03', 'edgecolor': '#1B5E20'},
    'DBF': {'marker': 'o', 'markersize': 40, 'color': '#FF4081', 'edgecolor': '#880E4F'},
    'MF':  {'marker': 'v', 'markersize': 50, 'color': '#FFFF00', 'edgecolor': '#F57F17'},
    'EBF': {'marker': 's', 'markersize': 40, 'color': '#18FFFF', 'edgecolor': '#006064'},
    'DNF': {'marker': 'X', 'markersize': 50, 'color': '#455A64', 'edgecolor': 'black'},
}

labels = dict(
    ENF='ENF, evergreen needleleaf forests (',
    DBF='DBF, deciduous broadleaf forests (',
    MF='MF, mixed forests (',
    EBF='EBF, evergreen broadleaf forests (',
    DNF='DNF, deciduous needleleaf forests (',
)

# ---------------------------------------------------------
# 2. ISOLATION CALCULATION (For Climate Plot Labels)
# ---------------------------------------------------------
mat_z = (valid_df['ERA5_MAT_1991_2020'] - valid_df['ERA5_MAT_1991_2020'].mean()) / valid_df['ERA5_MAT_1991_2020'].std()
map_z = (valid_df['ERA5_MAP_1991_2020'] - valid_df['ERA5_MAP_1991_2020'].mean()) / valid_df['ERA5_MAP_1991_2020'].std()

coords = np.column_stack((mat_z, map_z))
dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
np.fill_diagonal(dist_matrix, np.inf)

valid_df['Isolation_Score'] = dist_matrix.min(axis=1)
num_to_label = 7
isolated_sites = valid_df.nlargest(num_to_label, 'Isolation_Score')

# ---------------------------------------------------------
# 3. SET UP SIDE-BY-SIDE LAYOUT
# ---------------------------------------------------------
# 1 row, 2 columns. width_ratios=[1.8, 1] gives the map almost twice the width of the scatter plot.
fig, (ax_world, ax_climate) = plt.subplots(1, 2, figsize=(23, 8), gridspec_kw={'width_ratios': [1.8, 1]})

# ---------------------------------------------------------
# 4. PLOTTING THE WORLD MAP (Left Panel)
# ---------------------------------------------------------
world.plot(ax=ax_world, color='lightgray', edgecolor='#a0a0a0')

# Plot the sites on the map
for igbp_class, styles in igbp_markers.items():
    subset = gdf[gdf['IGBP'] == igbp_class]
    if not subset.empty:
        subset.plot(ax=ax_world,
                    marker=styles['marker'],
                    color=styles['color'],
                    edgecolor=styles['edgecolor'],
                    markersize=styles['markersize'],
                    alpha=1,
                    label=f"{labels[igbp_class]}n={len(subset)})",
                    zorder=3)

# Style Map
ax_world.set_title(f'Eddy Covariance Sites (n={n_sites_total})', fontsize=16, fontweight='bold', pad=15)
ax_world.set_xlabel('Longitude', fontsize=12)
ax_world.set_ylabel('Latitude', fontsize=12)
ax_world.set_facecolor('lightblue')
ax_world.set_xlim([-180, 180])
ax_world.set_ylim([-90, 90])
ax_world.grid(True, linestyle='--', alpha=0.5)

# Add map legend in the lower left so it stays out of the way of Europe/North America
ax_world.legend(title='IGBP Class', loc='lower left', fontsize=10, framealpha=0.9)

# ---------------------------------------------------------
# 5. PLOTTING THE CLIMATE SPACE (Right Panel)
# ---------------------------------------------------------
ax_climate.grid(True, linestyle='--', alpha=0.6, zorder=0)

# Plot the climate points
for igbp_class, group_df in valid_df.groupby('IGBP'):
    if igbp_class in igbp_markers:
        style = igbp_markers[igbp_class]
        ax_climate.scatter(
            group_df['ERA5_MAT_1991_2020'],
            group_df['ERA5_MAP_1991_2020'],
            marker=style['marker'],
            s=style['markersize'],
            c=style['color'],
            edgecolors=style['edgecolor'],
            linewidths=1,
            alpha=0.85,
            zorder=3
        )

# Label isolated sites
for _, row in isolated_sites.iterrows():
    ax_climate.annotate(
        row['SITE'],
        (row['ERA5_MAT_1991_2020'], row['ERA5_MAP_1991_2020']),
        xytext=(6, 6),
        textcoords='offset points',
        fontsize=9,
        fontweight='bold',
        color='#222222',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#aaaaaa", alpha=0.7),
        zorder=5
    )

# Style Climate Plot
ax_climate.set_title('Climate Space', fontsize=16, fontweight='bold', pad=15)
ax_climate.set_xlabel('Mean Annual Temperature (°C)', fontweight='bold', fontsize=12)
ax_climate.set_ylabel('Mean Annual Precipitation (mm)', fontweight='bold', fontsize=12)
ax_climate.set_facecolor('white')

# "n=" box for the climate plot
ax_climate.text(
    0.03, 0.97,
    f'n = {n_sites_climate}',
    transform=ax_climate.transAxes,
    fontsize=12,
    fontweight='bold',
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='#cccccc'),
    zorder=4
)

# ---------------------------------------------------------
# 6. SAVE AND SHOW
# ---------------------------------------------------------
# tight_layout handles the spacing between the two subplots perfectly
plt.tight_layout(pad=2.0)
out_plot = Path('../../data/outputs/10_datasets/18_site_map_and_climate_space_side_by_side.png')
# plt.savefig(out_plot, dpi=300)

print(f"Plot saved successfully to {out_plot}")
plt.show()