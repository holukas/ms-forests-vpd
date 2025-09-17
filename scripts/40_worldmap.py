from pathlib import Path

import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import src.files as files

# Load settings and site info
settings = files.read_settings_file("../config/settings.yaml")
siteinfo_df = files.load_siteinfo(settings)

# Define the path to your locally saved world map shapefile
shapefile_path = str(
    Path(r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\ms_co2_penalty\data\worldmap\ne_10m_admin_0_countries.shp"))
# shapefile_path = str(Path(r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\ms_co2_penalty\data\worldmap\ne_110m_admin_0_countries.shp"))
world = gpd.read_file(shapefile_path)

# Create a GeoDataFrame from your site data
gdf = gpd.GeoDataFrame(
    siteinfo_df,
    geometry=gpd.points_from_xy(siteinfo_df.LON, siteinfo_df.LAT),
    crs="EPSG:4326"
)

# Define a dictionary for IGBP classes, markers, and colors
igbp_markers = {
    'ENF': {'marker': '^', 'markersize': 50, 'color': '#76FF03', 'edgecolor': '#1B5E20'},
    'DBF': {'marker': 'o', 'markersize': 40, 'color': '#FF4081', 'edgecolor': '#880E4F'},
    'MF': {'marker': 'v', 'markersize': 50, 'color': '#FFFF00', 'edgecolor': '#F57F17'},
    'EBF': {'marker': 's', 'markersize': 40, 'color': '#18FFFF', 'edgecolor': '#006064'},
    'DNF': {'marker': 'X', 'markersize': 50, 'color': '#455A64', 'edgecolor': 'black'},
}

# --- Create a single figure with a gridspec layout ---
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(2, 2)

# --- Main world map plot (left side, spanning both rows) ---
ax_world = fig.add_subplot(gs[:, 0])
world.plot(ax=ax_world, color='lightgray', edgecolor='#a0a0a0')

for igbp_class, styles in igbp_markers.items():
    subset = gdf[gdf['IGBP'] == igbp_class]
    if not subset.empty:
        subset.plot(ax=ax_world,
                    marker=styles['marker'],
                    color=styles['color'],
                    edgecolor=styles['edgecolor'],
                    markersize=styles['markersize'],
                    alpha=1,
                    label=igbp_class)

ax_world.set_title('Worldwide Eddy Covariance Site Locations', fontsize=20)
ax_world.set_xlabel('Longitude', fontsize=12)
ax_world.set_ylabel('Latitude', fontsize=12)
ax_world.set_facecolor('lightblue')
ax_world.set_xlim([-180, 180])
ax_world.set_ylim([-90, 90])
ax_world.grid(True)

# --- Europe subplot (top-right) ---
ax_eu = fig.add_subplot(gs[0, 1])
world.plot(ax=ax_eu, color='lightgray', edgecolor='#a0a0a0')
for igbp_class, styles in igbp_markers.items():
    subset = gdf[gdf['IGBP'] == igbp_class]
    if not subset.empty:
        subset.plot(ax=ax_eu, marker=styles['marker'], color=styles['color'],
                    edgecolor=styles['edgecolor'], markersize=styles['markersize'], alpha=1)
ax_eu.set_xlim([-15, 45])
ax_eu.set_ylim([35, 70])
ax_eu.set_title('Europe', fontsize=16)
ax_eu.set_xlabel('Longitude', fontsize=10)
ax_eu.set_ylabel('Latitude', fontsize=10)
ax_eu.set_facecolor('lightblue')
ax_eu.grid(True)

# --- East Asia subplot (bottom-right) ---
ax_ea = fig.add_subplot(gs[1, 1])
world.plot(ax=ax_ea, color='lightgray', edgecolor='#a0a0a0')
for igbp_class, styles in igbp_markers.items():
    subset = gdf[gdf['IGBP'] == igbp_class]
    if not subset.empty:
        subset.plot(ax=ax_ea, marker=styles['marker'], color=styles['color'],
                    edgecolor=styles['edgecolor'], markersize=styles['markersize'], alpha=1)
ax_ea.set_xlim([115, 150])
ax_ea.set_ylim([25, 50])
ax_ea.set_title('East Asia', fontsize=16)
ax_ea.set_xlabel('Longitude', fontsize=10)
ax_ea.set_ylabel('Latitude', fontsize=10)
ax_ea.set_facecolor('lightblue')
ax_ea.grid(True)

# Add a single legend to the whole figure
handles, labels = ax_world.get_legend_handles_labels()
fig.legend(handles, labels, title='IGBP Class', loc='center left', bbox_to_anchor=(0.95, 0.5), fontsize=12)

plt.tight_layout()
plt.show()
