from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# Load datasets info
infile = Path('../data/outputs/12_datasets_parquet_vars_stats_subsets.csv')
datasets_df = pd.read_csv(infile)

# Keep datasets where SWC is available
datasets_df = datasets_df.loc[datasets_df['SWC_AVG'] != '-MISSING-'].copy()

# Remove DNF sites (only 2 sites)
datasets_df = datasets_df.loc[datasets_df['IGBP'] != 'DNF'].copy()

# Define the path to your locally saved world map shapefile
shapefile_path = str(
    Path(r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\ms_co2_penalty\data\worldmap\ne_10m_admin_0_countries.shp"))
world = gpd.read_file(shapefile_path)

# Create a GeoDataFrame from your site data
gdf = gpd.GeoDataFrame(
    datasets_df,
    geometry=gpd.points_from_xy(datasets_df.LON, datasets_df.LAT),
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

labels = dict(
    ENF='evergreen needleleaf forests (ENF)',
    DBF='deciduous broadleaf forests (DBF)',
    MF='mixed forests (MF)',
    EBF='evergreen broadleaf forests (EBF)',
    DNF='deciduous needleleaf forests (DNF)',
)

# Create a single figure and axes for the world map
fig, ax_world = plt.subplots(1, 1, figsize=(15, 8))

# Plot the world map
world.plot(ax=ax_world, color='lightgray', edgecolor='#a0a0a0')

# Plot the sites, colored by IGBP class
found_igbp = {}
for igbp_class, styles in igbp_markers.items():
    subset = gdf[gdf['IGBP'] == igbp_class]
    found_igbp[igbp_class] = len(subset)
    if not subset.empty:
        subset.plot(ax=ax_world,
                    marker=styles['marker'],
                    color=styles['color'],
                    edgecolor=styles['edgecolor'],
                    markersize=styles['markersize'],
                    alpha=1,
                    label=f"{labels[igbp_class]} ({len(subset)} sites)")

# Set plot titles and labels
ax_world.set_title('Worldwide Eddy Covariance Site Locations', fontsize=16)
ax_world.set_xlabel('Longitude', fontsize=12)
ax_world.set_ylabel('Latitude', fontsize=12)
ax_world.set_facecolor('lightblue')
ax_world.set_xlim([-180, 180])
ax_world.set_ylim([-90, 90])
ax_world.grid(True)

# Add a legend to the plot
ax_world.legend(title='IGBP Class', loc='lower left', fontsize=10)

plt.show()
