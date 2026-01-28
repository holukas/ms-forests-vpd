from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Latin Modern Roman'] + plt.rcParams['font.serif']

# Load datasets info
infile = Path('../../data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv')
datasets_df = pd.read_csv(infile)

n_sites_total = len(datasets_df)

datasets_df = datasets_df[['TA_AVG', 'PREC/YR', 'IGBP', 'SITE']].copy()
print(datasets_df.sort_values(by='PREC/YR'))

# # Create scatter plot
# plt.scatter(subset['TA_AVG'], subset['PREC/YR'], color='blue', marker='o')
# plt.xlabel('TA_AVG')
# plt.ylabel('PREC/YR')
# plt.title('Scatter Plot of TA_AVG vs PREC/YR')
# plt.grid(True)
# plt.show()


# Define a dictionary for IGBP classes, markers, and colors
igbp_markers = {
    'ENF': {'marker': '^', 'markersize': 50, 'color': '#76FF03', 'edgecolor': '#1B5E20'},
    'DBF': {'marker': 'o', 'markersize': 40, 'color': '#FF4081', 'edgecolor': '#880E4F'},
    'MF': {'marker': 'v', 'markersize': 50, 'color': '#FFFF00', 'edgecolor': '#F57F17'},
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

# Create a single figure and axes for the world map
fig, ax_scatter = plt.subplots(1, 1, figsize=(8, 8))

# Plot the sites, colored by IGBP class
found_igbp = {}
for igbp_class, styles in igbp_markers.items():
    subset = datasets_df[datasets_df['IGBP'] == igbp_class]
    found_igbp[igbp_class] = len(subset)
    if not subset.empty:
        ax_scatter.scatter(subset['TA_AVG'], subset['PREC/YR'],
                           label=f"{labels[igbp_class]}n={len(subset)})",
                           marker=styles['marker'],
                           color=styles['color'],
                           edgecolor=styles['edgecolor'],
                           s=styles['markersize'], alpha=1)

# Set plot titles and labels
# ax_scatter.set_title(f'Eddy covariance sites (n={n_sites_total})', fontsize=16)
# ax_scatter.set_xlabel('Longitude', fontsize=12)
# ax_scatter.set_ylabel('Latitude', fontsize=12)
# ax_scatter.set_facecolor('lightblue')
# ax_scatter.set_xlim([-180, 180])
# ax_scatter.set_ylim([-90, 90])
# ax_scatter.grid(True)

# Add a legend to the plot
ax_scatter.legend(title='IGBP Class', loc='upper right', fontsize=10)

plt.show()
