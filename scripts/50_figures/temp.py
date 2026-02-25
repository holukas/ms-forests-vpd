from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Load the updated dataset with the ERA5 variables
infile = Path('../../data/outputs/10_datasets/17_datasets_info_parquet_vars_stats_usedsites_era5.csv')
df = pd.read_csv(infile)

# Your custom markers dictionary
igbp_markers = {
    'ENF': {'marker': '^', 'markersize': 50, 'color': '#76FF03', 'edgecolor': '#1B5E20'},
    'DBF': {'marker': 'o', 'markersize': 40, 'color': '#FF4081', 'edgecolor': '#880E4F'},
    'MF':  {'marker': 'v', 'markersize': 50, 'color': '#FFFF00', 'edgecolor': '#F57F17'},
    'EBF': {'marker': 's', 'markersize': 40, 'color': '#18FFFF', 'edgecolor': '#006064'},
    'DNF': {'marker': 'X', 'markersize': 50, 'color': '#455A64', 'edgecolor': 'black'},
}

# Filter out any rows missing ERA5 data
valid_df = df.dropna(subset=['ERA5_MAT_1991_2020', 'ERA5_MAP_1991_2020']).copy()
n_sites = len(valid_df)

# --- ISOLATION CALCULATION ---
# Standardize MAT and MAP so they have equal weight in the distance calculation
mat_z = (valid_df['ERA5_MAT_1991_2020'] - valid_df['ERA5_MAT_1991_2020'].mean()) / valid_df['ERA5_MAT_1991_2020'].std()
map_z = (valid_df['ERA5_MAP_1991_2020'] - valid_df['ERA5_MAP_1991_2020'].mean()) / valid_df['ERA5_MAP_1991_2020'].std()

# Calculate distance matrix between all points
coords = np.column_stack((mat_z, map_z))
dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
np.fill_diagonal(dist_matrix, np.inf) # Ignore distance to itself

# Find the distance to the nearest neighbor for each site
valid_df['Isolation_Score'] = dist_matrix.min(axis=1)

# Select the top N most isolated sites to label
num_to_label = 7
isolated_sites = valid_df.nlargest(num_to_label, 'Isolation_Score')
# -----------------------------

# 2. Set up the quadratic (square) plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.grid(True, linestyle='--', alpha=0.6, zorder=0)

igbp_col = 'IGBP'

# 3. Loop through each IGBP class and plot them
for igbp_class, group_df in valid_df.groupby(igbp_col):
    if igbp_class in igbp_markers:
        style = igbp_markers[igbp_class]
        ax.scatter(
            group_df['ERA5_MAT_1991_2020'],
            group_df['ERA5_MAP_1991_2020'],
            marker=style['marker'],
            s=style['markersize'],
            c=style['color'],
            edgecolors=style['edgecolor'],
            linewidths=1,
            alpha=0.85,
            label=igbp_class,
            zorder=3
        )
    else:
        ax.scatter(
            group_df['ERA5_MAT_1991_2020'],
            group_df['ERA5_MAP_1991_2020'],
            marker='o',
            s=30,
            c='#E0E0E0',
            edgecolors='#9E9E9E',
            alpha=0.6,
            label=f'{igbp_class} (Other)',
            zorder=2
        )

# 4. Add labels for the isolated sites
for _, row in isolated_sites.iterrows():
    ax.annotate(
        row['SITE'],
        (row['ERA5_MAT_1991_2020'], row['ERA5_MAP_1991_2020']),
        xytext=(6, 6), # Offset the text slightly up and to the right
        textcoords='offset points',
        fontsize=9,
        fontweight='bold',
        color='#222222',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#aaaaaa", alpha=0.7),
        zorder=5
    )

# 5. Add labels, title, and site count text
ax.set_title('Climate Space of Study Sites by Forest Type', pad=15, fontweight='bold', fontsize=14)
ax.set_xlabel('Mean Annual Temperature (°C)', fontweight='bold', fontsize=12)
ax.set_ylabel('Mean Annual Precipitation (mm)', fontweight='bold', fontsize=12)

# Add "n=" text box
ax.text(
    0.03, 0.97,
    f'n = {n_sites}',
    transform=ax.transAxes,
    fontsize=12,
    fontweight='bold',
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#cccccc'),
    zorder=4
)

# Move legend inside the plot (e.g., lower right)
ax.legend(
    title='IGBP Class',
    loc='lower right',
    frameon=True,
    framealpha=0.9, # Make the background slightly opaque
    edgecolor='#cccccc'
)

# 6. Save and display
plt.tight_layout(pad=1.5)
out_plot = Path('../../data/outputs/10_datasets/18_site_climate_space_square_labeled.png')
# plt.savefig(out_plot, dpi=300)

print(f"Plot saved successfully to {out_plot}")
plt.show()