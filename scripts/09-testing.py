import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

# Generate more sample data to increase bin counts
np.random.seed(42) # Use a fixed seed for reproducibility
x = np.random.randn(20000) * 2 + 5 # Increased to 20,000 points
y = np.random.randn(20000) * 1.5 + 3

plt.figure(figsize=(10, 8))

# Create the hexbin plot
hb = plt.hexbin(x, y,
                gridsize=30,
                cmap='viridis',
                alpha=0.6,    # Make it semi-transparent
                edgecolors='none', # No default edges
                zorder=1      # Default zorder for PolyCollection is 1
               )

# Get the paths (vertices) and counts of each hexagon
paths = hb.get_paths()
counts = hb.get_array()

# --- Debugging: Print count statistics ---
if len(counts) > 0:
    min_count = counts.min()
    max_count = counts.max()
    avg_count = counts.mean()
    median_count = np.median(counts) # Median can be more robust than mean for skewed data

    print(f"Hexbin Count Statistics:")
    print(f"  Min Count: {min_count:.2f}")
    print(f"  Max Count: {max_count:.2f}")
    print(f"  Average Count: {avg_count:.2f}")
    print(f"  Median Count: {median_count:.2f}")
else:
    print("No counts available from hexbin.")
    plt.colorbar(hb, label='Hexbin Counts')
    plt.title('Hexbin Plot (No Data for Highlighting)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()
    exit() # Exit if no counts, no point in continuing

# --- Adjusting the threshold calculation ---
# Option 1: Use a lower percentile (e.g., 75th or 50th)
# threshold = np.percentile(counts, 75) # Highlight top 25%

# Option 2: Use a fixed number that you know will be met
# This is good for testing the highlighting mechanism itself
# For instance, if max_count is 20, try setting threshold to 5 or 10.
# Let's try highlighting anything above the median count for now
threshold = median_count
print(f"Highlighting bins with counts >= {threshold:.2f} (Median based)")


# Create a new PolyCollection for the highlighted outlines
highlighted_verts = []
for i, path in enumerate(paths):
    if i < len(counts) and counts[i] >= threshold:
        highlighted_verts.append(path.vertices)

if highlighted_verts:
    highlight_collection = PolyCollection(highlighted_verts,
                                          edgecolors='red',
                                          linewidths=2.5,
                                          facecolors='none',
                                          zorder=3)
    plt.gca().add_collection(highlight_collection)
    print(f"Added {len(highlighted_verts)} highlighted hexbins.")
else:
    print("No hexbins met the highlight threshold even after adjustment. Check data distribution.")


plt.colorbar(hb, label='Hexbin Counts')
plt.title('Hexbin Plot with Specific Bins Highlighted (Threshold Adjusted)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()