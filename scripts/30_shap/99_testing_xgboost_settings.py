import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import diive as dv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------
# 1. LOAD DATA (Update filepath to your CH-Dav subset)
# ---------------------------------------------------------
# Example filepath, update this to your actual file
filepath = '../../data/outputs/20_subsets/21_subsets_parquet/CH-Dav_subset_warmest4_qc0_daytime.parquet'
try:
    subset = dv.load_parquet(filepath, sanitize_timestamp=False)
except:
    print("Please update the 'filepath' variable to point to your CH-Dav parquet file!")
    raise

features = ['TA_ZSCORE', 'SWIN_ZSCORE', 'VPD_ZSCORE', 'SWC_ZSCORE']
target = 'NEP_ZSCORE'

X = subset[features].copy()
y = subset[target].copy()

# Representative split for early stopping
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)


# ---------------------------------------------------------
# 2. FUNCTION TO TRAIN AND EXTRACT SHAP
# ---------------------------------------------------------
def train_and_get_shap(x):
    print(f"\nTraining XGBoost with x={x}...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=3000,
        max_depth=x,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        early_stopping_rounds=100,
        n_jobs=-1
    )

    # Train
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Print R2
    r2_full = model.score(X, y)
    print(f"R2 (Full Dataset) with x={x}: {r2_full:.4f}")

    # Calculate Conditional SHAP
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X)

    # Return the SHAP values specifically for VPD (index 2 in features list)
    vpd_index = features.index('VPD_ZSCORE')
    return shap_values[:, vpd_index]


# ---------------------------------------------------------
# 3. RUN BOTH MODELS
# ---------------------------------------------------------
shap_vpd_depth4 = train_and_get_shap(x=4)
shap_vpd_depth8 = train_and_get_shap(x=20)

# ---------------------------------------------------------
# 4. PLOT THE RESULTS SIDE-BY-SIDE
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Scatter properties
x_data = X['VPD_ZSCORE']
color_data = X['SWC_ZSCORE']  # Color by Soil Moisture to show interactions
cmap = 'coolwarm_r'  # Red/Blue colormap

# PLOT 1: Depth 4
sc1 = ax1.scatter(x_data, shap_vpd_depth4, c=color_data, cmap=cmap, alpha=0.5, s=15)
ax1.set_title("max_depth = 4\n(Focuses on broader physics)", fontsize=14, fontweight='bold')
ax1.set_xlabel("VPD (Z-Score)", fontsize=12)
ax1.set_ylabel("SHAP Value for VPD\n(Impact on NEP)", fontsize=12)
ax1.axhline(0, color='black', linestyle='--', linewidth=1)

# PLOT 2: Depth 8
sc2 = ax2.scatter(x_data, shap_vpd_depth8, c=color_data, cmap=cmap, alpha=0.5, s=15)
ax2.set_title("max_depth = 8\n(Overfits to local noise)", fontsize=14, fontweight='bold')
ax2.set_xlabel("VPD (Z-Score)", fontsize=12)
ax2.axhline(0, color='black', linestyle='--', linewidth=1)

# Add Colorbar
cbar = fig.colorbar(sc2, ax=[ax1, ax2], fraction=0.02, pad=0.02)
cbar.set_label('Soil Moisture (Z-Score)', rotation=270, labelpad=15, fontsize=12)

plt.suptitle("How Tree Depth Impacts SHAP Value Clarity for Carbon Penalty", fontsize=16)
plt.show()