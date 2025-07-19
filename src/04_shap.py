import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from diive.core.io.files import load_parquet

df = pd.read_csv('../OUT/03_siteinfo.csv')
df = df.fillna(np.nan)

tacol = 'TA_F'
taqc = 'TA_F_QC'
vpdcol = 'VPD_F'
vpdqc = 'VPD_F_QC'
preccol = 'P_F'
swccol = 'SWC_F_MDS_1'
# fluxcol = 'GPP_NT_VUT_50'
fluxcol = 'NEE_VUT_50'
fluxqc = "NEE_VUT_50_QC"
swinpotcol = "SW_IN_POT"  # For daytime/nighttime
swincol = "SW_IN_F"

_df = df.copy()
for ix, row in _df.iterrows():

    # if row['SITE'] != 'CH-Dav':
    if row['SITE'] != 'BE-Bra':
        continue

    print(f"\nLoading data for site {row['SITE']} ...")

    filepath = row['_FILEPATH_PARQUET']
    sitedata = load_parquet(filepath)

    # [print(c) for c in sitedata.columns if "GPP" in c];

    # Keep data for 6 warmest months
    sitedata['MONTH'] = sitedata.index.month
    monthly_avg = sitedata.groupby('MONTH').mean()
    monthly_avg = monthly_avg.sort_values(by='TA_F', ascending=False, inplace=False)
    warmest6 = list(monthly_avg.head(6).index)
    locs = \
        (sitedata.index.month == warmest6[0]) | (sitedata.index.month == warmest6[1]) | (
                sitedata.index.month == warmest6[2]) | (sitedata.index.month == warmest6[3]) | (
                sitedata.index.month == warmest6[4]) | (sitedata.index.month == warmest6[5])
    sitedata_warmest6 = sitedata.loc[locs].copy()

    # Keep directly measured fluxes and meteo, no gap-filled data
    locs_qc0 = (sitedata_warmest6[fluxqc] == 0) & (sitedata_warmest6[taqc] == 0) & (sitedata_warmest6[vpdqc] == 0)
    sitedata_warmest6_qc0 = sitedata_warmest6.loc[locs_qc0].copy()

    # Keep daytime data
    locs_dt = sitedata_warmest6_qc0[swinpotcol] > 20
    sitedata_warmest6_qc0_dt = sitedata_warmest6_qc0.loc[locs_dt].copy()
    # locs_dt = sitedata_warmest6[swinpotcol] > 20
    # sitedata_warmest6_qc0_dt = sitedata_warmest6.loc[locs_dt].copy()

    subset = sitedata_warmest6_qc0_dt[[fluxcol, tacol, vpdcol, swccol, swincol]].copy()
    subset = subset.dropna()
    from scipy.stats import zscore
    subset = subset.apply(zscore)
    subset = subset.loc[subset.index.year == 2019].copy()

    features = [tacol, vpdcol, swccol, swincol]
    X = subset[features]
    y = subset[fluxcol]




    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    from sklearn.metrics import mean_squared_error, r2_score

    # Initialize XGBoost Regressor
    # objective='reg:squarederror' for standard regression
    # n_estimators: number of boosting rounds (trees)
    # learning_rate: step size shrinkage to prevent overfitting
    # max_depth: maximum depth of a tree
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,  # A good starting point, can be tuned
        learning_rate=0.5,
        max_depth=6,  # Can be tuned
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )

    # # Initialize Random Forest Regressor
    # # n_estimators: number of trees in the forest
    # # random_state: for reproducibility
    # # n_jobs: number of CPU cores to use (-1 means all available)
    # from sklearn.ensemble import RandomForestRegressor
    # model = RandomForestRegressor(
    #     n_estimators=100,  # A common starting point, can be tuned
    #     max_depth=6,  # Can be tuned, limits depth of each tree
    #     random_state=42,
    #     n_jobs=-1
    # )

    # Train the model
    print("Training XGBoost model...")
    model.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate model performance on the test set
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R-squared: {r2:.2f}")

    # Create a SHAP TreeExplainer for the trained XGBoost model
    explainer = shap.TreeExplainer(model)
    print("SHAP Explainer created.")

    # Calculate SHAP interaction values for a subset of the data or the full dataset
    # For large datasets, consider using a representative sample (e.g., X_train.sample(5000, random_state=42))
    print("Calculating SHAP interaction values... This might take some time for large datasets.")
    shap_interaction_values = explainer.shap_interaction_values(X_train)
    print("SHAP interaction values calculated.")

    # The result is a 3D NumPy array:
    # [number_of_samples, number_of_features, number_of_features]
    print(f"Shape of shap_interaction_values: {shap_interaction_values.shape}")

    # # For a heatmap of average absolute interaction values
    # # The plot shows feature interaction importance based on the mean absolute SHAP interaction values.
    # # The diagonal shows the main effect of each feature.
    # shap.summary_plot(shap_interaction_values, X_train, feature_names=X_train.columns)

    # Example: How does the effect of Solar Radiation (Rg) on LE change with Air Temperature (Ta)?
    # Replace 'Rg' and 'Ta' with your actual feature names if different
    # Ensure these features exist in X_train.columns
    shap_values = explainer.shap_values(X_train)
    if swincol in X_train.columns and tacol in X_train.columns:
        print("\nGenerating dependence plot for Rg interacting with Ta...")
        shap.dependence_plot(
            swincol,  # The feature whose SHAP values you want to plot on the y-axis
            shap_values=shap_values,  # Use standard shap_values for dependence plot
            features=X_train,
            interaction_index=tacol,  # The feature with which you want to see the interaction
            x_jitter=0.5  # Add jitter for better visualization of scattered points
        )
    else:
        print("Features 'Rg' or 'Ta' not found in the dataset for dependence plot example.")


    # Top interactions
    # Calculate the mean absolute interaction values
    mean_abs_interaction = np.abs(shap_interaction_values).mean(0)

    # Create a DataFrame for better viewing
    interaction_df = pd.DataFrame(mean_abs_interaction, index=X_train.columns, columns=X_train.columns)

    # Set diagonal values (main effects) to NaN or 0 for clearer interaction analysis
    np.fill_diagonal(interaction_df.values, np.nan)  # Or 0

    # Stack to get pairs and sort
    sorted_interactions = interaction_df.stack().sort_values(ascending=False)

    print("\nFeature interactions:")
    # print(sorted_interactions.head(10))
    print(sorted_interactions)





#     y = np.array(subset[fluxcol])
#     X = subset[[tacol, vpdcol, swccol, swincol]].copy()
#     # X = np.array(subset.drop(fluxcol, axis=1))
#     model = xgb.XGBRegressor()
#     model = model.fit(X=X, y=y)
#     explainer = shap.Explainer(model)
#     shap_values = explainer(X)
#     # shap_interaction_values = explainer.shap_interaction_values(X)
#     # shap.interaction_plot(shap_interaction_values[i], X.iloc[[i]], feature_names=X.columns)
#     # shap.plots.waterfall(shap_values[1])
#     shap.plots.scatter(shap_values[:, tacol], color=shap_values)
#     shap.plots.scatter(shap_values[:, vpdcol], color=shap_values)
#     shap.plots.scatter(shap_values[:, swccol], color=shap_values)
#     shap.plots.scatter(shap_values[:, swincol], color=shap_values)
#     shap.plots.bar(shap_values)
#     # shap.summary_plot(shap_values)
# # df.to_csv("../OUT/03_siteinfo.csv", index=False)
