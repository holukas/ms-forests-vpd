import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from diive.core.io.files import load_parquet
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import src.files as files
import diive as dv

# Load site info
siteinfo_df = files.load_siteinfo(filename="02_siteinfo.csv")

tacol = 'TA_F'
taqc = 'TA_F_QC'
vpdcol = 'VPD_F'
vpdqc = 'VPD_F_QC'
preccol = 'P_F'
swccol = 'SWC_F_MDS_1'
# fluxcol = 'GPP_NT_VUT_50'
# fluxcol = 'LE_F_MDS'
fluxcol = 'NEE_VUT_50'
fluxqc = "NEE_VUT_50_QC"
swinpotcol = "SW_IN_POT"  # For daytime/nighttime
swincol = "SW_IN_F"

_df = siteinfo_df.copy()
for ix, row in _df.iterrows():

    # if row['SITE'] != 'CH-Dav':
    # if row['SITE'] != 'BE-Bra':
    if ix != 0:
        continue

    print(f"\nLoading data for site {row['SITE']} ...")

    # Load site data
    filepath = row['_FILEPATH_PARQUET']
    sitedata = load_parquet(filepath)
    # [print(c) for c in sitedata.columns if "LE" in c];

    # Detect 6 warmest months
    ta = sitedata[[tacol]].copy()
    ta['MONTH'] = ta.index.month
    monthly_avg = ta.groupby('MONTH').mean()
    monthly_avg = monthly_avg.sort_values(by=tacol, ascending=False, inplace=False)
    warmest6 = monthly_avg.head(6).index.to_list()
    # locs = sitedata.index.month.isin(warmest6)

    # Make subset and convert to z-scores
    # todo data leakage train test, if needed
    subset = sitedata[[fluxcol, tacol, vpdcol, swccol, swincol]].copy()

    # Convert to z-scores, ignoring NaNs
    subset = subset.apply(lambda x: zscore(x, nan_policy='omit'))

    # Add original flux quality flag and potential radiation
    subset[fluxqc] = sitedata[fluxqc].copy()
    subset[swinpotcol] = sitedata[swinpotcol].copy()

    # Filter subset
    # Keep data from 6 warmest months
    # Keep directly measured fluxes, no gap-filled data
    # Keep daytime records
    # Keep required variables only
    # Keep records where all vars available
    subset = subset.loc[
        subset.index.month.isin(warmest6) &
        (subset[fluxqc] == 0) &
        (subset[swinpotcol] > 20),
        [fluxcol, tacol, vpdcol, swincol, swccol]
    ].copy()
    subset = subset.dropna()
    # keeplocs = subset.index.year == 2020
    # subset = subset[keeplocs].copy()
    print(f"Records: {len(subset)}")

    # Limit time range
    # subset = subset.loc[subset.index.year == 2019].copy()

    # Target and features
    # features = [tacol, vpdcol, swincol]
    features = [tacol, vpdcol, swccol, swincol]
    X = subset[features]
    y = subset[fluxcol]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(f"Training set size: {X_train.shape[0]} samples")
    # print(f"Testing set size: {X_test.shape[0]} samples")

    # Initialize XGBoost Regressor
    # objective='reg:squarederror' for standard regression
    # n_estimators: number of boosting rounds (trees)
    # learning_rate: step size shrinkage to prevent overfitting
    # max_depth: maximum depth of a tree
    # Initialize and train the XGBoost Regressor model
    model = xgb.XGBRegressor(objective='reg:squarederror',  # For regression tasks
                             n_estimators=1000,  # Number of boosting rounds
                             learning_rate=0.5,  # Step size shrinkage to prevent overfitting
                             max_depth=6,  # Maximum depth of a tree
                             subsample=.9,  # Subsample ratio of the training instance
                             colsample_bytree=.9,  # Subsample ratio of columns when constructing each tree
                             random_state=42,
                             early_stopping_rounds=50,  # Stop if validation metric doesn't improve for 50 rounds
                             n_jobs=-1)  # Use all available CPU cores

    # Train the model
    model.fit(X, y, eval_set=[(X, y)], verbose=False)


    # Evaluate model performance on the test set
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"R2: {model.score(X, y):.4f}")
    print(f"RMSE: {rmse:.2f}")
    # print(f"Train R2: {model.score(X_train, y_train):.4f}")
    # print(f"Test  R2: {model.score(X_test, y_test):.4f}")
    # y_pred_test = model.predict(X_test)
    # y_pred_train = model.predict(X_train)
    # rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    # rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    # print(f"Test RMSE: {rmse_test:.2f}")
    # print(f"Train RMSE: {rmse_train:.2f}")

    # Create a SHAP TreeExplainer for the trained XGBoost model
    explainer = shap.TreeExplainer(model)
    print("SHAP Explainer created.")

    # Get SHAP values
    shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value

    # Collect SHAP values in dataframe
    shapcols = [f'{c}_SHAPVALS' for c in X]
    shapdf = pd.DataFrame(data=shap_values, index=X.index, columns=shapcols)
    shapdf.index = pd.to_datetime(shapdf.index)
    shapdf['SUM'] = shapdf.sum(axis=1)
    shapdf['EXPECTED'] = expected_value
    shapdf['SUM+EXPECTED'] = shapdf['EXPECTED'].add(shapdf['SUM'])
    shapdf[fluxcol] = y.copy()
    # print(shapdf)

    merged = pd.concat([X, shapdf], axis=1)
    # print(merged)

    # from diive.core.plotting.scatter import ScatterXY
    # ScatterXY(x=merged['VPD_F'], y=merged['VPD_F_SHAPVALS'], nbins=200,
    #           xlim=[-1, 10], binagg='mean').plot()


    # xvar = vpdcol
    # yvar = tacol
    # zvar = fluxcol
    # plotdf = shapdf[[xvar, yvar, zvar]].copy()

    q = dv.ga(
        x=merged['TA_F'],
        y=merged['VPD_F'],
        z=merged['VPD_F_SHAPVALS'],
        binning_type='custom',
        custom_x_bins=list(np.arange(-8, 10, .5)),
        custom_y_bins=list(np.arange(-8, 10, .5)),
        # binning_type='quantiles',
        # n_bins=20,
        min_n_vals_per_bin=3,
        aggfunc='mean'
    )
    print(q.df_agg_wide)


    hm = dv.heatmapxyz(
        x=q.df_long[f'BIN_TA_F'],
        y=q.df_long[f'BIN_VPD_F'],
        z=q.df_long['VPD_F_SHAPVALS'],
        cb_digits_after_comma=0,
        xlabel=r'x',
        ylabel=r'y',
        zlabel=r'z',
        # vmin=-3,
        # vmax=3
    )
    hm.show()

#     # Calculate SHAP interaction values for a subset of the data or the full dataset
#     # For large datasets, consider using a representative sample (e.g., X_train.sample(5000, random_state=42))
#     print("Calculating SHAP interaction values... This might take some time for large datasets.")
#     shap_interaction_values = explainer.shap_interaction_values(X)
#     print("SHAP interaction values calculated.")
#
#     # The result is a 3D NumPy array:
#     # [number_of_samples, number_of_features, number_of_features]
#     print(f"Shape of shap_interaction_values: {shap_interaction_values.shape}")
#
#     # # For a heatmap of average absolute interaction values
#     # # The plot shows feature interaction importance based on the mean absolute SHAP interaction values.
#     # # The diagonal shows the main effect of each feature.
#     # shap.summary_plot(shap_interaction_values, X_train, feature_names=X_train.columns)
#

    # # Example: How does the effect of Solar Radiation (Rg) on LE change with Air Temperature (Ta)?
    # # Replace 'Rg' and 'Ta' with your actual feature names if different
    # # Ensure these features exist in X_train.columns
    # # shap_values = explainer.shap_values(X_train)
    # # shap_values = explainer.shap_values(X)
    # xvar = vpdcol
    # yvar = tacol
    # if xvar in X.columns and yvar in X.columns:
    #     print("\nGenerating dependence plot ...")
    #     shap.dependence_plot(
    #         xvar,  # The feature whose SHAP values you want to plot on the y-axis
    #         shap_values=shap_values,  # Use standard shap_values for dependence plot
    #         features=X,
    #         interaction_index=yvar,  # The feature with which you want to see the interaction
    #         x_jitter=0.5  # Add jitter for better visualization of scattered points
    #     )
    # else:
    #     print("Features not found in the dataset for dependence plot example.")

    # shap.summary_plot(shap_values, X, feature_names=X.columns)
#
#     # Top interactions
#     # Calculate the mean absolute interaction values
#     mean_abs_interaction = np.abs(shap_interaction_values).mean(0)
#
#     # Create a DataFrame for better viewing
#     interaction_df = pd.DataFrame(mean_abs_interaction, index=X.columns, columns=X.columns)
#
#     # Set diagonal values (main effects) to NaN or 0 for clearer interaction analysis
#     np.fill_diagonal(interaction_df.values, np.nan)  # Or 0
#
#     # Stack to get pairs and sort
#     sorted_interactions = interaction_df.stack().sort_values(ascending=False)
#
#     print("\nFeature interactions:")
#     # print(sorted_interactions.head(10))
#     print(sorted_interactions)
#
# #     y = np.array(subset[fluxcol])
# #     X = subset[[tacol, vpdcol, swccol, swincol]].copy()
# #     # X = np.array(subset.drop(fluxcol, axis=1))
# #     model = xgb.XGBRegressor()
# #     model = model.fit(X=X, y=y)
# #     explainer = shap.Explainer(model)
# #     shap_values = explainer(X)
# #     # shap_interaction_values = explainer.shap_interaction_values(X)
# #     # shap.interaction_plot(shap_interaction_values[i], X.iloc[[i]], feature_names=X.columns)
# #     # shap.plots.waterfall(shap_values[1])
# #     shap.plots.scatter(shap_values[:, tacol], color=shap_values)
# #     shap.plots.scatter(shap_values[:, vpdcol], color=shap_values)
# #     shap.plots.scatter(shap_values[:, swccol], color=shap_values)
# #     shap.plots.scatter(shap_values[:, swincol], color=shap_values)
# #     shap.plots.bar(shap_values)
# #     # shap.summary_plot(shap_values)
# # # df.to_csv("../OUT/03_siteinfo.csv", index=False)
