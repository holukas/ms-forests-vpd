"""
Train XGBoost model for each site and save SHAP values to file.
"""
from pathlib import Path

import diive as dv
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from diive.core.io.files import load_parquet
from sklearn.metrics import mean_squared_error

import src.files as files
from src.files import read_settings_file

# Load settings
settings = read_settings_file("../config/settings.yaml")

# Load site info
siteinfo_df = files.load_siteinfo(filename="03_siteinfo.csv")

# 1. Writing to a file (overwrites if file exists, creates if not)
modelstxt = Path(settings['DIR_DATA_OUT_SHAPVALS_SITE']) / "1_models_xgboost.txt"

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

with open(modelstxt, 'w') as file:
    file.write("XGBOOST MODELS\n")
    file.write(f"Target: {fluxcol}\n")

df_all = None
_df = siteinfo_df.copy()
for ix, row in _df.iterrows():
    # if ix != 0:
    #     continue

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
    subset = sitedata[[fluxcol, tacol, vpdcol, swccol, swincol]].copy()

    # Convert to z-scores, ignoring NaNs
    # subset[fluxcol] = subset[fluxcol].apply(lambda x: zscore(x, nan_policy='omit'))
    subset[fluxcol] = subset[fluxcol].transform(lambda x: (x - x.mean()) / x.std())
    # subset = subset.apply(lambda x: zscore(x, nan_policy='omit'))

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
    print(f"Records: {len(subset)}")

    # TODO testing: Limit time range
    # subset = subset.loc[subset.index.year == 2019].copy()
    # subset = subset.loc[subset.index.month == 7].copy()

    # Target and features
    # features = [tacol, vpdcol, swincol]
    features = [tacol, vpdcol, swccol, swincol]
    X = subset[features]
    y = subset[fluxcol]

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
    r2 = model.score(X, y)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"R2: {r2:.4f} / RMSE: {rmse:.2f}")
    with open(modelstxt, 'a') as file:
        file.write(f"SITE: {row['SITE']} / R2: {model.score(X, y):.4f} / RMSE: {rmse:.2f}\n")

    # Create SHAP TreeExplainer for the trained XGBoost model and get SHAP values
    explainer = shap.TreeExplainer(model)
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
    shapdf[f'{fluxcol}_PRED'] = y_pred.copy()

    # Merge SHAP values with measured
    merged = pd.concat([X, shapdf], axis=1)

    # from diive.core.plotting.scatter import ScatterXY
    # ScatterXY(x=merged[vpdcol], y=merged[f'{vpdcol}_SHAPVALS'], nbins=20,
    #           binagg='mean').plot()

    outfilepath = dv.save_parquet(
        filename=f"{row['SITE']}_shap_values",
        data=merged,
        outpath=Path(settings['DIR_DATA_OUT_SHAPVALS_SITE']))
    print(f"Saved SHAP values to file {outfilepath}.")
