from pathlib import Path

import diive as dv
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from common import get_variable_names


def train_xgboost_models_and_shap(settings: dict, siteinfo_df: pd.DataFrame, siteconfig, ix,
                                  modelstxt, conditional=False) -> pd.DataFrame:
    _df = siteinfo_df.copy()

    print(f"\nLoading data for site #{ix + 1} {siteconfig['SITE']} ...")

    if siteconfig['_FILEPATH_PARQUET_SUBSET'] == '-MISSING-':
        siteinfo_df.loc[ix, '_FILEPATH_SHAP_VALUES_STANDARD'] = '-MISSING-'
        return siteinfo_df

    # Load site data
    filepath = siteconfig['_FILEPATH_PARQUET_SUBSET']
    subset = dv.load_parquet(filepath, sanitize_timestamp=False)
    # [print(c) for c in sitedata.columns if "LE" in c];

    # Get variable names for this site
    varnames = get_variable_names(siteconfig)

    print(f"Records: {len(subset)}")

    # Target and features
    # features = [tacol, vpdcol, swincol]
    features = [varnames['ta_var'], varnames['vpd_var'], varnames['swc_var'], varnames['swin_var']]
    X = subset[features]
    # y = subset[varnames['le_var']]
    y = subset[varnames['nee_var']]

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
        file.write(f"SITE: {siteconfig['SITE']} / TARGET: {siteconfig['NEE_VAR']} "
                   f"/ R2: {model.score(X, y):.4f} / RMSE: {rmse:.2f}\n")

    if conditional:
        print("Calculating conditional SHAP values using KernelExplainer...")

        # Create a representative background dataset
        # We use shap.kmeans to select a small, representative sample of 100 instances.
        # This is a good balance between accuracy and computational efficiency.
        background_data = shap.kmeans(X, 100).data

        # Define a robust prediction wrapper function
        # KernelExplainer passes data as a NumPy array, but XGBoost expects a DataFrame
        # with the original column names. This wrapper handles the conversion.
        def predict_wrapper(data_array):
            data_df = pd.DataFrame(data_array, columns=X.columns)
            return model.predict(data_df)

        # Initialize the KernelExplainer
        # We pass the prediction wrapper and the background data.
        explainer = shap.KernelExplainer(predict_wrapper, background_data)
        shap_values = explainer.shap_values(X)
        expected_value = explainer.expected_value
    else:
        # Create SHAP TreeExplainer for the trained XGBoost model and get SHAP values
        print("Calculating SHAP values using TreeExplainer ...")
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
    # shapdf[varnames['le_var']] = y.copy()
    shapdf[varnames['nee_var']] = y.copy()
    # shapdf[f"{varnames['le_var']}_PRED"] = y_pred.copy()
    shapdf[f"{varnames['nee_var']}_PRED"] = y_pred.copy()

    # Merge SHAP values with measured
    merged = pd.concat([X, shapdf], axis=1)

    # from diive.core.plotting.scatter import ScatterXY
    # ScatterXY(x=merged[vpd_var], y=merged[f'{vpd_var}_SHAPVALS'], nbins=20,
    #           binagg='median').plot()

    if not conditional:
        outcol = '_FILEPATH_SHAP_VALUES_STANDARD'
        outpath = 'DIR_DATA_OUT_SHAPVALS_STANDARD'
    else:
        outpath = 'DIR_DATA_OUT_SHAPVALS_CONDITIONAL'
        outcol = '_FILEPATH_SHAP_VALUES_CONDITIONAL'

    outfilepath = dv.save_parquet(
        filename=f"{siteconfig['SITE']}_shap_values",
        data=merged,
        outpath=Path(settings[outpath]))
    print(f"Saved SHAP values to file {outfilepath}.")

    siteinfo_df.loc[ix, outcol] = outfilepath

    return siteinfo_df
