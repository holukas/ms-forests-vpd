from pathlib import Path

import diive as dv
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def train_xgboost_models_and_shap(
        settings: dict,
        siteinfo_df: pd.DataFrame,
        fluxcol: str,
        fluxqc: str = None,
) -> pd.DataFrame:
    # Writing to a file (overwrites if file exists, creates if not)
    modelstxt = Path(settings['DIR_DATA_OUT_SHAPVALS_SITE']) / "1_models_xgboost.txt"
    with open(modelstxt, 'w') as file:
        file.write("XGBOOST MODELS\n")
        file.write(f"Target: {fluxcol}\n")

    _df = siteinfo_df.copy()
    for ix, site in _df.iterrows():

        # if site['SITE'] != 'DK-RCW':
        #     continue

        print(f"\nLoading data for site #{ix + 1} {site['SITE']} ...")

        # Load site data
        filepath = site['_FILEPATH_PARQUET']
        sitedata = dv.load_parquet(filepath)
        # [print(c) for c in sitedata.columns if "LE" in c];

        # Get variable names for this site
        swin_var = site['SWIN_VAR']
        ta_var = site['TA_VAR']
        vpd_var = site['VPD_VAR']
        swc_var = site['SWC_VAR']
        swinpot_var = 'SW_IN_POT'

        if swc_var == '-MISSING-':
            siteinfo_df.loc[ix, '_FILEPATH_SHAP_VALUES'] = '-MISSING-'
            continue

        # Detect 6 warmest months
        ta = sitedata[[ta_var]].copy()
        ta['MONTH'] = ta.index.month
        monthly_avg = ta.groupby('MONTH').mean()
        monthly_avg = monthly_avg.sort_values(by=ta_var, ascending=False, inplace=False)
        warmest6 = monthly_avg.head(6).index.to_list()
        # locs = sitedata.index.month.isin(warmest6)

        # Make subset
        subset = sitedata[[fluxcol, ta_var, vpd_var, swc_var, swin_var]].copy()

        # Convert target to z-scores, ignoring NaNs
        # subset[fluxcol] = subset[fluxcol].apply(lambda x: zscore(x, nan_policy='omit'))
        subset[fluxcol] = subset[fluxcol].transform(lambda x: (x - x.mean()) / x.std())
        # subset = subset.apply(lambda x: zscore(x, nan_policy='omit'))

        # Add original flux quality flag and potential radiation
        if fluxqc is not None:
            subset[fluxqc] = sitedata[fluxqc].copy()
        subset[swinpot_var] = sitedata[swinpot_var].copy()

        # Keep directly measured fluxes, no gap-filled data
        if fluxqc is not None:
            subset = subset.loc[subset[fluxqc] == 0].copy()

        # Filter subset
        # Keep data from 6 warmest months
        # Keep daytime records
        # Keep required variables only
        subset = subset.loc[
            subset.index.month.isin(warmest6) &
            (subset[swinpot_var] > 20),
            [fluxcol, ta_var, vpd_var, swin_var, swc_var]
        ].copy()

        # Keep records where all vars available
        subset = subset.dropna()

        print(f"Records: {len(subset)}")

        # TODO testing: Limit time range
        # subset = subset.loc[subset.index.year == 2019].copy()
        subset = subset.loc[subset.index.month == 7].copy()
        # TODO testing: Limit time range

        # Target and features
        # features = [tacol, vpdcol, swincol]
        features = [ta_var, vpd_var, swc_var, swin_var]
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
            file.write(f"SITE: {site['SITE']} / R2: {model.score(X, y):.4f} / RMSE: {rmse:.2f}\n")

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
            filename=f"{site['SITE']}_shap_values",
            data=merged,
            outpath=Path(settings['DIR_DATA_OUT_SHAPVALS_SITE']))
        print(f"Saved SHAP values to file {outfilepath}.")
        siteinfo_df.loc[ix, '_FILEPATH_SHAP_VALUES'] = outfilepath

    return siteinfo_df
