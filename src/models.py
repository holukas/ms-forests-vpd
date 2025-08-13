from pathlib import Path

import diive as dv
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def train_xgboost_models_and_shap(target: str, features: list,
                                  siteconfig, ix, modelstxt, results_outdir: Path, conditional=False) -> None:
    # # TODO testing
    # if ix > 5:
    #     return None

    print(f"\nLoading data for site #{ix + 1} {siteconfig['SITE']} ...")

    if siteconfig['_FILEPATH_PARQUET_SUBSET'] == '-MISSING-':
        return None

    # Load site data
    filepath = siteconfig['_FILEPATH_PARQUET_SUBSET']
    subset = dv.load_parquet(filepath, sanitize_timestamp=False)
    print(f"Records: {len(subset)}")

    # Target and features
    X = subset[features].copy()
    y = subset[target].copy()

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
        file.write(f"SITE: {siteconfig['SITE']} / TARGET: {target} "
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
    shapdf[target] = y.copy()
    shapdf[f"{target}_PRED"] = y_pred.copy()

    # Merge SHAP values with measured
    merged = pd.concat([X, shapdf], axis=1)

    # from diive.core.plotting.scatter import ScatterXY
    # ScatterXY(x=merged[vpd_var], y=merged[f'{vpd_var}_SHAPVALS'], nbins=20,
    #           binagg='median').plot()

    substr = "conditional" if conditional else "standard"
    outfilepath = dv.save_parquet(
        filename=f"{siteconfig['SITE']}_shap-{substr}_{target}",
        data=merged,
        outpath=results_outdir)
    print(f"Saved SHAP values to file {outfilepath}.")
    # siteinfo_df.loc[ix, outcol] = outfilepath

    return None
