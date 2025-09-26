from pathlib import Path

import diive as dv
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def train_xgboost_models_and_shap(target: str, features: list,
                                  siteconfig, ix, modelstxt, results_outdir: Path, conditional=False) -> None:
    # # TODO testing
    if ix < 40:
        return None

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

    # Train/test data, NOT for final testing, only to guide the training process.
    # Model still trained on the majority (85%) of the data.
    # create a small validation set to guide the early stopping process.
    # The model will train on the majority of the data and use this small,
    # separate slice to know when to stop. The goal of explaining the full dataset
    # is still achieved.
    X_for_training, X_val, y_for_training, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # Initialize XGBoost Regressor
    # objective='reg:squarederror' for standard regression
    # n_estimators: number of boosting rounds (trees)
    # learning_rate: step size shrinkage to prevent overfitting
    # max_depth: maximum depth of a tree
    # Initialize and train the XGBoost Regressor model
    model = xgb.XGBRegressor(objective='reg:squarederror',  # For regression tasks
                             n_estimators=1000,  # Number of boosting rounds
                             learning_rate=0.1,  # Step size shrinkage to prevent overfitting
                             max_depth=6,  # Maximum depth of a tree
                             subsample=.9,  # Subsample ratio of the training instance
                             colsample_bytree=.9,  # Subsample ratio of columns when constructing each tree
                             random_state=42,
                             early_stopping_rounds=50,  # Stop if validation metric doesn't improve for 50 rounds
                             n_jobs=-1)  # Use all available CPU cores

    # Train the model
    print("Training model with early stopping based on a validation set...")
    model.fit(X_for_training, y_for_training, eval_set=[(X_val, y_val)], verbose=False)
    # OLD: model.fit(X, y, eval_set=[(X, y)], verbose=False)

    # Evaluate model performance on the test set
    print("Evaluating model and calculating SHAP for the ENTIRE dataset...")
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"FULL DATASET -> R2: {r2:.4f} / RMSE: {rmse:.2f}")
    with open(modelstxt, 'a') as file:
        file.write(f"SITE: {siteconfig['SITE']} / TARGET: {target} "
                   f"/ R2: {model.score(X, y):.4f} / RMSE: {rmse:.2f}\n")

    if conditional:
        print("Calculating conditional SHAP values using PartitionExplainer...")
        # Background data should represent the data the model was trained on.
        # OLD: background_data = shap.kmeans(X_for_training, 500).data
        # Dynamically determine the number of samples for the background dataset, safeguard for short datasets
        n_background_samples = min(300, len(X_for_training))
        print(f"Using {n_background_samples} samples for SHAP background data...")
        background_data = shap.kmeans(X_for_training, n_background_samples).data
        explainer = shap.PartitionExplainer(model.predict, background_data)
        shap_explanation = explainer(X)  # Explain the ENTIRE dataset
        shap_values = shap_explanation.values
        expected_value = shap_explanation.base_values[0]
    else:
        print("Calculating SHAP values using TreeExplainer ...")
        explainer = shap.TreeExplainer(model)
        # Explain the entire dataset
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
    merged.to_csv(outfilepath.replace('.parquet', '.csv'))

    return None
