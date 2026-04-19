from pathlib import Path

import diive as dv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from PyALE import ale
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def train_rf_models_and_shap(target: str, features: list,
                             siteconfig, ix, modelstxt, results_outdir: Path, conditional=False) -> None:
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

    # ------------------------------
    # RANDOM FOREST SETTINGS
    # ------------------------------
    # No train/test split needed, RF does not use early stopping
    # Train on entire dataset to maximize explanatory power for SHAP

    model = RandomForestRegressor(
        n_estimators=1000,  # Number of trees (1000 provides very stable SHAP values)
        max_depth=None,  # Limit depth to prevent overfitting on just 4 features
        min_samples_split=2,
        min_samples_leaf=1,  # (Equivalent to min_child_weight) Prevents splitting on outliers
        max_features=0.75,  # (Equivalent to colsample_bytree) Uses 3 out of 4 features per tree
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )

    # Train the model on the ENTIRE dataset
    print("Training Random Forest model on the entire dataset...")
    model.fit(X, y)

    # Evaluate explanatory power on the full dataset
    print("Generating predictions and evaluating explanatory power...")
    y_pred = model.predict(X)
    r2_full = model.score(X, y)
    rmse_full = np.sqrt(mean_squared_error(y, y_pred))
    print(f"FULL DATASET (Explanatory Power) -> R2: {r2_full:.4f} / RMSE: {rmse_full:.2f}")

    # Log performance on full dataset
    with open(modelstxt, 'a') as file:
        file.write(f"SITE: {siteconfig['SITE']} / TARGET: {target} "
                   f"/ R2_full: {r2_full:.4f} / RMSE_full: {rmse_full:.2f} [Explanatory fit on full dataset]\n")

    if conditional:
        print("Calculating CONDITIONAL SHAP values using TreeExplainer...")
        # Conditional SHAP: Respects the correlation between features.
        # TreeExplainer natively does this by default no background dataset
        # is provided. (it uses feature_perturbation="tree_path_dependent").
        # Tree_path_dependent is the default and calculates conditional SHAP
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_explanation = explainer(X)
        shap_values = shap_explanation.values
        expected_value = shap_explanation.base_values
    else:
        # For standard/marginal SHAP, you need a background dataset
        print("Calculating STANDARD (Marginal) SHAP values using TreeExplainer...")
        n_background_samples = min(300, len(X))
        background_data = shap.sample(X, n_background_samples)
        explainer = shap.TreeExplainer(model, data=background_data, feature_perturbation="interventional")
        shap_explanation = explainer(X)
        shap_values = shap_explanation.values
        expected_value = shap_explanation.base_values

    # Collect SHAP values in dataframe
    shapcols = [f'{c}_SHAPVALS' for c in X]
    shapdf = pd.DataFrame(data=shap_values, index=X.index, columns=shapcols)
    shapdf.index = pd.to_datetime(shapdf.index)
    shapdf['SUM'] = shapdf.sum(axis=1)

    # Handle base_values array vs scalar
    if isinstance(expected_value, np.ndarray) and len(expected_value) == len(X):
        shapdf['EXPECTED'] = expected_value
    else:
        # If it returns a scalar or single-item array, broadcast it
        shapdf['EXPECTED'] = float(expected_value[0]) if isinstance(expected_value,
                                                                    (list, np.ndarray)) else expected_value

    shapdf['SUM+EXPECTED'] = shapdf['EXPECTED'].add(shapdf['SUM'])
    shapdf[target] = y.copy()
    shapdf[f"{target}_PRED"] = y_pred.copy()

    # Merge SHAP values with measured
    merged = pd.concat([X, shapdf], axis=1)

    # Add additional columns (e.g., other fluxes)
    current_cols = merged.columns.tolist()
    available_cols = subset.columns.tolist()
    additional_cols = [col for col in available_cols if col not in current_cols]
    for addcol in additional_cols:
        print(f"Adding column {addcol} to SHAP dataframe.")
        merged[addcol] = subset[addcol].copy()
    merged = merged.sort_index(axis=1)

    substr = "conditional" if conditional else "standard"
    outfilepath = dv.save_parquet(
        filename=f"{siteconfig['SITE']}_shap-{substr}_{target}",
        data=merged,
        outpath=results_outdir)
    print(f"Saved SHAP values to file {outfilepath}.")
    merged.to_csv(outfilepath.replace('.parquet', '.csv'))

    return None


def train_xgboost_models_and_shap(target: str, features: list,
                                  siteconfig, ix, modelstxt, results_outdir: Path, conditional=False) -> None:
    # # TODO testing
    # if ix != 141:
    #     return None

    print(f"\nLoading data for site #{ix + 1} {siteconfig['SITE']} ...")

    if siteconfig['_FILEPATH_PARQUET_SUBSET'] == '-MISSING-':
        return None

    # Load site data
    filepath = siteconfig['_FILEPATH_PARQUET_SUBSET']
    subset = dv.load_parquet(filepath, sanitize_timestamp=False)
    # subset = subset.head(100)  # todo deactivate, for testing only
    print(f"Records: {len(subset)}")

    # Target and features
    X = subset[features].copy()
    y = subset[target].copy()

    # # Train/test data, NOT for final testing, only to guide the training process.
    # # Model still trained on the majority (85%) of the data.
    # # Create small validation set to guide the early stopping process.
    # # The model will train on the majority of the data and use this small,
    # # separate slice to know when to stop. The goal of explaining the full dataset
    # # is still achieved.
    # X_for_training, X_val, y_for_training, y_val = train_test_split(
    #     X, y, test_size=0.15, random_state=42
    # )

    # ---------------------------------------------------------
    # 1. REPRESENTATIVE SPLIT (Best for Explainable AI)
    # ---------------------------------------------------------
    # Because the goal is explanation (SHAP) rather than future forecasting,
    # a random split is used. This ensures the 15% validation set contains a
    # representative mix of all months and climate extremes, allowing early
    # stopping to find the true global optimum for the entire dataset.
    X_for_training, X_val, y_for_training, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # ------------------------------
    # XGBOOST SETTINGS
    # ------------------------------
    # Initialize XGBoost Regressor
    # objective='reg:squarederror' for standard regression
    model = xgb.XGBRegressor(
        objective='reg:squarederror',  # For regression tasks
        n_estimators=3000,  # Increased to allow the lower learning rate to work
        max_depth=6,  # Depth of tree, lowered because you only have 4 features
        learning_rate=0.05,  # Lowered for better generalization (smoother SHAP)
        subsample=0.8,  # Subsample ratio of training instance, slightly lower to increase randomness/robustness
        # Subsample ratio of columns when constructing each tree, randomly selects ~3 out of 4 features per tree
        colsample_bytree=0.8,
        min_child_weight=5,  # Prevents splitting on single outliers
        random_state=42,
        early_stopping_rounds=100,  # Increased patience for the lower learning rate
        # n_jobs=0
        n_jobs=-1
    )  # Use all available CPU cores

    # Train the model
    print("Training model with early stopping based on random validation set...")
    model.fit(X_for_training, y_for_training, eval_set=[(X_val, y_val)], verbose=False)

    # Generate predictions for the ENTIRE dataset
    print("Generating predictions and calculating SHAP for the ENTIRE dataset...")
    y_pred_full = model.predict(X)

    # Calculate EXPLANATORY POWER on the full dataset (matching the SHAP values)
    r2_full = model.score(X, y)
    rmse_full = np.sqrt(mean_squared_error(y, y_pred_full))
    print(f"FULL DATASET (Explanatory Power) -> R2: {r2_full:.4f} / RMSE: {rmse_full:.2f}")

    # # Evaluate true generalization on the validation set
    # y_pred_val = model.predict(X_val)
    # r2_val = model.score(X_val, y_val)
    # rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    # print(f"VALIDATION DATASET -> R2: {r2_val:.4f} / RMSE: {rmse_val:.2f}")

    # Log the full-dataset performance
    with open(modelstxt, 'a') as file:
        file.write(f"SITE: {siteconfig['SITE']} / TARGET: {target} "
                   f"/ R2_full: {r2_full:.4f} / RMSE_full: {rmse_full:.2f} [Explanatory fit on full dataset]\n")

    # # Log true performance
    # with open(modelstxt, 'a') as file:
    #     file.write(f"SITE: {siteconfig['SITE']} / TARGET: {target} "
    #                f"/ R2: {r2_val:.4f} / RMSE: {rmse_val:.2f} [Evaluation on validation set]\n")

    # # Predict for the entire dataset
    # print("Generating predictions and calculating SHAP for the ENTIRE dataset...")
    # y_pred = model.predict(X)

    if conditional:
        print("Calculating CONDITIONAL SHAP values using TreeExplainer...")
        # Conditional SHAP: Respects the correlation between features.
        # TreeExplainer natively does this by default no background dataset
        # is provided. (it uses feature_perturbation="tree_path_dependent").
        # Tree_path_dependent is the default and calculates conditional SHAP
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        shap_explanation = explainer(X)
        shap_values = shap_explanation.values
        expected_value = shap_explanation.base_values

        # # Previously used code:
        # print("Calculating conditional SHAP values using PartitionExplainer...")
        # # Background data should represent the data the model was trained on.
        # # Dynamically determine the number of samples for the background dataset, safeguard for short datasets
        # n_background_samples = min(300, len(X_for_training))
        # print(f"Using {n_background_samples} samples for SHAP background data...")
        # background_data = shap.kmeans(X_for_training, n_background_samples).data
        # explainer = shap.PartitionExplainer(model.predict, background_data)
        # shap_explanation = explainer(X)  # Explain the ENTIRE dataset
        # shap_values = shap_explanation.values
        # # expected_value = shap_explanation.base_values[0]
        # expected_value = shap_explanation.base_values
    else:
        print("Calculating STANDARD (Marginal) SHAP values using TreeExplainer...")
        # For standard/marginal SHAP, you need a background dataset
        n_background_samples = min(300, len(X_for_training))
        # shap.sample is generally preferred over shap.kmeans for tree models
        background_data = shap.sample(X_for_training, n_background_samples)
        explainer = shap.TreeExplainer(model, data=background_data, feature_perturbation="interventional")
        shap_explanation = explainer(X)
        shap_values = shap_explanation.values
        expected_value = shap_explanation.base_values

        # # Previously used code:
        # print("Calculating SHAP values using TreeExplainer ...")
        # explainer = shap.TreeExplainer(model)
        # # Explain the entire dataset
        # shap_values = explainer.shap_values(X)
        # expected_value = explainer.expected_value

    # Collect SHAP values in dataframe
    shapcols = [f'{c}_SHAPVALS' for c in X]
    shapdf = pd.DataFrame(data=shap_values, index=X.index, columns=shapcols)
    shapdf.index = pd.to_datetime(shapdf.index)
    shapdf['SUM'] = shapdf.sum(axis=1)

    # SAFETY CHECK
    if isinstance(expected_value, np.ndarray) and len(expected_value) == len(X):
        shapdf['EXPECTED'] = expected_value
    else:
        # If it returns a scalar or single-item array, broadcast it
        shapdf['EXPECTED'] = float(expected_value[0]) if isinstance(expected_value,
                                                                    (list, np.ndarray)) else expected_value

    shapdf['SUM+EXPECTED'] = shapdf['EXPECTED'].add(shapdf['SUM'])
    shapdf[target] = y.copy()
    shapdf[f"{target}_PRED"] = y_pred_full.copy()

    # Merge SHAP values with measured
    merged = pd.concat([X, shapdf], axis=1)

    # Add additional columns (e.g., other fluxes)
    current_cols = merged.columns.tolist()
    available_cols = subset.columns.tolist()
    additional_cols = [col for col in available_cols if col not in current_cols]
    for addcol in additional_cols:
        print(f"Adding column {addcol} to SHAP dataframe.")
        merged[addcol] = subset[addcol].copy()
    merged = merged.sort_index(axis=1)

    substr = "conditional" if conditional else "standard"
    outfilepath = dv.save_parquet(
        filename=f"{siteconfig['SITE']}_shap-{substr}_{target}",
        data=merged,
        outpath=results_outdir)
    print(f"Saved SHAP values to file {outfilepath}.")
    merged.to_csv(outfilepath.replace('.parquet', '.csv'))

    return None


def train_xgboost_models_and_ale(target: str, features: list,
                                 siteconfig, ix, modelstxt, results_outdir: Path) -> None:
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

    # Train/test split for model training
    X_for_training, X_val, y_for_training, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # Train XGBoost model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=3000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        early_stopping_rounds=100,
        n_jobs=-1
    )

    print("Training model with early stopping based on random validation set...")
    model.fit(X_for_training, y_for_training, eval_set=[(X_val, y_val)], verbose=False)

    # Generate predictions for the ENTIRE dataset
    print("Generating predictions and calculating ALE for the ENTIRE dataset...")
    y_pred_full = model.predict(X)

    # Calculate EXPLANATORY POWER on the full dataset
    r2_full = model.score(X, y)
    rmse_full = np.sqrt(mean_squared_error(y, y_pred_full))
    print(f"FULL DATASET (Explanatory Power) -> R2: {r2_full:.4f} / RMSE: {rmse_full:.2f}")

    # Log the full-dataset performance
    with open(modelstxt, 'a') as file:
        file.write(f"SITE: {siteconfig['SITE']} / TARGET: {target} "
                   f"/ R2_full: {r2_full:.4f} / RMSE_full: {rmse_full:.2f}\n")

    # Calculate and plot ALE for each feature
    print("Calculating ALE (Accumulated Local Effects) for each feature...")
    ale_data_all_features = {}

    for feature in features:
        print(f"  Calculating ALE for {feature}...")
        fig, ax = plt.subplots(figsize=(10, 6))
        ale(X=X, model=model, feature=[feature], grid_size=100)
        plt.title(f'ALE Plot: Effect of {feature} on {target}\nSite: {siteconfig["SITE"]}')
        plt.tight_layout()

        # Extract line data from the plot
        ax = plt.gca()
        ale_values = []
        for line in ax.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            if len(xdata) > 0:
                ale_values.append({'feature_value': xdata, 'effect': ydata})

        # Save ALE curve data if available
        if ale_values:
            ale_curve_data = pd.DataFrame({
                'feature_value': ale_values[0]['feature_value'],
                'effect': ale_values[0]['effect'],
                'feature': feature,
                'site': siteconfig['SITE'],
                'target': target
            })
            ale_data_all_features[feature] = ale_curve_data

            # Save individual feature ALE data
            ale_csv_filename = f"{siteconfig['SITE']}_ale_curve_{feature}_{target}.csv"
            ale_curve_data.to_csv(results_outdir / ale_csv_filename, index=False)
            print(f"    Saved ALE curve data to {ale_csv_filename}")

        # Save ALE plot
        plot_filename = f"{siteconfig['SITE']}_ale_{feature}_{target}.png"
        plt.savefig(results_outdir / plot_filename, dpi=150)
        plt.close()
        print(f"    Saved plot to {plot_filename}")

    # Create output dataframe with predictions
    ale_df = X.copy()
    ale_df.index = pd.to_datetime(ale_df.index)
    ale_df[target] = y.copy()
    ale_df[f"{target}_PRED"] = y_pred_full.copy()

    # Add additional columns from the original subset
    current_cols = ale_df.columns.tolist()
    available_cols = subset.columns.tolist()
    additional_cols = [col for col in available_cols if col not in current_cols]
    for addcol in additional_cols:
        print(f"Adding column {addcol} to ALE dataframe.")
        ale_df[addcol] = subset[addcol].copy()
    ale_df = ale_df.sort_index(axis=1)

    # Save results
    outfilepath = dv.save_parquet(
        filename=f"{siteconfig['SITE']}_ale_{target}",
        data=ale_df,
        outpath=results_outdir)
    print(f"Saved ALE results to file {outfilepath}.")
    ale_df.to_csv(outfilepath.replace('.parquet', '.csv'))

    return None
