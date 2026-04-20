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
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


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
    with open(modelstxt, 'a', encoding='utf-8') as file:
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
    with open(modelstxt, 'a', encoding='utf-8') as file:
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
    with open(modelstxt, 'a', encoding='utf-8') as file:
        file.write(f"SITE: {siteconfig['SITE']} / TARGET: {target} "
                   f"/ R2_full: {r2_full:.4f} / RMSE_full: {rmse_full:.2f}\n")

    # Step 1: Calculate and save ALE curve data for each feature
    print("Calculating ALE (Accumulated Local Effects) for each feature...")
    ale_data_all_features = {}

    for feature in features:
        print(f"  Calculating ALE for {feature}...")

        # Call ale() which creates its own figure
        ale(X=X, model=model, feature=[feature], grid_size=100)

        # Extract line data from the CURRENT figure (created by ale())
        current_fig = plt.gcf()  # Get current figure
        ax_temp = plt.gca()  # Get current axes
        ale_values = []

        # Extract lines from the plot
        for line in ax_temp.get_lines():
            xdata = np.array(line.get_xdata())
            ydata = np.array(line.get_ydata())
            if len(xdata) > 0:
                ale_values.append({'feature_value': xdata, 'effect': ydata})

        # PyALE creates multiple lines: first is the ALE curve (grid_size points),
        # others are optional overlays. We want the FIRST line which has ~grid_size points.
        ale_curve = None
        if len(ale_values) > 0:
            # Get the first line (smallest number of points, which is the ALE curve)
            ale_curve = min(ale_values, key=lambda x: len(x['feature_value']))
            print(f"    Extracted ALE curve with {len(ale_curve['feature_value'])} grid points for {feature}")

        # Close the ale() figure
        plt.close(current_fig)

        # Save ALE curve data if available
        if ale_curve is not None:
            ale_curve_data = pd.DataFrame({
                'feature_value': ale_curve['feature_value'],
                'effect': ale_curve['effect'],
                'feature': feature,
                'site': siteconfig['SITE'],
                'target': target
            })
            ale_data_all_features[feature] = ale_curve_data
            print(f"    Stored ALE data for {feature}")
        else:
            print(f"    WARNING: No ALE curve extracted for {feature}")

    # Step 2: Create combined subplot figure by reading the saved parquet files
    if ale_data_all_features:
        print(f"  Creating combined ALE plot from saved data...")

        # Create subplot grid (2x2 for 4 features)
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10))

        # Flatten axes for easier indexing
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)

        fig.suptitle(f'ALE (Accumulated Local Effects) for {target}\nSite: {siteconfig["SITE"]}',
                     fontsize=14, fontweight='bold', y=0.995)

        # Plot each feature from the saved data
        for idx, feature in enumerate(features):
            if feature in ale_data_all_features:
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col]

                # Get data from saved file
                ale_data = ale_data_all_features[feature]
                feature_values = ale_data['feature_value'].values
                effects = ale_data['effect'].values

                # Plot
                ax.plot(feature_values, effects, linewidth=2.5, color='steelblue')
                ax.fill_between(feature_values, effects, alpha=0.25, color='steelblue')
                ax.set_xlabel(feature, fontweight='bold', fontsize=10)
                ax.set_ylabel('ALE Effect', fontsize=10)
                ax.set_title(f'{feature}', fontweight='bold', fontsize=11)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        # Remove empty subplots
        for idx in range(n_features, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])

        # Save combined plot
        plt.tight_layout()
        combined_plot_filename = f"{siteconfig['SITE']}_ale_combined_{target}.png"
        plt.savefig(results_outdir / combined_plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved combined ALE plot to {combined_plot_filename}")

        # Save consolidated ALE curves (all features in one file)
        consolidated_ale = pd.concat(ale_data_all_features.values(), ignore_index=True)

        # Add metadata for aggregation across sites and IGBP
        consolidated_ale['IGBP'] = siteconfig.get('IGBP', '-MISSING-')
        consolidated_ale['LAT'] = siteconfig.get('LAT', np.nan)
        consolidated_ale['LON'] = siteconfig.get('LON', np.nan)
        consolidated_ale['ELEVATION'] = siteconfig.get('ELEVATION', np.nan)
        consolidated_ale['N_RECORDS'] = siteconfig.get('N_RECORDS', np.nan)

        consolidated_filename_csv = f"{siteconfig['SITE']}_ale_curves_{target}.csv"
        consolidated_filename_parquet = f"{siteconfig['SITE']}_ale_curves_{target}.parquet"
        consolidated_ale.to_csv(results_outdir / consolidated_filename_csv, index=False)
        consolidated_ale.to_parquet(results_outdir / consolidated_filename_parquet, index=False)
        print(f"  Saved consolidated ALE curves to {consolidated_filename_csv} and {consolidated_filename_parquet}")

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


def calculate_partial_correlations(target: str, features: list,
                                   siteconfig, ix, modelstxt, results_outdir: Path) -> None:
    """
    Calculate partial correlations between target and each feature,
    controlling for all other features.

    This validates SHAP importance by showing the isolated effect of each
    feature on the target, accounting for confounding variables.
    """
    print(f"\nCalculating Partial Correlations for site #{ix + 1} {siteconfig['SITE']} ...")

    if siteconfig['_FILEPATH_PARQUET_SUBSET'] == '-MISSING-':
        return None

    # Load site data
    filepath = siteconfig['_FILEPATH_PARQUET_SUBSET']
    subset = dv.load_parquet(filepath, sanitize_timestamp=False)
    print(f"Records: {len(subset)}")

    # Target and features
    X = subset[features].copy()
    y = subset[target].copy()

    # Remove NaN values
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]
    print(f"Valid records after removing NaN: {len(X)}")

    # Store results
    partial_corr_results = []

    # For each feature, calculate partial correlation
    for target_feature in features:
        # Get other features (for controlling)
        control_features = [f for f in features if f != target_feature]

        # Residualize target
        if len(control_features) > 0:
            X_control = X[control_features]
            model_target = LinearRegression()
            model_target.fit(X_control, y)
            y_resid = y - model_target.predict(X_control)
        else:
            y_resid = y

        # Residualize feature
        X_feature = X[[target_feature]]
        if len(control_features) > 0:
            model_feature = LinearRegression()
            model_feature.fit(X_control, X_feature)
            X_feature_resid = X_feature.values.flatten() - model_feature.predict(X_control).flatten()
        else:
            X_feature_resid = X_feature.values.flatten()

        # Calculate correlation between residuals
        corr_coef, p_value = pearsonr(X_feature_resid, y_resid)

        partial_corr_results.append({
            'SITE': siteconfig['SITE'],
            'TARGET': target,
            'FEATURE': target_feature,
            'CONTROL_VARIABLES': ', '.join(control_features),
            'PARTIAL_CORRELATION': corr_coef,
            'P_VALUE': p_value,
            'SIGNIFICANT': 'YES' if p_value < 0.05 else 'NO',
            'IGBP': siteconfig.get('IGBP', '-MISSING-'),
            'LAT': siteconfig.get('LAT', np.nan),
            'LON': siteconfig.get('LON', np.nan),
            'ELEVATION': siteconfig.get('ELEVATION', np.nan),
            'N_RECORDS': siteconfig.get('N_RECORDS', np.nan)
        })

        print(f"  {target_feature}: r={corr_coef:.4f}, p={p_value:.2e}")

    # Save results
    partial_corr_df = pd.DataFrame(partial_corr_results)
    outfile = results_outdir / f"{siteconfig['SITE']}_partial_correlations_{target}.csv"
    partial_corr_df.to_csv(outfile, index=False)
    print(f"Saved partial correlations to {outfile}")

    # Log summary
    with open(modelstxt, 'a', encoding='utf-8') as file:
        file.write(f"\nPARTIAL CORRELATIONS - SITE: {siteconfig['SITE']} / TARGET: {target}\n")
        for _, row in partial_corr_df.iterrows():
            file.write(f"  {row['FEATURE']}: r={row['PARTIAL_CORRELATION']:.4f} (p={row['P_VALUE']:.2e})\n")

    return None


def calculate_path_analysis(target: str, features: list,
                           siteconfig, ix, modelstxt, results_outdir: Path) -> None:
    """
    Calculate path coefficients (standardized regression coefficients) to test
    direct and indirect effects of predictors on target.

    This implements a simplified path analysis where we calculate how much each
    feature directly predicts the target (path coefficient) and the strength of
    correlations between features (mediator paths).
    """
    print(f"\nCalculating Path Analysis for site #{ix + 1} {siteconfig['SITE']} ...")

    if siteconfig['_FILEPATH_PARQUET_SUBSET'] == '-MISSING-':
        return None

    # Load site data
    filepath = siteconfig['_FILEPATH_PARQUET_SUBSET']
    subset = dv.load_parquet(filepath, sanitize_timestamp=False)
    print(f"Records: {len(subset)}")

    # Target and features
    X = subset[features].copy()
    y = subset[target].copy()

    # Remove NaN values
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]
    print(f"Valid records after removing NaN: {len(X)}")

    # Standardize variables for path analysis
    X_std = (X - X.mean()) / X.std()
    y_std = (y - y.mean()) / y.std()

    # Direct effects (path coefficients): regression coefficients
    model = LinearRegression()
    model.fit(X_std, y_std)

    # Calculate R-squared for model fit
    y_pred = model.predict(X_std)
    ss_res = np.sum((y_std - y_pred) ** 2)
    ss_tot = np.sum((y_std - y_std.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Store direct path effects
    path_results = []
    for feature, coef in zip(features, model.coef_):
        path_results.append({
            'SITE': siteconfig['SITE'],
            'TARGET': target,
            'FEATURE': feature,
            'PATH_TYPE': 'DIRECT',
            'PATH_COEFFICIENT': coef,
            'ABS_COEFFICIENT': abs(coef),
            'IGBP': siteconfig.get('IGBP', '-MISSING-'),
            'LAT': siteconfig.get('LAT', np.nan),
            'LON': siteconfig.get('LON', np.nan),
            'ELEVATION': siteconfig.get('ELEVATION', np.nan),
            'N_RECORDS': siteconfig.get('N_RECORDS', np.nan)
        })
        print(f"  Direct effect {feature} -> {target}: coef={coef:.4f}")

    # Correlations between features (potential mediation paths)
    print(f"  Feature intercorrelations (mediation potential):")
    for i, feat_i in enumerate(features):
        for feat_j in features[i+1:]:
            corr, _ = pearsonr(X[feat_i], X[feat_j])
            path_results.append({
                'SITE': siteconfig['SITE'],
                'TARGET': target,
                'FEATURE': f"{feat_i} <-> {feat_j}",
                'PATH_TYPE': 'MEDIATION',
                'PATH_COEFFICIENT': corr,
                'ABS_COEFFICIENT': abs(corr),
                'IGBP': siteconfig.get('IGBP', '-MISSING-'),
                'LAT': siteconfig.get('LAT', np.nan),
                'LON': siteconfig.get('LON', np.nan),
                'ELEVATION': siteconfig.get('ELEVATION', np.nan),
                'N_RECORDS': siteconfig.get('N_RECORDS', np.nan)
            })
            print(f"    {feat_i} <-> {feat_j}: r={corr:.4f}")

    # Save results
    path_df = pd.DataFrame(path_results)
    outfile = results_outdir / f"{siteconfig['SITE']}_path_analysis_{target}.csv"
    path_df.to_csv(outfile, index=False)
    print(f"Saved path analysis to {outfile}")

    # Log summary (use UTF-8 encoding for special characters)
    with open(modelstxt, 'a', encoding='utf-8') as file:
        file.write(f"\nPATH ANALYSIS (STANDARDIZED COEFFICIENTS) - SITE: {siteconfig['SITE']} / TARGET: {target}\n")
        file.write(f"Model R-squared: {r_squared:.4f}\n")
        file.write(f"Direct Effects (path coefficients):\n")
        for _, row in path_df[path_df['PATH_TYPE'] == 'DIRECT'].iterrows():
            file.write(f"  {row['FEATURE']} -> {target}: coef={row['PATH_COEFFICIENT']:.4f}\n")

    return None


def create_validation_summary(target: str, features: list,
                             siteconfig, ix, results_outdir: Path) -> None:
    """
    Create a comprehensive summary integrating ALE, Partial Correlations, and Path Analysis.

    Combines all three validation methods to:
    1. Identify which variables are most important
    2. Interpret what each variable shows
    3. Check agreement between methods
    """
    site = siteconfig['SITE']

    # Load the three output files
    ale_file = results_outdir / f"{site}_ale_curves_{target}.csv"
    pc_file = results_outdir / f"{site}_partial_correlations_{target}.csv"
    pa_file = results_outdir / f"{site}_path_analysis_{target}.csv"

    if not all([ale_file.exists(), pc_file.exists(), pa_file.exists()]):
        print(f"  WARNING: Missing output files for {site}, skipping summary")
        return None

    ale_df = pd.read_csv(ale_file)
    pc_df = pd.read_csv(pc_file)
    pa_df = pd.read_csv(pa_file)

    # Create summary file
    summary_file = results_outdir / f"{site}_summary_{target}.txt"

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"INTEGRATED VALIDATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Site: {site}\n")
        f.write(f"Target: {target}\n")
        f.write(f"IGBP: {siteconfig.get('IGBP', 'UNKNOWN')}\n")
        f.write(f"Location: LAT {siteconfig.get('LAT', 'N/A')}, LON {siteconfig.get('LON', 'N/A')}\n")
        f.write(f"Records: {siteconfig.get('N_RECORDS', 'N/A')}\n\n")

        f.write("=" * 80 + "\n")
        f.write("FEATURE IMPORTANCE RANKING\n")
        f.write("=" * 80 + "\n\n")

        # Rank features by importance across methods
        feature_scores = {}
        for feature in features:
            # ALE: max absolute effect
            ale_feature = ale_df[ale_df['feature'] == feature]
            ale_range = ale_feature['effect'].max() - ale_feature['effect'].min()

            # Partial Correlation: absolute correlation
            pc_feature = pc_df[pc_df['FEATURE'] == feature]
            pc_corr = abs(pc_feature['PARTIAL_CORRELATION'].values[0]) if len(pc_feature) > 0 else 0

            # Path Analysis: absolute direct coefficient
            pa_feature = pa_df[(pa_df['FEATURE'] == feature) & (pa_df['PATH_TYPE'] == 'DIRECT')]
            pa_coef = abs(pa_feature['PATH_COEFFICIENT'].values[0]) if len(pa_feature) > 0 else 0

            # Combined score (normalize and average)
            feature_scores[feature] = {
                'ale_range': ale_range,
                'pc_corr': pc_corr,
                'pa_coef': pa_coef,
                'mean_score': np.mean([ale_range, pc_corr, pa_coef])
            }

        # Sort by importance
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1]['mean_score'], reverse=True)

        for rank, (feature, scores) in enumerate(sorted_features, 1):
            f.write(f"{rank}. {feature}\n")
            f.write(f"   - ALE effect range: {scores['ale_range']:.4f}\n")
            f.write(f"   - Partial correlation (|r|): {scores['pc_corr']:.4f}\n")
            f.write(f"   - Path coefficient (|coef|): {scores['pa_coef']:.4f}\n")
            f.write(f"   - Combined importance score: {scores['mean_score']:.4f}\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED FEATURE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        for feature in features:
            f.write(f"\n{'-' * 80}\n")
            f.write(f"FEATURE: {feature}\n")
            f.write(f"{'-' * 80}\n\n")

            # ALE Analysis
            f.write("1. ACCUMULATED LOCAL EFFECTS (ALE)\n")
            ale_feature = ale_df[ale_df['feature'] == feature]
            if len(ale_feature) > 0:
                min_effect = ale_feature['effect'].min()
                max_effect = ale_feature['effect'].max()
                mean_effect = ale_feature['effect'].mean()
                f.write(f"   Effect Range: [{min_effect:.4f}, {max_effect:.4f}]\n")
                f.write(f"   Mean Effect: {mean_effect:.4f}\n")
                f.write(f"   Interpretation: ")
                if abs(max_effect - min_effect) > 0.5:
                    f.write(f"STRONG effect on {target}. ")
                elif abs(max_effect - min_effect) > 0.2:
                    f.write(f"MODERATE effect on {target}. ")
                else:
                    f.write(f"WEAK effect on {target}. ")

                if max_effect > abs(min_effect):
                    f.write(f"Higher {feature} values increase {target}.\n")
                else:
                    f.write(f"Lower {feature} values increase {target}.\n")

            # Partial Correlation
            f.write("\n2. PARTIAL CORRELATION\n")
            pc_feature = pc_df[pc_df['FEATURE'] == feature]
            if len(pc_feature) > 0:
                pc_r = pc_feature['PARTIAL_CORRELATION'].values[0]
                pc_p = pc_feature['P_VALUE'].values[0]
                pc_sig = pc_feature['SIGNIFICANT'].values[0]
                f.write(f"   Correlation (r): {pc_r:.4f}\n")
                f.write(f"   P-value: {pc_p:.2e}\n")
                f.write(f"   Significant (p<0.05): {pc_sig}\n")
                f.write(f"   Interpretation: ")
                if pc_sig == 'YES':
                    f.write(f"Statistically significant relationship (p={pc_p:.2e}). ")
                    if abs(pc_r) > 0.7:
                        f.write(f"STRONG relationship. ")
                    elif abs(pc_r) > 0.4:
                        f.write(f"MODERATE relationship. ")
                    else:
                        f.write(f"WEAK relationship. ")
                    if pc_r > 0:
                        f.write(f"Positive: higher {feature} -> higher {target}.\n")
                    else:
                        f.write(f"Negative: higher {feature} -> lower {target}.\n")
                else:
                    f.write(f"Not statistically significant (p={pc_p:.2e}).\n")

            # Path Analysis
            f.write("\n3. PATH ANALYSIS (Direct Effect)\n")
            pa_feature = pa_df[(pa_df['FEATURE'] == feature) & (pa_df['PATH_TYPE'] == 'DIRECT')]
            if len(pa_feature) > 0:
                pa_coef = pa_feature['PATH_COEFFICIENT'].values[0]
                f.write(f"   Standardized coefficient: {pa_coef:.4f}\n")
                f.write(f"   Interpretation: ")
                if abs(pa_coef) > 0.5:
                    f.write(f"STRONG direct effect. ")
                elif abs(pa_coef) > 0.2:
                    f.write(f"MODERATE direct effect. ")
                else:
                    f.write(f"WEAK direct effect. ")
                if pa_coef > 0:
                    f.write(f"Positive effect on {target}.\n")
                else:
                    f.write(f"Negative effect on {target}.\n")

            # Method Agreement
            f.write("\n4. METHOD AGREEMENT\n")
            ale_positive = ale_feature['effect'].max() > abs(ale_feature['effect'].min())
            pc_positive = pc_r > 0 if len(pc_feature) > 0 else None
            pa_positive = pa_coef > 0 if len(pa_feature) > 0 else None

            direction_agreement = sum([ale_positive == True, pc_positive == True, pa_positive == True])
            if direction_agreement == 3:
                f.write(f"   STRONG AGREEMENT: All three methods agree on effect direction.\n")
            elif direction_agreement == 2:
                f.write(f"   PARTIAL AGREEMENT: Two of three methods agree.\n")
            else:
                f.write(f"   DISAGREEMENT: Methods show different effect directions.\n")

        f.write("\n\n" + "=" * 80 + "\n")
        f.write("SUMMARY INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")

        # Identify most important variables
        important_features = [f for f, s in sorted_features[:2]]
        f.write(f"Most Important Variables: {', '.join(important_features)}\n\n")

        f.write("Guidance for Interpretation:\n")
        f.write("- ALE shows isolated feature effects accounting for correlations\n")
        f.write("- Partial Correlations show statistical significance after controlling for confounders\n")
        f.write("- Path Analysis shows direct standardized effects\n")
        f.write("- Agreement across methods increases confidence in findings\n\n")

        f.write("For Cross-Site Analysis:\n")
        f.write(f"- Use this summary alongside the {site}_ale_curves_{target}.csv/.parquet files\n")
        f.write(f"- Compare IGBP-specific patterns using the aggregated data\n")
        f.write(f"- Weight results by N_RECORDS ({siteconfig.get('N_RECORDS', 'N/A')}) in meta-analysis\n")

    print(f"  Saved summary to {summary_file.name}")
    return None

    return None
