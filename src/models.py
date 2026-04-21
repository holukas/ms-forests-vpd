from pathlib import Path

import diive as dv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from PyALE import ale
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
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
                                  siteconfig, ix, modelstxt, results_outdir: Path, conditional=False) -> dict:
    """
    Train XGBoost models with 5-fold cross-validation and compute out-of-sample SHAP values.

    ## Logic: Out-of-Sample SHAP Values via 5-Fold CV

    **Why 5-fold CV?**
    - Standard SHAP calculation trains on 85% of data, then explains the ENTIRE dataset
    - This means SHAP values for training data (in-sample) reflect patterns the model learned
    - In-sample SHAP can be misleading: high values don't prove causality (model memorized)
    - Out-of-sample SHAP ensures each data point is explained by a model that never saw it
    - This gives unbiased feature importance: high SHAP = the feature truly predicts the target

    **Per-Fold Workflow:**
    1. KFold splits data into 5 folds (80% train, 20% test)
    2. For each fold:
       a) Train set (80%): further split into 68% training, 12% validation
          - 68% trains the model
          - 12% guides early stopping (prevents overfitting)
       b) Test set (20%): held out completely, used for:
          - Model evaluation (R², RMSE)
          - SHAP calculation (out-of-sample explanations)
    3. All 5 folds combined = full dataset coverage with unbiased explanations

    **SHAP Calculation:**
    - Conditional SHAP (default): Respects feature correlations (realistic effects)
    - Standard SHAP: Marginal effects (what if features were independent?)
    - Background data for standard SHAP: sampled from training set only (avoids data leakage)

    **Output:**
    - Per-fold metrics: R² and RMSE for each fold (shows generalization)
    - Global metrics: Out-of-sample R² and RMSE across all 5 folds (overall model quality)
    - SHAP values: One per sample, from a model that never saw that sample
    - Predictions: Out-of-sample predictions for the entire dataset

    Parameters
    ----------
    target : str
        Target variable name (e.g., 'NEP_ZSCORE')
    features : list
        List of feature column names (e.g., ['TA_ZSCORE', 'SWIN_ZSCORE', ...])
    siteconfig : dict
        Row from site configuration CSV, containing 'SITE', '_FILEPATH_PARQUET_SUBSET', etc.
    ix : int
        Site index (for logging/progress)
    modelstxt : Path
        Path to text file for logging model metrics and performance
    results_outdir : Path
        Directory to save SHAP results (parquet + CSV)
    conditional : bool, default=False
        If True: Calculate conditional SHAP (respects correlations)
        If False: Calculate standard/marginal SHAP (assumes independence)

    Returns
    -------
    dict
        Dictionary with CV results for aggregation:
        {
            'site': str (site name),
            'target': str (target variable),
            'fold_1_r2': float, 'fold_1_rmse': float,
            'fold_2_r2': float, 'fold_2_rmse': float,
            ...,
            'fold_5_r2': float, 'fold_5_rmse': float,
            'global_r2': float, 'global_rmse': float
        }

    Notes
    -----
    - Fixed hyperparameters (reg_lambda=1, reg_alpha=0.1, gamma=0.2, etc.)
      ensure consistent model behavior across sites
    - Early stopping on validation set prevents overfitting within each fold
    - Output files include all features, SHAP values, predictions, and original measurements
    """

    print(f"\nLoading data for site #{ix + 1} {siteconfig['SITE']} ...")

    if siteconfig['_FILEPATH_PARQUET_SUBSET'] == '-MISSING-':
        return {}

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
    # 5-FOLD CROSS-VALIDATION for OUT-OF-SAMPLE SHAP VALUES
    # ---------------------------------------------------------
    # Each data point gets SHAP values from a model that never saw it during training.
    # This ensures unbiased feature importance explanations.

    print("Initializing 5-fold cross-validation for out-of-sample SHAP values...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Storage for CV results
    shap_values_all = []
    expected_values_folds = []
    predictions_all = []
    fold_indices = []
    fold_metrics = []

    # XGBoost hyperparameters (used for each fold)
    xgb_params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',  # Faster training
        'n_estimators': 3000,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        # 'colsample_bynode': 0.8,  # Feature diversity
        'reg_lambda': 1,  # L2 Regularization
        'reg_alpha': 0.1,  # L1 Regularization
        'gamma': 0.2,  # Conservative splitting
        'min_child_weight': 5,
        'random_state': 42,
        'early_stopping_rounds': 100,
        'n_jobs': -1
    }

    # Per-fold processing
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
        print(f"\n--- FOLD {fold_idx + 1}/5 ---")

        # Split data
        X_train_full = X.iloc[train_idx]
        y_train_full = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # Further split training set: 85% for training, 15% for validation (early stopping)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.15, random_state=42
        )

        # Train XGBoost model
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Predictions and metrics on test set
        y_pred_test = model.predict(X_test)
        r2_test = model.score(X_test, y_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        print(f"Fold {fold_idx + 1} Performance -> R2: {r2_test:.4f} / RMSE: {rmse_test:.2f}")
        fold_metrics.append({'fold': fold_idx + 1, 'r2': r2_test, 'rmse': rmse_test})

        # Calculate SHAP values ONLY for test set (out-of-sample)
        if conditional:
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            shap_explanation = explainer(X_test)
            shap_vals_fold = shap_explanation.values
            expected_val_fold = shap_explanation.base_values
        else:
            # Use background data from training set only
            n_background_samples = min(300, len(X_train))
            background_data = shap.sample(X_train, n_background_samples)
            explainer = shap.TreeExplainer(model, data=background_data, feature_perturbation="interventional")
            shap_explanation = explainer(X_test)
            shap_vals_fold = shap_explanation.values
            expected_val_fold = shap_explanation.base_values

        # Store results for this fold
        shap_values_all.append(shap_vals_fold)
        expected_values_folds.append(expected_val_fold)
        predictions_all.append(y_pred_test)
        fold_indices.append(test_idx)

    # Log per-fold metrics
    print(f"\n=== CV FOLD METRICS ===")
    with open(modelstxt, 'a', encoding='utf-8') as file:
        file.write(f"\nSITE: {siteconfig['SITE']} / TARGET: {target} / METHOD: 5-FOLD CV\n")
        for metrics in fold_metrics:
            log_line = f"  FOLD {metrics['fold']}: R2={metrics['r2']:.4f} / RMSE={metrics['rmse']:.2f}\n"
            file.write(log_line)
            print(log_line.strip())

    # Reassemble data in original index order
    print("\nReassembling out-of-sample predictions and SHAP values...")

    # Concatenate all fold results
    shap_values_cv = np.vstack(shap_values_all)
    predictions_cv = np.concatenate(predictions_all)
    all_fold_indices = np.concatenate(fold_indices)

    # Sort back to original order
    sort_idx = np.argsort(all_fold_indices)
    shap_values_cv = shap_values_cv[sort_idx]
    predictions_cv = predictions_cv[sort_idx]

    # Calculate global out-of-sample R² and RMSE across all folds
    print(f"\n=== GLOBAL OUT-OF-SAMPLE METRICS ===")
    global_r2 = r2_score(y, predictions_cv)
    global_rmse = np.sqrt(mean_squared_error(y, predictions_cv))
    print(f"GLOBAL Out-of-Sample R²: {global_r2:.4f}")
    print(f"GLOBAL Out-of-Sample RMSE: {global_rmse:.2f}")

    with open(modelstxt, 'a', encoding='utf-8') as file:
        file.write(f"GLOBAL OUT-OF-SAMPLE R2: {global_r2:.4f} / RMSE: {global_rmse:.2f}\n")

    # Compute expected value as mean across all folds
    expected_value_cv = np.mean([
        float(ev[0]) if isinstance(ev, np.ndarray) else float(ev)
        for ev in expected_values_folds
    ])

    # Create SHAP dataframe
    shapcols = [f'{c}_SHAPVALS' for c in X.columns]
    shapdf = pd.DataFrame(data=shap_values_cv, index=X.index, columns=shapcols)
    shapdf.index = pd.to_datetime(shapdf.index)
    shapdf['SUM'] = shapdf.sum(axis=1)
    shapdf['EXPECTED'] = expected_value_cv  # Single value across all data
    shapdf['SUM+EXPECTED'] = shapdf['EXPECTED'] + shapdf['SUM']
    shapdf[target] = y.copy()
    shapdf[f"{target}_PRED"] = predictions_cv

    # Merge SHAP values with measured
    merged = pd.concat([X, shapdf], axis=1)

    # Add additional columns (e.g., other fluxes)
    current_cols = merged.columns.tolist()
    available_cols = subset.columns.tolist()
    additional_cols = [col for col in available_cols if col not in current_cols]
    for addcol in additional_cols:
        # print(f"Adding column {addcol} to SHAP dataframe.")
        merged[addcol] = subset[addcol].copy()
    merged = merged.sort_index(axis=1)

    substr = "conditional" if conditional else "standard"
    outfilepath = dv.save_parquet(
        filename=f"{siteconfig['SITE']}_shap-{substr}_{target}",
        data=merged,
        outpath=results_outdir)
    print(f"Saved SHAP values to file {outfilepath}.")
    merged.to_csv(outfilepath.replace('.parquet', '.csv'))

    # Return CV results for aggregation
    cv_result_row = {'site': siteconfig['SITE'], 'target': target}
    for metrics in fold_metrics:
        cv_result_row[f"fold_{metrics['fold']}_r2"] = metrics['r2']
        cv_result_row[f"fold_{metrics['fold']}_rmse"] = metrics['rmse']
    cv_result_row['global_r2'] = global_r2
    cv_result_row['global_rmse'] = global_rmse

    return cv_result_row


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


def tune_xgboost_hyperparameters(target: str, features: list,
                                 siteconfig, ix, results_outdir: Path, n_iter: int = 25) -> dict:
    """
    Hyperparameter tuning for XGBoost using RandomizedSearchCV with 5-fold CV.

    Tunes only the core model complexity parameters (n_estimators, max_depth, learning_rate)
    while keeping regularization and feature subsampling fixed at well-tested values.

    Useful for validating or optimizing model performance.

    Parameters:
    - n_iter: Number of parameter combinations to test (default: 25)

    Returns:
    - Dictionary with best parameters and best CV score
    """
    print(f"\nHyperparameter Tuning for site #{ix + 1} {siteconfig['SITE']} ...")

    if siteconfig['_FILEPATH_PARQUET_SUBSET'] == '-MISSING-':
        return {}

    # Load site data
    filepath = siteconfig['_FILEPATH_PARQUET_SUBSET']
    subset = dv.load_parquet(filepath, sanitize_timestamp=False)
    print(f"Records: {len(subset)}")

    # Target and features
    X = subset[features].copy()
    y = subset[target].copy()

    # Simplified parameter grid (tuning only core complexity parameters)
    # Regularization and feature subsampling are fixed at well-tested values
    param_dist = {
        'n_estimators': [2000, 2500, 3000, 3500, 4000],
        'max_depth': [4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
    }

    # Base model with fixed hyperparameters (consistent with train_xgboost_models_and_shap)
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',        # Faster histogram-based training
        subsample=0.8,             # 80% of rows per tree
        colsample_bytree=0.8,      # 80% of features per tree
        reg_lambda=1,              # L2 regularization
        reg_alpha=0.1,             # L1 regularization
        gamma=0.2,                 # Minimum loss reduction for split
        min_child_weight=5,        # Prevent splitting on outliers
        random_state=42,
        n_jobs=-1
    )

    # Randomized search with 5-fold CV
    print(f"Testing {n_iter} parameter combinations with 5-fold CV...")
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='r2',
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    # Fit
    search.fit(X, y)

    # Results
    print(f"\n{'='*80}")
    print(f"BEST PARAMETERS for {siteconfig['SITE']}:")
    print(f"{'='*80}")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest CV R² Score: {search.best_score_:.4f}")
    print(f"{'='*80}\n")

    # Save results to file
    tuning_results = {
        'site': siteconfig['SITE'],
        'target': target,
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'n_iter': n_iter,
        'records': len(X)
    }

    # Save to text file
    tuning_file = Path(results_outdir) / f"{siteconfig['SITE']}_hyperparameter_tuning_{target}.txt"
    with open(tuning_file, 'w', encoding='utf-8') as f:
        f.write(f"Hyperparameter Tuning Results\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Site: {siteconfig['SITE']}\n")
        f.write(f"Target: {target}\n")
        f.write(f"Records: {len(X)}\n")
        f.write(f"Parameter Combinations Tested: {n_iter}\n")
        f.write(f"CV Method: 5-fold\n\n")

        f.write(f"TUNED PARAMETERS:\n")
        for param, value in search.best_params_.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\nBest CV R² Score: {search.best_score_:.4f}\n\n")

        f.write(f"FIXED PARAMETERS (used in all combinations):\n")
        f.write(f"  tree_method: hist\n")
        f.write(f"  subsample: 0.8\n")
        f.write(f"  colsample_bytree: 0.8\n")
        f.write(f"  reg_lambda: 1.0  (L2 regularization)\n")
        f.write(f"  reg_alpha: 0.1   (L1 regularization)\n")
        f.write(f"  gamma: 0.2       (minimum loss reduction)\n")
        f.write(f"  min_child_weight: 5\n")
        f.write(f"{'='*80}\n")

    print(f"Saved tuning results to {tuning_file}")

    return tuning_results
