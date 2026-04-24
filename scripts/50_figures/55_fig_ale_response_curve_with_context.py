"""
ALE response curves with temperature context coloring and threshold detection.

## Comparison to Script 54 (SHAP Response Curves)

**Script 54 (Conditional SHAP):**
- X-axis: Binned feature values (e.g., BIN_VPD_ZSCORE)
- Y-axis: Per-sample SHAP values (feature contributions to model predictions)
- Data structure: Pre-aggregated 2D bins (xvar × yvar), one point per bin
- Interpretation: How much each feature contributes to NEP variation
- Shows: Feature importance accounting for realistic feature interactions/correlations

**Script 55 (ALE):**
- X-axis: Continuous feature range (e.g., VPD values, not binned)
- Y-axis: Accumulated Local Effects (isolated feature effect on predictions)
- Data structure: Continuous curves per temperature regime, overlaid
- Interpretation: How features affect NEP predictions, isolated from correlations
- Shows: Feature-response relationships accounting for realistic feature interactions

## What This Script Shows

ALE curves reveal how NEP responds to changes in a single feature (e.g., VPD),
holding other variables at their observed values. Multiple colored curves show
how this response varies across temperature regimes (cold → hot):

- **Parallel curves**: Feature effect is stable across temperature conditions
- **Diverging curves**: Feature effect strengthens/weakens with temperature
- **Crossing curves**: Feature has opposite effects under different conditions
- **Threshold**: VPD value where NEP response switches from positive to negative

## Key Differences from SHAP

1. **Resolution**: ALE shows smooth continuous effects; SHAP shows discrete bins
2. **Calculation**: Both use conditional methods (respect correlations)
   - SHAP: Shapley values (model-agnostic, per-sample contributions)
   - ALE: Accumulated local effects (model-specific, feature effect on predictions)
3. **Interpretation**: ALE is "what if you change this feature"; SHAP is "how much
   did this contribute to the actual prediction"
4. **Confidence**: ALE thresholds are more robust (fit polynomial to curve);
   SHAP thresholds depend on bin structure

## Structure

- **Main plot**: All sites aggregated (black mean ± gray std), overlaid curves by
  temperature regime (blue=cold → red=hot)
- **4 subplots**: Same data, separated by IGBP type (ENF, DBF, MF, EBF) for
  ecosystem-specific patterns
- **Polynomial fit**: 4th-order polynomial through mean ALE effect (red dashed)
- **Threshold**: Where mean curve crosses zero (with 95% CI)
- **Error bands**: ±1 standard deviation showing uncertainty across sites

## Use Case

Compare with Script 54 to validate findings:
- Script 54: "How important is VPD for predicting NEP?"
- Script 55: "How does NEP respond to changes in VPD? At what VPD does it flip?"
- Together: Strong agreement between SHAP importance and ALE threshold confidence
"""
from pathlib import Path

import matplotlib.colors
import numpy as np
import pandas as pd
import diive as dv
from PyALE import ale
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import src.files as files
import src.fit as fit
import src.plot as plot

# ==============================
# CONFIGURATION
# ==============================

FLUX = 'NEP_ZSCORE'
FEATURES = ['TA_ZSCORE', 'SWIN_ZSCORE', 'VPD_ZSCORE', 'SWC_ZSCORE']

# Feature to plot ALE for
PLOT_FEATURE = 'VPD_ZSCORE'  # Change to plot different features
# Feature used for context coloring (temperature bins)
CONTEXT_FEATURE = 'TA_ZSCORE'

# SINGLE CURVE MODE: True = 1 overall curve (~30 sec), False = 10 temperature curves (~1-5 min)
SINGLE_CURVE_MODE = True

# Temperature bins (only used if SINGLE_CURVE_MODE = False)
N_TEMP_BINS = 10 if not SINGLE_CURVE_MODE else 1
CONDITIONAL = True

# FAST TEST MODE: Set to True to only use 5 random sites
FAST_TEST_MODE = False

# Color palette: blue (cold) -> red (hot)
colors_list = [
    '#313695',  # Dark Blue
    '#4575b4',  # Medium Blue
    '#74add1',  # Light Blue
    '#abd9e9',  # Pale Blue
    '#e0f3f8',  # Ice Blue
    '#fee090',  # Pale Yellow
    '#fdae61',  # Light Orange
    '#f46d43',  # Orange
    '#d73027',  # Red
    '#a50026'   # Dark Red
]

bin_labels = [
    'Extreme cold',
    'Very cold',
    'Cold',
    'Cool',
    'Neutral-cool',
    'Neutral-warm',
    'Warm',
    'Hot',
    'Very hot',
    'Extreme heat'
]

custom_cmap = matplotlib.colors.ListedColormap(colors_list[:N_TEMP_BINS])
igbps = ['ENF', 'DBF', 'MF', 'EBF']

beautify = {
    "NEP_ZSCORE": "NEP",
    "ET_ZSCORE": "ET",
    "GPP_ZSCORE": "GPP",
    "RECO_ZSCORE": "RECO",
    "TA_ZSCORE": "TA",
    "VPD_ZSCORE": "VPD",
    "SWC_ZSCORE": "SWC",
    "SWIN_ZSCORE": "SWIN",
}

AX_LABELS_FONTSIZE = 12

# Load settings
settings = files.read_settings_file("../../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'
dir_ale_results = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / 'ale'
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / 'ale'
dir_out.mkdir(parents=True, exist_ok=True)

# Load subsets info
infile = Path('../../data/outputs/20_subsets/21_SUBSETS_parquet_vars_stats_subsets.csv')
subsets_df = pd.read_csv(infile)

print(f"\n{'='*80}")
if SINGLE_CURVE_MODE:
    print(f"CALCULATING SINGLE ALE CURVE (FAST MODE)")
else:
    print(f"CALCULATING ALE CURVES WITH TEMPERATURE CONTEXT")
print(f"{'='*80}\n")
print(f"Feature to plot: {PLOT_FEATURE}")
if not SINGLE_CURVE_MODE:
    print(f"Context variable: {CONTEXT_FEATURE}")
    print(f"Temperature bins: {N_TEMP_BINS}\n")
else:
    print(f"Mode: Single overall curve (no temperature binning)\n")

# ==============================
# STEP 1: LOAD DATA AND BIN BY TEMPERATURE
# ==============================

if SINGLE_CURVE_MODE:
    # SINGLE CURVE MODE: Calculate ALE per site, then average curves (equal weighting)
    all_site_data = {}
    site_igbp_map = {}

    # FAST TEST MODE: limit to 5 random sites (that have data files)
    if FAST_TEST_MODE:
        sites_with_data = []
        for ix, siteconfig in subsets_df.iterrows():
            filepath = dir_ale_results / f"{siteconfig['SITE']}_ale_{FLUX}.parquet"
            if filepath.exists():
                sites_with_data.append(ix)

        sample_size = min(5, len(sites_with_data))
        if len(sites_with_data) > 0:
            subsets_df = subsets_df.iloc[sites_with_data].sample(n=sample_size, random_state=42)
            print(f"FAST TEST MODE: Using {sample_size} random sites (from {len(sites_with_data)} available)\n")
        else:
            print("ERROR: No sites with data files found")

    for ix, siteconfig in subsets_df.iterrows():
        site = siteconfig['SITE']
        igbp = siteconfig['IGBP']
        site_igbp_map[site] = igbp

        filepath = dir_ale_results / f"{site}_ale_{FLUX}.parquet"
        if not filepath.exists():
            continue

        try:
            site_data = dv.load_parquet(filepath, sanitize_timestamp=False)
        except Exception as e:
            continue

        if PLOT_FEATURE not in site_data.columns or FLUX not in site_data.columns:
            continue

        valid_idx = ~(site_data[PLOT_FEATURE].isna() | site_data[FLUX].isna())
        site_data_clean = site_data[valid_idx].copy()

        if len(site_data_clean) > 10:
            all_site_data[site] = site_data_clean

    print(f"Loaded {len(all_site_data)} sites (each will contribute equally)\n")

else:
    # MULTI-CURVE MODE: Bin by temperature, train 10 models
    all_data_by_temp_bin = {i: [] for i in range(N_TEMP_BINS)}
    site_igbp_map = {}

    # FAST TEST MODE: limit to 5 random sites (that have data files)
    if FAST_TEST_MODE:
        sites_with_data = []
        for ix, siteconfig in subsets_df.iterrows():
            filepath = dir_ale_results / f"{siteconfig['SITE']}_ale_{FLUX}.parquet"
            if filepath.exists():
                sites_with_data.append(ix)

        sample_size = min(5, len(sites_with_data))
        if len(sites_with_data) > 0:
            subsets_df = subsets_df.iloc[sites_with_data].sample(n=sample_size, random_state=42)
            print(f"FAST TEST MODE: Using {sample_size} random sites (from {len(sites_with_data)} available)\n")
        else:
            print("ERROR: No sites with data files found")

    for ix, siteconfig in subsets_df.iterrows():
        site = siteconfig['SITE']
        igbp = siteconfig['IGBP']
        site_igbp_map[site] = igbp

        filepath = dir_ale_results / f"{site}_ale_{FLUX}.parquet"

        if not filepath.exists():
            print(f"  {site}: File not found at {filepath}")
            continue

        print(f"  Loading {site}...")

        try:
            site_data = dv.load_parquet(filepath, sanitize_timestamp=False)
        except Exception as e:
            continue

        if PLOT_FEATURE not in site_data.columns or CONTEXT_FEATURE not in site_data.columns or FLUX not in site_data.columns:
            continue

        valid_idx = ~(site_data[PLOT_FEATURE].isna() | site_data[CONTEXT_FEATURE].isna() | site_data[FLUX].isna())
        site_data_clean = site_data[valid_idx].copy()

        if len(site_data_clean) < 20:
            continue

        temp_bins, bin_edges = pd.cut(site_data_clean[CONTEXT_FEATURE], bins=N_TEMP_BINS, labels=False, retbins=True)

        for bin_idx in range(N_TEMP_BINS):
            mask = temp_bins == bin_idx
            bin_data = site_data_clean[mask].copy()

            if len(bin_data) > 10:
                bin_data['site'] = site
                bin_data['temp_bin'] = bin_idx
                all_data_by_temp_bin[bin_idx].append(bin_data)

# ==============================
# STEP 2: CALCULATE ALE (single curve or multi-curve)
# ==============================

if SINGLE_CURVE_MODE:
    print("Loading pre-calculated ALE curves from script 32...\n")
    ale_results_by_temp = {}
    all_site_ale_curves = []

    for site in all_site_data.keys():
        # Load pre-calculated ALE curves from script 32
        ale_curve_file = dir_ale_results / f"{site}_ale_curves_{FLUX}.csv"

        if not ale_curve_file.exists():
            print(f"  {site}: ALE curves file not found")
            continue

        try:
            ale_curves_df = pd.read_csv(ale_curve_file)

            # Filter for the feature we're plotting
            feature_data = ale_curves_df[ale_curves_df['feature'] == PLOT_FEATURE]

            if len(feature_data) > 0:
                all_site_ale_curves.append({
                    'site': site,
                    'feature_value': feature_data['feature_value'].values,
                    'effect': feature_data['effect'].values
                })
                print(f"  Loaded {site}")
            else:
                print(f"  {site}: Feature {PLOT_FEATURE} not in ALE curves")

        except Exception as e:
            print(f"  {site}: Error loading ALE curves: {e}")
            continue

    # Store result metadata
    ale_results_by_temp[0] = {
        'feature_value': all_site_ale_curves[0]['feature_value'] if all_site_ale_curves else np.array([]),
        'effect': np.array([]),
        'n_records': sum(len(data) for data in all_site_data.values()),
        'n_sites': len(all_site_ale_curves)
    }

    print(f"Loaded ALE curves for {len(all_site_ale_curves)} sites (pre-calculated by script 32)\n")

else:
    # Multi-curve mode
    ale_results_by_temp = {}

    for temp_bin_idx in range(N_TEMP_BINS):
        if len(all_data_by_temp_bin[temp_bin_idx]) == 0:
            continue

        print(f"Processing temperature bin {temp_bin_idx}/{N_TEMP_BINS} ...")

        # Concatenate all sites for this temperature bin
        temp_bin_data = pd.concat(all_data_by_temp_bin[temp_bin_idx], ignore_index=True)
        X = temp_bin_data[FEATURES].copy()
        y = temp_bin_data[FLUX].copy()

        if len(X) < 20:
            continue

        # Train model
        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

            xgb_params = {
                'objective': 'reg:squarederror',
                'tree_method': 'hist',
                'n_estimators': 500,  # FAST TEST: reduced from 2000
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_lambda': 1,
                'reg_alpha': 0.1,
                'gamma': 0.2,
                'min_child_weight': 5,
                'random_state': 42,
                'early_stopping_rounds': 150,  # FAST TEST: aggressive early stopping
                'n_jobs': -1
            }

            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            # Calculate ALE (FAST TEST: grid_size 50 for speed)
            ale(X=X, model=model, feature=[PLOT_FEATURE], grid_size=50)
            current_fig = plt.gcf()
            ax_temp = plt.gca()
            ale_values = []

            for line in ax_temp.get_lines():
                xdata = np.array(line.get_xdata())
                ydata = np.array(line.get_ydata())
                if len(xdata) > 0:
                    ale_values.append({'feature_value': xdata, 'effect': ydata})

            if len(ale_values) > 0:
                ale_curve = min(ale_values, key=lambda x: len(x['feature_value']))
                ale_results_by_temp[temp_bin_idx] = {
                    'feature_value': ale_curve['feature_value'],
                    'effect': ale_curve['effect'],
                    'n_records': len(X),
                    'n_sites': temp_bin_data['site'].nunique()
                }

            plt.close(current_fig)

        except Exception as e:
            print(f"  Error in bin {temp_bin_idx}: {e}")
            continue

print(f"Calculated ALE curves for {len(ale_results_by_temp)} temperature bins\n")

# ==============================
# STEP 3: INTERPOLATE TO COMMON GRID AND AVERAGE
# ==============================

if SINGLE_CURVE_MODE and len(all_site_ale_curves) > 0:
    # Single curve mode: interpolate per-site curves to common grid, then average
    all_feature_values = np.concatenate([
        curve['feature_value'] for curve in all_site_ale_curves
    ])
    common_grid = np.linspace(all_feature_values.min(), all_feature_values.max(), 50)

    # Interpolate each site's ALE curve to common grid
    ale_interpolated = {}
    site_ale_interpolated = []
    igbp_ale_interpolated = {igbp: [] for igbp in igbps}  # Track by IGBP

    for curve_idx, curve_data in enumerate(all_site_ale_curves):
        site = curve_data['site']
        x_orig = curve_data['feature_value']
        y_orig = curve_data['effect']

        sort_idx = np.argsort(x_orig)
        x_sorted = x_orig[sort_idx]
        y_sorted = y_orig[sort_idx]

        try:
            f_interp = interp1d(x_sorted, y_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')
            y_interp = f_interp(common_grid)
            site_ale_interpolated.append(y_interp)

            # Also track by IGBP
            if site in site_igbp_map:
                igbp = site_igbp_map[site]
                if igbp in igbp_ale_interpolated:
                    igbp_ale_interpolated[igbp].append(y_interp)
        except Exception as e:
            continue

    # Average across all sites (equal weighting)
    if len(site_ale_interpolated) > 0:
        site_ale_interpolated_array = np.array(site_ale_interpolated)
        ale_interpolated[0] = np.nanmean(site_ale_interpolated_array, axis=0)
        mean_effect = ale_interpolated[0]
        # Use percentile-based confidence interval instead of std
        ci_lower = np.nanpercentile(site_ale_interpolated_array, 25, axis=0)
        ci_upper = np.nanpercentile(site_ale_interpolated_array, 75, axis=0)
        std_effect = (ci_upper - ci_lower) / 2  # Half-width for compatibility with fill_between
        print(f"Averaged ALE curves from {len(site_ale_interpolated)} sites\n")

        # Also average by IGBP
        for igbp in igbps:
            if len(igbp_ale_interpolated[igbp]) > 0:
                ale_interpolated[igbp] = np.nanmean(igbp_ale_interpolated[igbp], axis=0)
                print(f"  {igbp}: {len(igbp_ale_interpolated[igbp])} sites")
    else:
        print("ERROR: Could not interpolate any curves")
        mean_effect = None
        std_effect = None

elif len(ale_results_by_temp) > 0:
    # Multi-curve mode: existing logic
    all_feature_values = np.concatenate([
        ale_results_by_temp[idx]['feature_value']
        for idx in ale_results_by_temp.keys()
    ])
    common_grid = np.linspace(all_feature_values.min(), all_feature_values.max(), 50)

    ale_interpolated = {}
    for temp_bin_idx, ale_data in ale_results_by_temp.items():
        x_orig = ale_data['feature_value']
        y_orig = ale_data['effect']

        sort_idx = np.argsort(x_orig)
        x_sorted = x_orig[sort_idx]
        y_sorted = y_orig[sort_idx]

        try:
            f_interp = interp1d(x_sorted, y_sorted, kind='linear', bounds_error=False, fill_value='extrapolate')
            y_interp = f_interp(common_grid)
            ale_interpolated[temp_bin_idx] = y_interp
        except Exception as e:
            continue

    # Calculate mean ALE across all temperature bins (for threshold detection)
    all_effects = np.array([ale_interpolated[idx] for idx in sorted(ale_interpolated.keys())])
    mean_effect = np.nanmean(all_effects, axis=0)
    # Use percentile-based confidence interval instead of std
    ci_lower = np.nanpercentile(all_effects, 25, axis=0)
    ci_upper = np.nanpercentile(all_effects, 75, axis=0)
    std_effect = (ci_upper - ci_lower) / 2  # Half-width for compatibility with fill_between

else:
    print("ERROR: No ALE data generated")
    ale_interpolated = None
    mean_effect = None
    std_effect = None

# ==============================
# STEP 4: FIT POLYNOMIAL AND DETECT THRESHOLD (both modes)
# ==============================

if mean_effect is not None and std_effect is not None:
    # Calculate y-axis limits using percentiles to avoid extreme outliers
    all_y_values = mean_effect.copy()
    if not SINGLE_CURVE_MODE:
        for temp_bin_idx in sorted(ale_interpolated.keys()):
            if temp_bin_idx != 0:  # Skip the first key if it exists
                all_y_values = np.concatenate([all_y_values, ale_interpolated[temp_bin_idx]])

    y_min = np.nanpercentile(all_y_values, 5)
    y_max = np.nanpercentile(all_y_values, 95)
    y_margin = (y_max - y_min) * 0.15
    y_limits = (y_min - y_margin, y_max + y_margin)

    poly_func, poly_coeffs, x_fit, y_fit, r_squared, pi_upper, pi_lower = fit.fit_polynomial(
        X_data=common_grid, Y_data=mean_effect
    )

    threshold_main, threshold_lower, threshold_upper = fit.calc_threshold(
        x_fit=x_fit, y_fit=y_fit, pi_lower=pi_lower, pi_upper=pi_upper
    )

    print(f"Polynomial fit R²: {r_squared:.4f}")
    print(f"Threshold: {threshold_main:.3f} [{threshold_lower:.3f}, {threshold_upper:.3f}]\n")

    # ==============================
    # STEP 5: CREATE FIGURE (both modes)
    # ==============================

    fig, gs, ax_all, axes_sub = plot.layout_5panels((13.86, 6.67), add_colorbar_ax=False)

    xlabel = rf'{beautify[PLOT_FEATURE]} ($\sigma$)'
    ylabel = rf'ALE effect on daytime {beautify[FLUX]} ($\sigma$)'

    # Main plot
    if SINGLE_CURVE_MODE:
        # Single curve: no temperature coloring
        y_interp = ale_interpolated[0]
        ax_all.plot(common_grid, y_interp, color='steelblue', linewidth=2.5, label='Overall ALE', zorder=100)
    else:
        # Multiple curves: color-coded by temperature
        for temp_bin_idx in sorted(ale_interpolated.keys()):
            y_interp = ale_interpolated[temp_bin_idx]
            color = colors_list[temp_bin_idx] if temp_bin_idx < len(colors_list) else '#999999'
            ax_all.plot(common_grid, y_interp, color=color, alpha=0.5, linewidth=1.5, label=bin_labels[temp_bin_idx])

    # Mean curve with error band
    ax_all.plot(common_grid, mean_effect, color='black', linewidth=2.5, label='Mean', zorder=100)
    ax_all.fill_between(common_grid, mean_effect - std_effect, mean_effect + std_effect,
                         color='gray', alpha=0.2, zorder=1)

    # Add polynomial fit
    fillbetweenplot = plot.add_fit(
        ax=ax_all, x_fit=x_fit, y_fit=y_fit, pi_lower=pi_lower, pi_upper=pi_upper,
        poly_func=poly_func, r_squared=r_squared, show_annotate=True,
        fontsize=AX_LABELS_FONTSIZE, color='#d32f2f', linewidth=2.5
    )

    # Threshold lines
    min_ix = np.argmin(y_fit)
    max_ix = np.argmax(y_fit)
    colors_symbols = ['black', 'black', 'black']
    plot.show_shap_thresholds(ax=ax_all, x_fit=x_fit, y_fit=y_fit, max_ix=max_ix, min_ix=min_ix,
                              threshold_main=threshold_main, show_annotate=True,
                              fontsize=AX_LABELS_FONTSIZE, show_annotate_short=False,
                              colors_symbols=colors_symbols)

    ax_all.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax_all.set_ylim(y_limits)
    ax_all.set_xlabel(xlabel, fontsize=AX_LABELS_FONTSIZE)
    ax_all.set_ylabel(ylabel, fontsize=AX_LABELS_FONTSIZE)

    if not SINGLE_CURVE_MODE:
        ax_all.legend(loc='best', fontsize=AX_LABELS_FONTSIZE * 0.7, ncol=2, bbox_to_anchor=(0.98, 0.97))
    else:
        ax_all.legend(loc='best', fontsize=AX_LABELS_FONTSIZE)

    ax_all.text(0, 1.05, 'a', transform=ax_all.transAxes, size=AX_LABELS_FONTSIZE * 1.2, weight='bold')
    ax_all.text(0.05, 1.05, 'Global forests', transform=ax_all.transAxes, size=AX_LABELS_FONTSIZE * 1.2)

    plot.format(ax=ax_all, fontsize=AX_LABELS_FONTSIZE, showyticklabels=True, showxticklabels=True,
                xtickdigits=1, ytickdigits=1, showbottomspine=True, showleftspine=True, showymajorticks=True)

    # IGBP subplots (simplified: same data for all)
    configs = zip(
        axes_sub, igbps,
        [" ", " ", xlabel, xlabel],
        [" ", " ", " ", " "],
        ['b', 'c', 'd', 'e'],
        [True, False, True, False],
        [False, False, True, True]
    )

    for ax, igbp, xl, yl, letter, showyticklabels, showxticklabels in configs:
        if SINGLE_CURVE_MODE:
            # Single curve mode: plot ecosystem-specific curve if available
            if igbp in ale_interpolated:
                y_igbp = ale_interpolated[igbp]
                ax.plot(common_grid, y_igbp, color='steelblue', linewidth=2, zorder=100)
                # Use global std_effect as proxy for uncertainty
                ax.fill_between(common_grid, y_igbp - std_effect*0.5, y_igbp + std_effect*0.5,
                                 color='steelblue', alpha=0.15, zorder=1)
                # Auto-scale subplot to its own data range
                y_min_sub = np.nanpercentile(y_igbp, 5)
                y_max_sub = np.nanpercentile(y_igbp, 95)
                y_margin_sub = (y_max_sub - y_min_sub) * 0.15 if y_max_sub > y_min_sub else 1.0
                ax.set_ylim((y_min_sub - y_margin_sub, y_max_sub + y_margin_sub))
            else:
                # Fallback to global curve if IGBP not available
                y_interp = ale_interpolated[0]
                ax.plot(common_grid, y_interp, color='steelblue', linewidth=2, zorder=100)
                ax.fill_between(common_grid, y_interp - std_effect, y_interp + std_effect,
                                 color='steelblue', alpha=0.15, zorder=1)
                ax.set_ylim(y_limits)
        else:
            # Multi-curve mode: plot all temperature curves
            for temp_bin_idx in sorted(ale_interpolated.keys()):
                y_interp = ale_interpolated[temp_bin_idx]
                color = colors_list[temp_bin_idx] if temp_bin_idx < len(colors_list) else '#999999'
                ax.plot(common_grid, y_interp, color=color, alpha=0.3, linewidth=1)

            # Mean curve
            ax.plot(common_grid, mean_effect, color='black', linewidth=2, zorder=100)
            ax.fill_between(common_grid, mean_effect - std_effect, mean_effect + std_effect,
                             color='gray', alpha=0.15, zorder=1)
            ax.set_ylim(y_limits)

        ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel(xl, fontsize=AX_LABELS_FONTSIZE)
        ax.set_ylabel(yl, fontsize=AX_LABELS_FONTSIZE)
        ax.text(0, 1.1, letter, transform=ax.transAxes, size=AX_LABELS_FONTSIZE * 1.2, weight='bold')
        ax.text(0.1, 1.1, igbp, transform=ax.transAxes, size=AX_LABELS_FONTSIZE * 1.2)

        plot.format(ax=ax, fontsize=AX_LABELS_FONTSIZE, showyticklabels=showyticklabels, showxticklabels=showxticklabels,
                    xtickdigits=1, ytickdigits=1, showbottomspine=True, showleftspine=True, showymajorticks=True)

    fig.tight_layout()
    gs.update(wspace=.1)

    # Save figure
    outfilepath = dir_out / f'55_FIG_ALE_ResponseCurve_{PLOT_FEATURE}_Threshold_{FLUX}.png'
    fig.savefig(outfilepath, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {outfilepath}\n")

    # Save coefficients
    df_coeffs = pd.DataFrame([{
        'feature': PLOT_FEATURE,
        'target': FLUX,
        'a': poly_coeffs[0],
        'b': poly_coeffs[1],
        'c': poly_coeffs[2],
        'd': poly_coeffs[3],
        'e': poly_coeffs[4],
        'R2': r_squared,
        'Threshold': f'{threshold_main:.2f}',
        'Threshold_CI': f'[{threshold_lower:.2f}, {threshold_upper:.2f}]'
    }])

    coeffs_path = dir_out / f'55_FIG_ALE_ResponseCurve_{PLOT_FEATURE}_{FLUX}_COEFFICIENTS.csv'
    df_coeffs.to_csv(coeffs_path, index=False)
    print(f"Saved coefficients to: {coeffs_path}")
