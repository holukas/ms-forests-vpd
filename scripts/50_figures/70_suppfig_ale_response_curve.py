"""
ALE (Accumulated Local Effects) response curves with ecosystem-specific aggregation and threshold detection.

## Overview

Generates ALE curves showing how NEP responds to changes in a single feature (e.g., VPD).
Loads pre-calculated per-site ALE curves from script 32, aggregates them via equal weighting,
fits 4th-order polynomials for smoothing, and detects response thresholds via zero-crossing analysis.

## Data Pipeline

1. **Load per-site ALE curves** from script 32 (pre-calculated, no recalculation)
2. **Equal weighting aggregation**: Each site contributes one curve regardless of record count
3. **Polynomial pre-fitting**: Fit 4th-order polynomial to each site's curve individually
4. **Aggregate smoothed curves**: Average polynomial-fitted curves across sites
5. **Calculate confidence intervals**: 95% CI or SEM from distribution of individual thresholds
6. **Threshold detection**: Find positive-to-negative zero crossings in individual curves
7. **Ecosystem-specific aggregation**: Separate aggregation by IGBP type (ENF, DBF, MF, EBF)

## Key Features

### Aggregation Methods
- `USE_MEDIAN=False` (default): Mean aggregation - equal weighting, traditional approach
- `USE_MEDIAN=True`: Median aggregation - robust to outlier sites with extreme curves

### Uncertainty Quantification
- `USE_CI=True` (default): 95% Confidence Interval for mean threshold (t-distribution based)
- `USE_CI=False`: Standard Error of Mean (SEM) - shows precision of mean estimate

### Threshold Detection
- Finds individual zero-crossings in each site's poly-fitted curve
- Only includes curves that actually cross zero (positive → negative transition)
- Consensus threshold = mean of individual crossing points
- Reports: threshold ± SEM, or threshold [CI_lower, CI_upper]
- Includes: number of sites with crossing, total sites, range of thresholds

### Visualization
- **Main panel (a)**: Global curve + per-site background curves (gray, alpha=0.15)
- **Subplots (b-e)**: Same data separated by IGBP ecosystem
- **Threshold markers**: Open circle at zero-crossing point
- **Threshold labels**: Show value ± uncertainty (SEM or 95% CI)
- **Zero-crossing line**: Dashed line at y=0 for reference
- **Axis styling**: Matches script 54 (black spines, solid lines, xtickdigits=0)

## Comparison to Script 54 (SHAP)

**Script 54:**
- X-axis: Binned feature values
- Y-axis: Per-sample SHAP values (feature importance)
- Interpretation: How much features contribute to NEP variation

**Script 70:**
- X-axis: Continuous feature values
- Y-axis: Isolated feature effects on predictions
- Interpretation: How NEP responds to feature changes

**Together**: Validate findings across different explanation methods

## Configuration Options

```python
PLOT_FEATURE = 'VPD_ZSCORE'        # Feature to analyze
FLUX = 'NEP_ZSCORE'                # Target variable

FAST_TEST_MODE = False              # True: use 5 random sites for testing
USE_MEDIAN = False                  # True: median aggregation (robust to outliers)
USE_CI = True                       # True: show 95% CI, False: show SEM
```

## Output

**Console:**
- THRESHOLD SUMMARY table showing all results:
  - Global threshold for all sites combined
  - Per-IGBP thresholds with uncertainty quantification
  - Number of sites with zero-crossings per group
  - Header indicates which uncertainty metric is displayed

**Figure:**
- 5-panel layout (1 main + 4 IGBP subplots)
- Threshold markers and labels on each panel
- Consistent axis styling across all panels
- Per-site background curves for context

**Interpretation:**
- Threshold value: Where ALE curve crosses zero (NEP switches from stimulation to suppression)
- Uncertainty: ±SEM shows precision; 95% CI shows likely range of true threshold
- Site count: Shows robustness of threshold estimate (more sites = more confidence)
- IGBP subplots: Show ecosystem-specific response patterns
"""
from pathlib import Path

import diive as dv
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import src.files as files
import src.plot as plot
from src.paths import data_path, load_settings

# Open the figure in a window after saving. False by default so a script can run
# unattended: matplotlib picks the interactive TkAgg backend here, and plt.show()
# then blocks until the window is closed by hand.
SHOW_PLOT = False


def fit_polynomial_to_curves(curves_array, grid):
    """Fit 4th-order polynomial to each curve, preserving NaN regions."""
    fitted = []
    for curve in curves_array:
        valid = ~np.isnan(curve)
        if np.sum(valid) > 4:
            try:
                x_valid, y_valid = grid[valid], curve[valid]
                poly = np.poly1d(np.polyfit(x_valid, y_valid, 4))
                result = poly(grid)
                result[~valid] = np.nan
                fitted.append(result)
            except Exception:
                fitted.append(curve)
        else:
            fitted.append(curve)
    return fitted


def find_zero_crossing(curve, grid):
    """Find first positive-to-negative zero crossing via linear interpolation. Returns NaN if no crossing."""
    valid = ~np.isnan(curve)
    if np.sum(valid) <= 1:
        return np.nan
    curve_valid = curve[valid]
    grid_valid = grid[valid]
    if np.all(curve_valid >= 0) or np.all(curve_valid <= 0):
        return np.nan
    sign_changes = np.diff(np.sign(curve_valid))
    for idx in np.where(sign_changes != 0)[0]:
        if curve_valid[idx] > 0 and curve_valid[idx + 1] <= 0:
            x1, x2 = grid_valid[idx], grid_valid[idx + 1]
            y1, y2 = curve_valid[idx], curve_valid[idx + 1]
            return x1 - y1 * (x2 - x1) / (y2 - y1) if (y2 - y1) != 0 else (x1 + x2) / 2
    return np.nan


def calc_ci_95(values):
    """Calculate 95% confidence interval for mean using t-distribution."""
    from scipy import stats
    mean = values.mean()
    sem = stats.sem(values)
    ci = sem * stats.t.ppf((1 + 0.95) / 2, len(values) - 1)
    return mean, mean - ci, mean + ci


# ==============================
# CONFIGURATION
# ==============================

FLUX = 'NEP_ZSCORE'

# Run variant. An empty string reads the results behind the submitted figures and
# writes to the baseline plot folder. Any other value reads the matching variant
# folder and writes the figures next to it, so a sensitivity run cannot overwrite a
# published figure. The aggregation must have run with the same value.
VARIANT = ""
PLOT_FEATURE = 'VPD_ZSCORE'
FAST_TEST_MODE = False
USE_MEDIAN = False
USE_CI = True  # True: 95% CI, False: SEM
IGBPS = ['ENF', 'DBF', 'MF', 'EBF']
AX_LABELS_FONTSIZE = 12

BEAUTIFY = {
    "NEP_ZSCORE": "NEP", "ET_ZSCORE": "ET", "GPP_ZSCORE": "GPP", "RECO_ZSCORE": "RECO",
    "TA_ZSCORE": "TA", "VPD_ZSCORE": "VPD", "SWC_ZSCORE": "SWC", "SWIN_ZSCORE": "SWIN",
}

settings = load_settings()
dir_ale_results = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / 'ale' / VARIANT
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / 'ale' / VARIANT

agg_name = "median" if USE_MEDIAN else "mean"
print(f"\n{'=' * 80}\nALE Response Curves | Feature: {BEAUTIFY[PLOT_FEATURE]} | Aggregation: {agg_name}\n{'=' * 80}\n")
dir_out.mkdir(parents=True, exist_ok=True)

subsets_df = pd.read_csv(str(data_path("data/outputs/20_subsets/21_SUBSETS_parquet_vars_stats_subsets.csv")))

# ==============================
# STEP 1: LOAD DATA
# ==============================

all_site_data = {}
site_igbp_map = {}

if FAST_TEST_MODE:
    sites_with_data = [ix for ix, row in subsets_df.iterrows()
                       if (dir_ale_results / f"{row['SITE']}_ale_{FLUX}.parquet").exists()]
    sample_size = min(5, len(sites_with_data))
    if sites_with_data:
        subsets_df = subsets_df.iloc[sites_with_data].sample(n=sample_size, random_state=42)
        print(f"FAST TEST MODE: Using {sample_size} random sites\n")
    else:
        print("ERROR: No sites with data files found\n")

for ix, siteconfig in subsets_df.iterrows():
    site = siteconfig['SITE']
    igbp = siteconfig['IGBP']
    site_igbp_map[site] = igbp

    filepath = dir_ale_results / f"{site}_ale_{FLUX}.parquet"
    if not filepath.exists():
        continue

    try:
        site_data = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
    except Exception:
        continue

    if PLOT_FEATURE not in site_data.columns or FLUX not in site_data.columns:
        continue

    valid_idx = ~(site_data[PLOT_FEATURE].isna() | site_data[FLUX].isna())
    site_data_clean = site_data[valid_idx].copy()

    if len(site_data_clean) > 10:
        all_site_data[site] = site_data_clean

# ==============================
# STEP 2: CALCULATE ALE
# ==============================

ale_results_by_temp = {}
all_site_ale_curves = []

for site in all_site_data.keys():
    ale_curve_file = dir_ale_results / f"{site}_ale_curves_{FLUX}.csv"
    if not ale_curve_file.exists():
        continue

    try:
        ale_curves_df = pd.read_csv(ale_curve_file)
        feature_data = ale_curves_df[ale_curves_df['feature'] == PLOT_FEATURE]
        if len(feature_data) > 0:
            all_site_ale_curves.append({
                'site': site,
                'feature_value': feature_data['feature_value'].values,
                'effect': feature_data['effect'].values
            })
    except Exception:
        continue

# Store result metadata
ale_results_by_temp[0] = {
    'feature_value': all_site_ale_curves[0]['feature_value'] if all_site_ale_curves else np.array([]),
    'effect': np.array([]),
    'n_records': sum(len(data) for data in all_site_data.values()),
    'n_sites': len(all_site_ale_curves)
}

# ==============================
# STEP 3: INTERPOLATE TO COMMON GRID AND AVERAGE
# ==============================

if len(all_site_ale_curves) > 0:
    # Single curve mode: interpolate per-site curves to common grid, then average
    all_feature_values = np.concatenate([
        curve['feature_value'] for curve in all_site_ale_curves
    ])
    common_grid = np.linspace(all_feature_values.min(), all_feature_values.max(), 50)

    # Interpolate each site's ALE curve to common grid
    ale_interpolated = {}
    site_ale_interpolated = []
    igbp_ale_interpolated = {igbp: [] for igbp in IGBPS}  # Track by IGBP

    for curve_idx, curve_data in enumerate(all_site_ale_curves):
        site = curve_data['site']
        x_orig = curve_data['feature_value']
        y_orig = curve_data['effect']

        sort_idx = np.argsort(x_orig)
        x_sorted = x_orig[sort_idx]
        y_sorted = y_orig[sort_idx]

        try:
            # Only interpolate within data range; set extrapolated regions to NaN (no extrapolation)
            f_interp = interp1d(x_sorted, y_sorted, kind='linear', bounds_error=False, fill_value=np.nan)
            y_interp = f_interp(common_grid)
            site_ale_interpolated.append(y_interp)

            # Also track by IGBP
            if site in site_igbp_map:
                igbp = site_igbp_map[site]
                if igbp in igbp_ale_interpolated:
                    igbp_ale_interpolated[igbp].append(y_interp)
        except Exception as e:
            continue

    if len(site_ale_interpolated) > 0:
        site_ale_poly_fitted = fit_polynomial_to_curves(site_ale_interpolated, common_grid)
        site_ale_interpolated_array = np.array(site_ale_poly_fitted)
        agg_func = np.nanmedian if USE_MEDIAN else np.nanmean
        mean_effect = agg_func(site_ale_interpolated_array, axis=0)
        ale_interpolated[0] = mean_effect

        # Calculate confidence intervals from the spread of poly-fitted site curves
        # Use percentiles to get uncertainty band
        ci_lower = np.nanpercentile(site_ale_interpolated_array, 2.5, axis=0)
        ci_upper = np.nanpercentile(site_ale_interpolated_array, 97.5, axis=0)
        std_effect = np.nanstd(site_ale_interpolated_array, axis=0)

        # Mask regions with insufficient site coverage (require 50% of sites)
        min_sites_required = len(site_ale_poly_fitted) / 2
        site_coverage = np.sum(~np.isnan(site_ale_interpolated_array), axis=0)
        insufficient_mask = site_coverage < min_sites_required

        mean_effect[insufficient_mask] = np.nan
        std_effect[insufficient_mask] = np.nan

        for igbp in IGBPS:
            if len(igbp_ale_interpolated[igbp]) > 0:
                igbp_poly_fitted = fit_polynomial_to_curves(igbp_ale_interpolated[igbp], common_grid)
                igbp_array = np.array(igbp_poly_fitted)
                igbp_mean = agg_func(igbp_array, axis=0)

                min_sites_igbp = len(igbp_poly_fitted) / 2
                igbp_coverage = np.sum(~np.isnan(igbp_array), axis=0)
                igbp_mean[igbp_coverage < min_sites_igbp] = np.nan

                ale_interpolated[igbp] = igbp_mean
    else:
        print("ERROR: Could not interpolate any curves")
        mean_effect = None
        std_effect = None

else:
    print("ERROR: No ALE data generated")
    ale_interpolated = None
    mean_effect = None
    std_effect = None

# ==============================
# STEP 4: FIT POLYNOMIAL AND DETECT THRESHOLD (both modes)
# ==============================

if mean_effect is not None and std_effect is not None:
    # Calculate y-axis limits based on the ENTIRE mean_effect curve (including masked regions)
    # This ensures the complete curve is visible even if peaks are in sparse regions
    y_min = np.nanmin(mean_effect)
    y_max = np.nanmax(mean_effect)

    y_margin = (y_max - y_min) * 0.20 if (y_max - y_min) > 0 else 0.5
    y_limits = (y_min - y_margin, y_max + y_margin) if not np.isnan(y_min) else (-1, 1)

    individual_thresholds = np.array([find_zero_crossing(c, common_grid) for c in site_ale_poly_fitted])
    n_curves_total = len(site_ale_poly_fitted)
    individual_thresholds = individual_thresholds[~np.isnan(individual_thresholds)]
    n_curves_crossing = len(individual_thresholds)

    if n_curves_crossing > 0:
        threshold_main = individual_thresholds.mean()
        threshold_std = individual_thresholds.std()
        threshold_sem = threshold_std / np.sqrt(n_curves_crossing)
    else:
        threshold_main = threshold_sem = np.nan

    # ==============================
    # STEP 5: CREATE FIGURE (both modes)
    # ==============================

    fig, gs, ax_all, axes_sub = plot.layout_5panels((13.86 * 0.9, 6.67 * 0.9), add_colorbar_ax=False)

    xlabel = rf'{BEAUTIFY[PLOT_FEATURE]} ($\sigma$)'
    ylabel = rf'ALE effect on daytime {BEAUTIFY[FLUX]} ($\sigma$)'

    # Main plot: individual site curves in background (light gray)
    for site_curve in site_ale_interpolated_array:
        ax_all.plot(common_grid, site_curve, color='gray', alpha=0.15, linewidth=0.8, zorder=1)

    # Add legend entries for threshold and gray curves
    ax_all.scatter([], [], edgecolors='black', s=200, linewidth=2,
                   marker='o', facecolors='none',
                   label='Mean zero-crossing of individual ALE curves')
    ax_all.plot([], [], color='gray', alpha=0.15, linewidth=0.8,
                label=f'Individual sites')

    # Add threshold marker
    if not np.isnan(threshold_main) and len(individual_thresholds) > 0:
        # Consensus threshold marker (open circle)
        ax_all.scatter(threshold_main, 0, edgecolors='black', s=200, linewidth=2,
                       marker='o', facecolors='none', zorder=100)

        # Format label based on USE_CI flag
        if USE_CI:
            _, ci_lower, ci_upper = calc_ci_95(individual_thresholds)
            label_text = f'Threshold\nx={threshold_main:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]'
        else:
            threshold_sem = individual_thresholds.std() / np.sqrt(len(individual_thresholds))
            label_text = f'Threshold\nx={threshold_main:.2f}±{threshold_sem:.2f} SEM'

        ax_all.annotate(label_text,
                        xy=(threshold_main, 0),
                        xytext=(threshold_main - 0.7, -0.5),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2, shrinkB=10),
                        ha='center', fontsize=AX_LABELS_FONTSIZE * 0.75,
                        color='black', zorder=11)

    else:
        print("WARNING: Threshold is NaN or no crossings found")

    ax_all.axhline(0, color='k', linestyle='--', linewidth=1)
    ax_all.set_ylim(y_limits)
    ax_all.set_xlabel(xlabel, fontsize=AX_LABELS_FONTSIZE)
    ax_all.set_ylabel(ylabel, fontsize=AX_LABELS_FONTSIZE)

    ax_all.legend(loc='best', fontsize=AX_LABELS_FONTSIZE)

    ax_all.text(0, 1.05, 'a', transform=ax_all.transAxes, size=AX_LABELS_FONTSIZE * 1.2, weight='bold')
    ax_all.text(0.05, 1.05, 'Global forests', transform=ax_all.transAxes, size=AX_LABELS_FONTSIZE * 1.2)
    ax_all.text(0.4, 1.05, f'(n={len(all_site_ale_curves)})', transform=ax_all.transAxes, size=AX_LABELS_FONTSIZE * 1.2)

    plot.format(ax=ax_all, fontsize=AX_LABELS_FONTSIZE, showyticklabels=True, showxticklabels=True,
                xtickdigits=0, ytickdigits=1, showbottomspine=True, showleftspine=True, showymajorticks=True)

    ax_all.spines['bottom'].set_color('black')
    ax_all.spines['left'].set_color('black')

    # Set x-axis to span only the data range (first to last valid point of mean curve)
    valid_indices = np.where(~np.isnan(mean_effect))[0]
    if len(valid_indices) > 0:
        x_min = common_grid[valid_indices[0]]
        x_max = common_grid[valid_indices[-1]]
        ax_all.set_xlim(x_min, x_max)

    # Reduce x-axis ticks for cleaner appearance
    from matplotlib.ticker import MaxNLocator

    ax_all.xaxis.set_major_locator(MaxNLocator(nbins=6))

    # IGBP subplots (simplified: same data for all)
    igbp_threshold_data = []  # Collect threshold data for table output

    configs = zip(
        axes_sub, IGBPS,
        [" ", " ", xlabel, xlabel],
        [" ", " ", " ", " "],
        ['b', 'c', 'd', 'e'],
        [True, False, True, False],
        [False, False, True, True]
    )

    for ax, igbp, xl, yl, letter, showyticklabels, showxticklabels in configs:
        if igbp in ale_interpolated:
            # Plot individual IGBP site curves in background
            if igbp in igbp_ale_interpolated:
                for site_curve in igbp_ale_interpolated[igbp]:
                    ax.plot(common_grid, site_curve, color='gray', alpha=0.15, linewidth=0.8, zorder=1)

            y_igbp = ale_interpolated[igbp]

            igbp_individual_thresholds = np.array([])
            if igbp in igbp_ale_interpolated:
                igbp_poly_fitted = fit_polynomial_to_curves(igbp_ale_interpolated[igbp], common_grid)
                igbp_individual_thresholds = np.array([find_zero_crossing(c, common_grid) for c in igbp_poly_fitted])
                igbp_individual_thresholds = igbp_individual_thresholds[~np.isnan(igbp_individual_thresholds)]

            # Auto-scale to own data range
            y_min_sub, y_max_sub = np.nanpercentile(y_igbp, [5, 95])
            y_margin_sub = (y_max_sub - y_min_sub) * 0.5 if y_max_sub > y_min_sub else 1.0
            ax.set_ylim(y_min_sub - y_margin_sub, y_max_sub + y_margin_sub)

            # Plot threshold marker and label (same format as main plot)
            if len(igbp_individual_thresholds) > 0:
                igbp_threshold_main = igbp_individual_thresholds.mean()
                igbp_sem = igbp_individual_thresholds.std() / np.sqrt(len(igbp_individual_thresholds))

                igbp_threshold_data.append({
                    'IGBP': igbp,
                    'Threshold': igbp_threshold_main,
                    'SEM': igbp_sem,
                    'N_crossing': len(igbp_individual_thresholds),
                    'N_total': len(igbp_poly_fitted),
                    'individual_thresholds': igbp_individual_thresholds
                })

                ax.scatter(igbp_threshold_main, 0, edgecolors='black', s=200, linewidth=2,
                           marker='o', facecolors='none', zorder=100)

                # Format label based on USE_CI flag
                if USE_CI:
                    _, ci_lower, ci_upper = calc_ci_95(igbp_individual_thresholds)
                    label_text = f'x={igbp_threshold_main:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]'
                else:
                    label_text = f'x={igbp_threshold_main:.2f}±{igbp_sem:.2f} SEM'

                ax.annotate(label_text,
                            xy=(igbp_threshold_main, 0), xytext=(igbp_threshold_main - 0.4, -0.5),
                            arrowprops=dict(arrowstyle='->', color='black', lw=2, shrinkB=10),
                            ha='center', fontsize=AX_LABELS_FONTSIZE * 0.75,
                            color='black', zorder=11)
            else:
                igbp_threshold_data.append({
                    'IGBP': igbp,
                    'Threshold': np.nan,
                    'SEM': np.nan,
                    'N_crossing': 0,
                    'N_total': len(igbp_ale_interpolated.get(igbp, [])),
                    'individual_thresholds': np.array([])
                })
        else:
            # Fallback: plot individual site curves only
            for site_curve in site_ale_interpolated_array:
                ax.plot(common_grid, site_curve, color='gray', alpha=0.15, linewidth=0.8, zorder=1)
            ax.set_ylim(y_limits)

        ax.axhline(0, color='k', linestyle='--', linewidth=1)
        ax.set_xlabel(xl, fontsize=AX_LABELS_FONTSIZE)
        ax.set_ylabel(yl, fontsize=AX_LABELS_FONTSIZE)
        ax.text(0, 1.1, letter, transform=ax.transAxes, size=AX_LABELS_FONTSIZE * 1.2, weight='bold')
        ax.text(0.1, 1.1, igbp, transform=ax.transAxes, size=AX_LABELS_FONTSIZE * 1.2)

        # Add n= for this IGBP
        n_igbp = len(igbp_ale_interpolated.get(igbp, []))
        if n_igbp > 0:
            ax.text(0.4, 1.1, f'(n={n_igbp})', transform=ax.transAxes, size=AX_LABELS_FONTSIZE * 1.2)

        plot.format(ax=ax, fontsize=AX_LABELS_FONTSIZE, showyticklabels=showyticklabels,
                    showxticklabels=showxticklabels,
                    xtickdigits=0, ytickdigits=1, showbottomspine=True, showleftspine=True, showymajorticks=True)

        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')

        # Set x-axis to data range for subplots
        ax.set_xlim(x_min, x_max)

        # Reduce x-axis ticks for subplots too
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    # Print comprehensive threshold summary table
    print("\n" + "=" * 95)
    print(f"THRESHOLD SUMMARY | Feature: {BEAUTIFY[PLOT_FEATURE]} | Target: {BEAUTIFY[FLUX]} | {'95% CI' if USE_CI else 'SEM'}")
    print("=" * 95)

    if USE_CI:
        print(f"{'Group':<15} {'Threshold':<15} {'95% CI':<30} {'N crossing':<15} {'N total':<10}")
        print("-" * 95)
        if n_curves_crossing > 0:
            _, ci_lower, ci_upper = calc_ci_95(individual_thresholds)
            print(f"{'GLOBAL':<15} {threshold_main:>8.4f}       [{ci_lower:>7.4f}, {ci_upper:>7.4f}]     "
                  f"{n_curves_crossing:>3}/{n_curves_total:<3}            {n_curves_total:>6}")
        else:
            print(f"{'GLOBAL':<15} {'N/A':>14} {'N/A':>28} {0:>3}/{n_curves_total:<3}            {n_curves_total:>6}")

        for data in igbp_threshold_data:
            if not np.isnan(data['Threshold']):
                # **Use per-IGBP thresholds for CI95 calculation**
                _, ci_lower, ci_upper = calc_ci_95(data['individual_thresholds'])
                print(f"{data['IGBP']:<15} {data['Threshold']:>8.4f}       [{ci_lower:>7.4f}, {ci_upper:>7.4f}]     "
                      f"{data['N_crossing']:>3}/{data['N_total']:<3}            {data['N_total']:>6}")
            else:
                print(f"{data['IGBP']:<15} {'N/A':>14} {'N/A':>28} {0:>3}/{data['N_total']:<3}            {data['N_total']:>6}")
    else:
        print(f"{'Group':<15} {'Threshold':<15} {'SEM':<12} {'N crossing':<15} {'N total':<10}")
        print("-" * 95)
        if n_curves_crossing > 0:
            threshold_sem = individual_thresholds.std() / np.sqrt(n_curves_crossing)
            print(f"{'GLOBAL':<15} {threshold_main:>8.4f}       {threshold_sem:>8.4f}     "
                  f"{n_curves_crossing:>3}/{n_curves_total:<3}            {n_curves_total:>6}")
        else:
            print(f"{'GLOBAL':<15} {'N/A':>14} {'N/A':>11} {0:>3}/{n_curves_total:<3}            {n_curves_total:>6}")

        for data in igbp_threshold_data:
            if not np.isnan(data['Threshold']):
                print(f"{data['IGBP']:<15} {data['Threshold']:>8.4f}       {data['SEM']:>8.4f}     "
                      f"{data['N_crossing']:>3}/{data['N_total']:<3}            {data['N_total']:>6}")
            else:
                print(f"{data['IGBP']:<15} {'N/A':>14} {'N/A':>11} {0:>3}/{data['N_total']:<3}            {data['N_total']:>6}")

    print("=" * 95 + "\n")

    fig.tight_layout()
    gs.update(wspace=.1)

    if SHOW_PLOT:

        fig.show()
    # Save figure. Commented out on 25 April 2026, in the same commit that changed the
    # layout, so the PNG on disk kept the old layout while the code moved on.
    #
    # The number is a placeholder. This was Extended Data Fig. 2 in the submitted
    # version, Extended Data is not a Nature Communications category (T26), so it
    # becomes a supplementary figure and X stands in until the number is assigned.
    outfilepath = dir_out / f'70_SUPPFIG-X_ALE_ResponseCurve_{PLOT_FEATURE}_{FLUX}.png'
    fig.savefig(outfilepath, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {outfilepath}\n")

    # Save threshold results
    # All sites, then one row per forest type. The per-type thresholds were already
    # computed for the subplot labels but were not written out, so they could not be
    # compared with the SHAP or the measured thresholds.
    rows = [{
        'feature': PLOT_FEATURE,
        'target': FLUX,
        'method': 'Direct zero-crossing',
        'group': 'ALL SITES',
        'threshold': f'{threshold_main:.3f}',
        'ci_lower': '',
        'ci_upper': '',
        'n_sites': len(all_site_ale_curves),
        'mode': 'Single curve'
    }]
    for entry in igbp_threshold_data:
        _, ci_lo, ci_hi = calc_ci_95(entry['individual_thresholds'])
        rows.append({
            'feature': PLOT_FEATURE,
            'target': FLUX,
            'method': 'Per-site zero-crossings, mean',
            'group': entry['IGBP'],
            'threshold': f"{entry['Threshold']:.3f}",
            'ci_lower': f'{ci_lo:.3f}',
            'ci_upper': f'{ci_hi:.3f}',
            'n_sites': entry['N_crossing'],
            'mode': 'Per-site curves'
        })
    df_threshold = pd.DataFrame(rows)

    threshold_path = dir_out / f'70_SUPPFIG-X_ALE_ResponseCurve_{PLOT_FEATURE}_{FLUX}_THRESHOLD.csv'
    df_threshold.to_csv(threshold_path, index=False)
    print(f"Saved threshold results to: {threshold_path}")
