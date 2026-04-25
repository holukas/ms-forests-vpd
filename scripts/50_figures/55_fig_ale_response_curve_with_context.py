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
4. **Confidence**: ALE thresholds use direct zero-crossing with bootstrap CI;
   SHAP thresholds depend on bin structure
5. **Aggregation**: Can use mean (default) or median (USE_MEDIAN=True) for robustness to outlier sites

## Structure

- **Main plot**: All sites aggregated (black mean ± gray percentile band), overlaid
  curves by temperature regime (blue=cold → red=hot)
- **4 subplots**: Same data, separated by IGBP type (ENF, DBF, MF, EBF) for
  ecosystem-specific patterns
- **Threshold**: Direct zero-crossing point where mean ALE curve crosses zero
- **Bootstrap CI**: 95% confidence interval via bootstrap resampling (1000 iterations)
- **Error bands**: IQR (25th-75th percentile) showing uncertainty across sites

## Use Case

Compare with Script 54 to validate findings:
- Script 54: "How important is VPD for predicting NEP?"
- Script 55: "How does NEP respond to changes in VPD? At what VPD does it flip?"
- Together: Strong agreement between SHAP importance and ALE threshold confidence
"""
from pathlib import Path

import diive as dv
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import src.files as files
import src.plot as plot


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
    # Only consider points with valid data
    curve_valid = curve[valid]
    grid_valid = grid[valid]
    # Check if curve actually crosses zero (has positive and negative values)
    if np.all(curve_valid >= 0) or np.all(curve_valid <= 0):
        return np.nan
    sign_changes = np.diff(np.sign(curve_valid))
    for idx in np.where(sign_changes != 0)[0]:
        if curve_valid[idx] > 0 and curve_valid[idx + 1] <= 0:
            x1, x2 = grid_valid[idx], grid_valid[idx + 1]
            y1, y2 = curve_valid[idx], curve_valid[idx + 1]
            return x1 - y1 * (x2 - x1) / (y2 - y1) if (y2 - y1) != 0 else (x1 + x2) / 2
    return np.nan


# ==============================
# CONFIGURATION
# ==============================

FLUX = 'NEP_ZSCORE'
PLOT_FEATURE = 'VPD_ZSCORE'
FAST_TEST_MODE = False
USE_MEDIAN = False
IGBPS = ['ENF', 'DBF', 'MF', 'EBF']
AX_LABELS_FONTSIZE = 12

BEAUTIFY = {
    "NEP_ZSCORE": "NEP", "ET_ZSCORE": "ET", "GPP_ZSCORE": "GPP", "RECO_ZSCORE": "RECO",
    "TA_ZSCORE": "TA", "VPD_ZSCORE": "VPD", "SWC_ZSCORE": "SWC", "SWIN_ZSCORE": "SWIN",
}

settings = files.read_settings_file("../../config/settings.yaml")
dir_ale_results = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / 'ale'
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / 'ale'

agg_name = "median" if USE_MEDIAN else "mean"
print(f"\n{'=' * 80}\nALE Response Curves | Feature: {BEAUTIFY[PLOT_FEATURE]} | Aggregation: {agg_name}\n{'=' * 80}\n")
dir_out.mkdir(parents=True, exist_ok=True)

subsets_df = pd.read_csv('../../data/outputs/20_subsets/21_SUBSETS_parquet_vars_stats_subsets.csv')

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
        site_data = dv.load_parquet(filepath, sanitize_timestamp=False)
    except Exception:
        continue

    if PLOT_FEATURE not in site_data.columns or FLUX not in site_data.columns:
        continue

    valid_idx = ~(site_data[PLOT_FEATURE].isna() | site_data[FLUX].isna())
    site_data_clean = site_data[valid_idx].copy()

    if len(site_data_clean) > 10:
        all_site_data[site] = site_data_clean

print(f"Loaded {len(all_site_data)} sites (each will contribute equally)\n")

# ==============================
# STEP 2: CALCULATE ALE
# ==============================

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

        if np.sum(insufficient_mask) > 0:
            print(f"Masked {np.sum(insufficient_mask)}/{len(common_grid)} points with <50% coverage")
        print(f"Aggregated {len(site_ale_poly_fitted)} sites using {agg_name}\n")

        for igbp in IGBPS:
            if len(igbp_ale_interpolated[igbp]) > 0:
                igbp_poly_fitted = fit_polynomial_to_curves(igbp_ale_interpolated[igbp], common_grid)
                igbp_array = np.array(igbp_poly_fitted)
                igbp_mean = agg_func(igbp_array, axis=0)

                min_sites_igbp = len(igbp_poly_fitted) / 2
                igbp_coverage = np.sum(~np.isnan(igbp_array), axis=0)
                igbp_mean[igbp_coverage < min_sites_igbp] = np.nan

                ale_interpolated[igbp] = igbp_mean
                print(f"  {igbp}: {len(igbp_poly_fitted)} sites")
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
        print(f"Threshold: {threshold_main:.3f}±{threshold_sem:.3f} SEM")
        print(f"  {n_curves_crossing}/{n_curves_total} individual curves cross zero")
        print(f"  Range: [{individual_thresholds.min():.3f}, {individual_thresholds.max():.3f}]")
    else:
        threshold_main = threshold_sem = np.nan
        print(f"WARNING: None of the {n_curves_total} individual curves cross zero")

    # ==============================
    # STEP 5: CREATE FIGURE (both modes)
    # ==============================

    fig, gs, ax_all, axes_sub = plot.layout_5panels((13.86, 6.67), add_colorbar_ax=False)

    xlabel = rf'{BEAUTIFY[PLOT_FEATURE]} ($\sigma$)'
    ylabel = rf'ALE effect on daytime {BEAUTIFY[FLUX]} ($\sigma$)'

    # Main plot: individual site curves in background (light gray)
    for site_curve in site_ale_interpolated_array:
        ax_all.plot(common_grid, site_curve, color='gray', alpha=0.15, linewidth=0.8, zorder=1)

    # Add both thresholds: consensus (SD error bar) and mean curve polynomial crossing
    if not np.isnan(threshold_main) and len(individual_thresholds) > 0:
        threshold_sd = individual_thresholds.std()
        threshold_sem = threshold_sd / np.sqrt(len(individual_thresholds))

        # Consensus threshold marker (open circle)
        ax_all.scatter(threshold_main, 0, c='none', edgecolors='black', s=200, linewidth=2,
                       marker='o', facecolors='none', zorder=100,
                       label=f'Mean zero-crossing of individual ALE curves')

        # Consensus threshold label (show SD for reference)
        label_text = f'Threshold\nx={threshold_main:.2f}±{threshold_sem:.2f} SEM'

        ax_all.annotate(label_text,
                        xy=(threshold_main, 0),
                        xytext=(threshold_main - 0.7, -0.5),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2, shrinkB=10),
                        ha='center', fontsize=AX_LABELS_FONTSIZE * 0.75,
                        color='black', zorder=11)

    else:
        print("WARNING: Threshold is NaN or no crossings found")

    ax_all.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax_all.set_ylim(y_limits)
    ax_all.set_xlabel(xlabel, fontsize=AX_LABELS_FONTSIZE)
    ax_all.set_ylabel(ylabel, fontsize=AX_LABELS_FONTSIZE)

    ax_all.legend(loc='best', fontsize=AX_LABELS_FONTSIZE)

    ax_all.text(0, 1.05, 'a', transform=ax_all.transAxes, size=AX_LABELS_FONTSIZE * 1.2, weight='bold')
    ax_all.text(0.05, 1.05, 'Global forests', transform=ax_all.transAxes, size=AX_LABELS_FONTSIZE * 1.2)

    plot.format(ax=ax_all, fontsize=AX_LABELS_FONTSIZE, showyticklabels=True, showxticklabels=True,
                xtickdigits=1, ytickdigits=1, showbottomspine=True, showleftspine=True, showymajorticks=True)

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
                    'N_total': len(igbp_poly_fitted)
                })

                ax.scatter(igbp_threshold_main, 0, c='none', edgecolors='black', s=200, linewidth=2,
                           marker='o', facecolors='none', zorder=100)
                label_text = f'Threshold\nx={igbp_threshold_main:.2f}±{igbp_sem:.2f} SEM'
                ax.annotate(label_text,
                            xy=(igbp_threshold_main, 0), xytext=(igbp_threshold_main - 0.7, -0.5),
                            arrowprops=dict(arrowstyle='->', color='black', lw=2, shrinkB=10),
                            ha='center', fontsize=AX_LABELS_FONTSIZE * 0.75,
                            color='black', zorder=11)
            else:
                igbp_threshold_data.append({
                    'IGBP': igbp,
                    'Threshold': np.nan,
                    'SEM': np.nan,
                    'N_crossing': 0,
                    'N_total': len(igbp_ale_interpolated.get(igbp, []))
                })
        else:
            # Fallback: plot individual site curves only
            for site_curve in site_ale_interpolated_array:
                ax.plot(common_grid, site_curve, color='gray', alpha=0.15, linewidth=0.8, zorder=1)
            ax.set_ylim(y_limits)

        ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel(xl, fontsize=AX_LABELS_FONTSIZE)
        ax.set_ylabel(yl, fontsize=AX_LABELS_FONTSIZE)
        ax.text(0, 1.1, letter, transform=ax.transAxes, size=AX_LABELS_FONTSIZE * 1.2, weight='bold')
        ax.text(0.1, 1.1, igbp, transform=ax.transAxes, size=AX_LABELS_FONTSIZE * 1.2)

        plot.format(ax=ax, fontsize=AX_LABELS_FONTSIZE, showyticklabels=showyticklabels,
                    showxticklabels=showxticklabels,
                    xtickdigits=1, ytickdigits=1, showbottomspine=True, showleftspine=True, showymajorticks=True)

        # Set x-axis to data range for subplots
        ax.set_xlim(x_min, x_max)

        # Reduce x-axis ticks for subplots too
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    # Print comprehensive threshold summary table
    print("\n" + "=" * 95)
    print(f"THRESHOLD SUMMARY | Feature: {BEAUTIFY[PLOT_FEATURE]} | Target: {BEAUTIFY[FLUX]}")
    print("=" * 95)
    print(f"{'Group':<15} {'Threshold':<15} {'SEM':<12} {'N crossing':<15} {'N total':<10}")
    print("-" * 95)

    # Global threshold
    if n_curves_crossing > 0:
        print(f"{'GLOBAL':<15} {threshold_main:>8.4f}       {threshold_sem:>8.4f}     "
              f"{n_curves_crossing:>3}/{n_curves_total:<3}            {n_curves_total:>6}")
    else:
        print(f"{'GLOBAL':<15} {'N/A':>14} {'N/A':>11} {0:>3}/{n_curves_total:<3}            {n_curves_total:>6}")

    # IGBP-specific thresholds
    for data in igbp_threshold_data:
        if not np.isnan(data['Threshold']):
            print(f"{data['IGBP']:<15} {data['Threshold']:>8.4f}       {data['SEM']:>8.4f}     "
                  f"{data['N_crossing']:>3}/{data['N_total']:<3}            {data['N_total']:>6}")
        else:
            print(f"{data['IGBP']:<15} {'N/A':>14} {'N/A':>11} {0:>3}/{data['N_total']:<3}            {data['N_total']:>6}")

    print("=" * 95 + "\n")

    fig.tight_layout()
    gs.update(wspace=.1)

    fig.show()

    # # Save figure
    # outfilepath = dir_out / f'55_FIG_ALE_ResponseCurve_{PLOT_FEATURE}_Threshold_{FLUX}.png'
    # fig.savefig(outfilepath, dpi=300, bbox_inches='tight')
    # print(f"Saved figure to: {outfilepath}\n")

    # Save threshold results
    df_threshold = pd.DataFrame([{
        'feature': PLOT_FEATURE,
        'target': FLUX,
        'method': 'Direct zero-crossing',
        'threshold': f'{threshold_main:.3f}',
        'n_sites': len(all_site_ale_curves),
        'mode': 'Single curve'
    }])

    threshold_path = dir_out / f'55_FIG_ALE_ResponseCurve_{PLOT_FEATURE}_{FLUX}_THRESHOLD.csv'
    df_threshold.to_csv(threshold_path, index=False)
    print(f"Saved threshold results to: {threshold_path}")
