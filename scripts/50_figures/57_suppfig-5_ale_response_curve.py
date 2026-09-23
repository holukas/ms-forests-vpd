"""
Supplementary Fig. 5: ALE response curves of NEP to VPD and the ALE threshold.

Panel a shows the per-site curves of all sites, panels b to e those of one forest type
each. Each site curve is smoothed with a fourth-order polynomial, and the threshold is the
mean of the sites' first positive-to-negative zero crossings. Sites without a crossing do
not enter the threshold, and the panel title then gives both counts.

Settings:
- PLOT_FEATURE = 'VPD_ZSCORE', FLUX = 'NEP_ZSCORE'
- VARIANT: empty for the main analysis, otherwise the value script 46 ran with
- USE_MEDIAN: aggregate the site curves by median instead of mean
- USE_CI: label the threshold with its 95% confidence interval (True) or its SEM (False)

Reads 46_ALE_SiteCurves_<PLOT_FEATURE>_<FLUX>.parquet, written by
40_aggregation/46_ale_thresholds.py. Writes
57_SUPPFIG-5_ALE_ResponseCurve_<PLOT_FEATURE>_<FLUX>.png and prints a threshold summary;
the threshold table itself comes from script 46.
"""
from pathlib import Path

import numpy as np
import pandas as pd

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

# Run variant. An empty string reads the results of the main analysis and
# writes to the baseline plot folder. Any other value reads the matching variant
# folder and writes the figures next to it, so a sensitivity run cannot overwrite a
# figure of the main analysis. The aggregation must have run with the same value.
VARIANT = ""
PLOT_FEATURE = 'VPD_ZSCORE'
USE_MEDIAN = False
USE_CI = True  # True: 95% CI, False: SEM
IGBPS = ['ENF', 'DBF', 'MF', 'EBF']
AX_LABELS_FONTSIZE = 12

BEAUTIFY = {
    "NEP_ZSCORE": "NEP", "ET_ZSCORE": "ET", "GPP_ZSCORE": "GPP", "RECO_ZSCORE": "RECO",
    "TA_ZSCORE": "TA", "VPD_ZSCORE": "VPD", "SWC_ZSCORE": "SWC", "SWIN_ZSCORE": "SWIN",
}

settings = load_settings()
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / 'ale' / VARIANT

agg_name = "median" if USE_MEDIAN else "mean"
print(f"\n{'=' * 80}\nALE Response Curves | Feature: {BEAUTIFY[PLOT_FEATURE]} | Aggregation: {agg_name}\n{'=' * 80}\n")
dir_out.mkdir(parents=True, exist_ok=True)


# ==============================
# LOAD WHAT STAGE 46 WROTE
# ==============================
# The per-site curves used to be built here, by reading 208 ALE files and interpolating
# them. That is aggregation, so it moved to `40_aggregation/46_ale_thresholds.py` and this
# script only draws. The variable names below are the ones the plotting code already used.

curves_file = (Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / 'ale' / VARIANT /
               f'46_ALE_SiteCurves_{PLOT_FEATURE}_{FLUX}.parquet')
if not curves_file.is_file():
    raise FileNotFoundError(f"No curves at {curves_file}. Run "
                            f"40_aggregation/46_ale_thresholds.py first.")

stored = pd.read_parquet(curves_file)
common_grid = np.array([float(c) for c in stored.columns if c != 'IGBP'])
site_igbp_map = stored['IGBP'].to_dict()
all_site_ale_curves = list(stored.index)          # only the count is used below

site_ale_interpolated = [row for row in stored.drop(columns='IGBP').to_numpy(float)]
igbp_ale_interpolated = {igbp: [] for igbp in IGBPS}
for site, curve in zip(stored.index, site_ale_interpolated):
    igbp = site_igbp_map.get(site)
    if igbp in igbp_ale_interpolated:
        igbp_ale_interpolated[igbp].append(curve)

print(f"{len(site_ale_interpolated)} site curves on a grid of {len(common_grid)} points")

ale_interpolated = {}
if len(all_site_ale_curves) > 0:
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
    # Both counts, because the threshold is the mean over the sites that cross, not over
    # every site drawn. US-Cwt is the one that does not: its curve rises with VPD, so it
    # has no positive to negative crossing.
    _n_label = (f'(n={n_curves_total})' if n_curves_crossing == n_curves_total
                else f'(n={n_curves_total}, {n_curves_crossing} crossing)')
    ax_all.text(0.4, 1.05, _n_label, transform=ax_all.transAxes, size=AX_LABELS_FONTSIZE * 1.2)

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

                # Anchored in axes coordinates, not data coordinates. Placing it a fixed
                # distance left of the marker pushed it into the y axis whenever the
                # threshold sat near the left edge, which is most panels.
                ax.annotate(label_text,
                            xy=(igbp_threshold_main, 0), xycoords='data',
                            xytext=(0.06, 0.12), textcoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', color='black', lw=2, shrinkB=10),
                            ha='left', va='center', fontsize=AX_LABELS_FONTSIZE * 0.75,
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
        n_igbp_crossing = len(igbp_individual_thresholds)
        if n_igbp > 0:
            label = (f'(n={n_igbp})' if n_igbp_crossing == n_igbp
                     else f'(n={n_igbp}, {n_igbp_crossing} crossing)')
            ax.text(0.4, 1.1, label, transform=ax.transAxes, size=AX_LABELS_FONTSIZE * 1.2)

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
    # Save figure.
    outfilepath = dir_out / f'57_SUPPFIG-5_ALE_ResponseCurve_{PLOT_FEATURE}_{FLUX}.png'
    fig.savefig(outfilepath, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {outfilepath}\n")

    # The threshold table is not written here any more. Stage 46 owns it, and two
    # scripts writing the same numbers is how they drift apart.
