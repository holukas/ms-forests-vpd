import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from scipy.stats import gaussian_kde

from src.common import findpoi


def show_shap_thresholds(ax, x_fit, y_fit, max_ix, min_ix, idx, ydim_max, ydim_min, show_annotate, show_annotate_short,
                         fontsize):
    y_top = ax.get_ylim()[-1]
    y_bottom = ax.get_ylim()[0]

    _params = dict(color='black', linestyle='--', linewidth=1, zorder=100)
    _params2 = dict(linewidth=2, zorder=100, s=200, alpha=1)
    _params3 = dict(color='black', fontsize=fontsize, linespacing=1.2)

    # Calculate y-offset for lines leading to annotations
    # Needs to be a bit larger for the smaller subplots
    # As relative_height goes down (0.2), the factor goes UP
    # As relative_height goes up (1.0), the factor goes DOWN
    bbox = ax.get_position()
    relative_height = bbox.height  # Height of ax as a fraction of figure (0.0 to 1.0)
    y_range = y_top - y_bottom
    scaling_factor = 0.02 + (0.04 / relative_height)
    scaling_factor = min(0.08, scaling_factor)  # Clamp so it does not get huge on small plots
    y_offset_topline = y_range * scaling_factor

    # Maximum positive effect
    # ax_all.scatter(x_fit[max_ix], y_fit[max_ix], color='black', marker='^', edgecolors='none', **_params2)
    ax.scatter(x_fit[max_ix], y_fit[max_ix], color='none', marker='^', edgecolor='black', **_params2)
    if show_annotate:
        # Get the y-position for the text below the plotted points
        text_y_pos_max = y_fit[max_ix] - 0.45 * (ydim_max - ydim_min)
        if show_annotate_short:
            showntext = f'(x={x_fit[max_ix]:.2f})'
        else:
            showntext = f'Maximum positive effect\n(x={x_fit[max_ix]:.2f})'
        ax.text(x_fit[max_ix], text_y_pos_max, showntext,
                va='bottom', ha='center', **_params3)
        ax.plot([x_fit[max_ix], x_fit[max_ix]], [text_y_pos_max + y_offset_topline, y_fit[max_ix] - 0], **_params)
        ax.plot([x_fit[max_ix], x_fit[max_ix]], [y_bottom, text_y_pos_max], **_params)

    # Threshold
    ax.scatter(x_fit[idx], y_fit[idx], c="none", edgecolors='black', **_params2)
    if show_annotate:
        text_y_pos_zero = y_fit[idx] - 0.45 * (ydim_max - ydim_min)
        if show_annotate_short:
            showntext = f'(x={x_fit[idx]:.2f})'
        else:
            showntext = f'Threshold\n(x={x_fit[idx]:.2f})'
        ax.text(x_fit[idx], text_y_pos_zero, showntext,
                va='bottom', ha='center', **_params3)
        ax.plot([x_fit[idx], x_fit[idx]], [text_y_pos_zero + y_offset_topline, y_fit[idx] - 0], **_params)
        ax.plot([x_fit[idx], x_fit[idx]], [y_bottom, text_y_pos_zero], **_params)

    # Maximum negative impact
    # ax_all.scatter(x_fit[min_ix], y_fit[min_ix], color='black', marker='_', edgecolors='none', **_params2)
    ax.scatter(x_fit[min_ix], y_fit[min_ix], color='none', marker='v', edgecolor='black', **_params2)
    if show_annotate:
        text_y_pos_min = ydim_min * 0.99
        # text_y_pos_min = y_fit[min_ix] - 0.15 * (ydim_max - ydim_min) + show_annotate_short_yoffset[2]
        if show_annotate_short:
            showntext = f'(x={x_fit[min_ix]:.2f})'
        else:
            showntext = f'Maximum negative effect\n(x={x_fit[min_ix]:.2f})'
        ax.text(x_fit[min_ix] + 0.2, text_y_pos_min, showntext,
                va='bottom', ha='right', **_params3)
        ax.plot([x_fit[min_ix], x_fit[min_ix]], [text_y_pos_min + y_offset_topline, y_fit[min_ix] - 0], **_params)
        ax.plot([x_fit[min_ix], x_fit[min_ix]], [y_bottom, text_y_pos_min], **_params)


def format(ax, fontsize, showxticklabels, showyticklabels, xtickdigits, ytickdigits):
    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)

    # Set the tick width for both x and y axes
    ax.tick_params(axis='both', which='major', width=1, length=5, labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', width=1, length=2, labelsize=fontsize)

    # Visibility for ticklabels
    if showxticklabels:
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(f"{f'%.{xtickdigits}f'}"))
        ax.tick_params(axis='x', labelbottom=True)
    else:
        ax.tick_params(axis='x', labelbottom=False)  # Hide labels

    if showyticklabels:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(f"{f'%.{ytickdigits}f'}"))
        ax.tick_params(axis='y', labelleft=True)
    else:
        ax.tick_params(axis='y', labelleft=False)  # Hide labels

    ax.grid(False)


def add_fit(ax, x_fit, y_fit, pi_lower, pi_upper, poly_func, r_squared, show_annotate, fontsize):
    # Plot fitted polynomial curve
    ax.plot(x_fit, y_fit, color='#004e98', linewidth=3, zorder=99)
    # label=rf'$y = {poly_coeffs[0]:.4f}x^4 - {poly_coeffs[1]:.4f}x^3 + {poly_coeffs[2]:.4f}x^2 + {poly_coeffs[3]:.4f}x - {poly_coeffs[4]:.4f}$'

    # Plot prediction interval
    fillbetweenplot = ax.fill_between(x_fit, pi_lower, pi_upper, color='#004e98', alpha=0.2,
                                      label='95% prediction interval', zorder=1)

    if show_annotate:
        # Add an arrow to the fitted line
        # Find a point on the line to place the arrow.
        # Let's place it a little past the middle of the x-range.
        arrow_x = 3
        arrow_y = poly_func(arrow_x)
        # Find a point slightly to the left to define the arrow direction
        tail_x = arrow_x - 0.1
        tail_y = poly_func(tail_x)
        # Calculate the angle of the line at this point to get the correct arrow orientation
        angle = np.arctan2(arrow_y - tail_y, arrow_x - tail_x) * 180 / np.pi
        ax.annotate(f'Fitted 4th degree\npolynomial (r$^2$={r_squared:.2f})',
                    xy=(arrow_x, arrow_y),
                    xytext=(arrow_x - 0.2, arrow_y + 0.4),  # Adjust text position as needed
                    arrowprops=dict(arrowstyle="->", color='#004e98', lw=1.5),
                    fontsize=fontsize, color='#004e98', ha='left', va='center', zorder=100)
    return fillbetweenplot


def layout_5panels(figsize, add_colorbar_ax: bool = False):
    fig = plt.figure(figsize=figsize, dpi=150, facecolor="white")
    ncols = 5 if add_colorbar_ax else 5
    width_ratios = [1, 1, 1, 1, 0.1] if add_colorbar_ax else [1, 1, 0.1, 1, 1]
    gs = gridspec.GridSpec(2, ncols, width_ratios=width_ratios)
    ax_all = fig.add_subplot(gs[0:2, 0:2])

    if add_colorbar_ax:
        axes_sub = [fig.add_subplot(gs[r, c], sharex=ax_all, sharey=ax_all) for r, c in
                    [(0, 2), (0, 3), (1, 2), (1, 3)]]
        cax = fig.add_subplot(gs[:, 4])
        return fig, gs, ax_all, axes_sub, cax
    else:
        axes_sub = [fig.add_subplot(gs[r, c], sharex=ax_all, sharey=ax_all) for r, c in
                    [(0, 3), (0, 4), (1, 3), (1, 4)]]
        return fig, gs, ax_all, axes_sub


def plot_markers(ax, df, xvals, yvals, zvals, flux_txt, ax_labels_fontsize, area_size, annotate=False):
    """Finds min/max regions, plots markers, and optionally adds arrows."""
    piv = df.pivot(index=xvals, columns=yvals, values=zvals)
    locs = {m: findpoi(df=piv, k=area_size, agg='mean', what=m)[0] for m in ['max', 'min']}

    for m_type, (lx, ly) in locs.items():
        x, y = lx + 0.05, ly + 0.05
        marker = '+' if m_type == 'max' else '_'
        ax.scatter(x, y, c='k', marker=marker, lw=3, s=650, zorder=100, alpha=0.5)
        ax.scatter(x, y, facecolors='none', edgecolors='k', marker='o', lw=3, s=650, zorder=100, alpha=0.5)

        if annotate:
            txt, y_off = (f'highest {flux_txt} increase', 2) if m_type == 'max' else (
                f'highest {flux_txt} decrease', 0.9)
            tx_pos = (x - 3, y + y_off) if m_type == 'max' else (x - 1, y + y_off)
            ha = 'left' if m_type == 'max' else 'center'
            ax.annotate(txt, xy=(x, y), xytext=tx_pos, arrowprops=dict(arrowstyle="->", color='k', lw=3, shrinkB=15),
                        fontsize=ax_labels_fontsize, color='black', ha=ha, va='center', zorder=100)
    # print(f"Max: {locs['max']}, Min: {locs['min']}")
    return locs


def style_ax(ax, title, ax_labels_fontsize):
    ax.text(0.03 if 'All' in title else 0.06, 0.98 if 'All' in title else 1, title, transform=ax.transAxes,
            size=ax_labels_fontsize, ha='left', va='bottom' if 'All' in title else 'top',
            zorder=100, backgroundcolor='none')


def plot_scenario_panel(ax, df, feature_col, color, columns, n_scenarios, y_limits, is_top_row, group_name,
                        scenario_labels, show_x=False, show_y=False, is_main=False, is_first=False):
    # SHAP value means per site
    pivot = df.pivot(index='SITE', columns='SCENARIO', values=feature_col).reindex(columns=columns)

    # SHAP value SDs per site
    sd_col = feature_col.replace('_AVG', '_SD')
    pivot_sd = df.pivot(index='SITE', columns='SCENARIO', values=sd_col).reindex(columns=columns)

    if pivot.dropna(how='all').empty:
        ax.set_visible(False)
        return

    # Stats: collect in table
    stats_list = []
    for col in [1, 4, 5]:

        # Drop NaNs for the specific scenario
        data_vec = pivot[col].dropna()

        # Selects rows from data_sd where the index exists in data
        data_sd_vec = pivot_sd[col].copy()
        data_sd_vec = data_sd_vec.loc[data_sd_vec.index.intersection(data_vec.index)]

        # Calculate total SD using the law of total variance (sqrt(mean of variances + variance of means)).
        # This approach treats sites as equally representative (macro-average), normalizing
        # differences in sample counts between sites.
        mean_of_variances = (data_sd_vec ** 2).mean()  # Average of within-site variances
        variance_of_means = data_vec.var(ddof=0)  # Variance of the site means
        total_sd = np.sqrt(mean_of_variances + variance_of_means)  # Global SD

        if not data_vec.empty:
            stats_list.append({
                'IGBP': group_name,
                'Feature': feature_col,
                'n_sites': len(data_vec),
                'Scenario': col,
                'Mean': data_vec.mean(),
                'total_SD': total_sd,
                'SEM': data_vec.sem(),
                'Median': data_vec.median(),
                'Min': data_vec.min(),
                'Max': data_vec.max(),
                'P25': data_vec.quantile(0.25),
                'P75': data_vec.quantile(0.75),
                'Sites < 0': (data_vec < 0).sum(),
                '% < 0': (data_vec < 0).mean() * 100
            })
    # Display as table
    featurestats_df = pd.DataFrame(stats_list)

    means = featurestats_df['Mean']
    p25 = featurestats_df['P25']
    p75 = featurestats_df['P75']

    x_coords = np.arange(n_scenarios)

    # Ghost lines (faint)
    alpha_ghost = 0.1 if is_main else 0.15
    lw_ghost = 0.5
    ax.plot(x_coords, pivot.T.values, color='gray', alpha=alpha_ghost, linewidth=lw_ghost, zorder=1)

    # IQR ribbon
    ax.fill_between(x_coords, p25, p75, color=color, alpha=0.25, linewidth=0, zorder=2)

    # Sina / jitter points
    # Controlled jitter that respects density but stays tight
    for x_i, scen in enumerate(columns):
        if scen not in pivot:
            continue
        data_vec = pivot[scen].dropna()
        if len(data_vec) < 2:
            ax.scatter([x_i] * len(data_vec), data_vec, color=color, s=2, alpha=0.5, zorder=3)
            continue

        kde = gaussian_kde(data_vec)
        density = kde(data_vec)
        # Normalize width for jitter
        width_factor = 0.15
        width = (density / density.max()) * width_factor
        rng = np.random.RandomState(42 + x_i)
        jitter = rng.uniform(-1, 1, size=len(data_vec)) * width

        # Plot points
        s_sina = 4 if is_main else 4
        alpha_sina = 0.4 if is_main else 0.5
        ax.scatter(x_i + jitter, data_vec, color=color, s=s_sina, alpha=alpha_sina, linewidth=0, zorder=3)

        n_sites = len(data_vec)

        # Sample size annotation, show in first row only
        if is_top_row:
            ax.text(x_coords[x_i], 0.9, f'n={n_sites}',
                    fontsize=7, color='#555555', ha='center', va='center')

        # Percentage of sites below zero (i.e., negatively affected)
        n_sites_below_zero = data_vec[data_vec < 0].count()
        perc_n_sites_below_zero = n_sites_below_zero / n_sites * 100

        # Decide text label
        text = f'{perc_n_sites_below_zero:.0f}%'

        # Implement percentage pill background
        alpha = 0.8 if perc_n_sites_below_zero > 70 else 0.5
        ax.text(x_coords[x_i], -1.38, text,
                fontsize=7,
                color='white',  # White text for high contrast inside the pill
                fontweight='bold',  # Bold to make it pop
                ha='center', va='center',
                zorder=10,  # Ensure it stays on top of all lines
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=color,  # Match the variable's color (VPD gold, etc.)
                    edgecolor='none',  # No border for a cleaner look
                    alpha=alpha  # Slight transparency to stay approachable
                ))

    # Mean trend line and nodes
    lw_trend = 2.0
    s_node = 25
    ax.plot(x_coords, means, color=color, linewidth=lw_trend, alpha=1.0, zorder=5)
    ax.scatter(x_coords, means, facecolor=color, edgecolor='white', linewidth=1.0, s=s_node, zorder=6)

    # Formatting
    ax.set_ylim(y_limits)
    ax.axhspan(y_limits[0], 0, facecolor='#f0f0f0', alpha=0.6, zorder=0)  # Shade the negative region
    ax.axhline(0, color='black', linestyle='--', linewidth=0.6, alpha=0.6, zorder=0)

    # Format x-axis
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks(x_coords)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    # Tick Styling
    if show_x:
        ax.set_xticklabels(scenario_labels, rotation=45, ha='right', color='black')
        ax.tick_params(axis='x', length=4, width=0.8)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)

    if show_y:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))  # Only 1 digit after comma
        ax.tick_params(axis='y', colors='black', length=3, direction='out')
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)

    return featurestats_df


def flameplot(df: pd.DataFrame, fig, ax: plt.axis, vmin: float = None, vmax: float = None, show_colormap: bool = True,
              xlabel: str = None, ylabel: str = None, zlabel: str = None, title: str = None, cmap: str = "RdYlBu_r",
              cb_digits_after_comma: int = 1, show_grid: bool = False, cb_extend: str = 'both'):
    # Heatmap
    hm = dv.heatmapxyz(
        ax=ax,
        title=title,
        x=df.iloc[:, 0],
        y=df.iloc[:, 1],
        z=df.iloc[:, 2],
        cb_digits_after_comma=cb_digits_after_comma,
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
        # show_values_n_dec_places=1,
        # show_values=True,
        # show_values_fontsize=4,
        figdpi=300,
        color_bad='white',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_colormap=show_colormap,
        show_grid=show_grid,
        cb_extend=cb_extend
    )
    hm.plot()
    # hm.export_borderless_heatmap(
    #     name="TEST",
    #     outpath=r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\ms_co2_penalty\data\outputs\borderless_heatmaps")
    # ax = hm.get_ax()

    # ax.set_xlabel('Air temperature (z-score)')
    # ax.set_ylabel("Vapor pressure deficit (z-score)")
    if title:
        ax.set_title(title, fontsize=18, pad=10, y=1.02)

    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.tick_params(axis='both', which='major', width=1, length=5)
    ax.tick_params(axis='both', which='minor', width=1, length=2)
    # ax.axvline(x=0, color='black', linestyle='-', lw=99)

    return hm.p  # Return the pcolormesh object
