import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from scipy.stats import gaussian_kde

from src.common import findpoi


def add_gradient_arrow(ax, vertices, color_main, direction='up'):
    # Create Path
    path = mpath.Path(vertices)
    patch = mpatches.PathPatch(path, facecolor='none', edgecolor='none')
    ax.add_patch(patch)

    # Define gradient (top to bottom)
    # Custom colormap from chosen color to a lighter/faded version
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    if direction == 'down':
        gradient = np.flipud(gradient)  # Flip for the down arrow

    # Display and clip
    # Extent should cover the bounding box of the arrow
    ymin, ymax = (0, 0.2) if direction == 'up' else (-0.2, 0)
    im = ax.imshow(gradient, interpolation='bicubic',
                   extent=[-1, -0.3, ymin, ymax],
                   cmap=plt.cm.colors.LinearSegmentedColormap.from_list('custom', [color_main, '#ffffff']),
                   aspect='auto', alpha=0.6, zorder=3)
    im.set_clip_path(patch)


def get_panel_limits(df: pd.DataFrame):
    max_vals = []
    min_vals = []

    # Process each bar (grouped by IGBP and Scenario)
    grouped = df.groupby(['igbp', 'scenario'])

    for name, group in grouped:
        # Separate individual drivers from the Net effect
        components = group[group['Variable'] != 'NET_SHAPVALS']
        net_row = group[group['Variable'] == 'NET_SHAPVALS']

        # Check stack heights (sum of means)
        pos_sum = components.loc[components['mean'] > 0, 'mean'].sum()
        neg_sum = components.loc[components['mean'] < 0, 'mean'].sum()

        max_vals.append(pos_sum)
        min_vals.append(neg_sum)

        # Check net point with error bars

        if not net_row.empty:
            net_val = net_row['mean'].values[0]
            net_err = net_row['sem'].values[0]

            max_vals.append(net_val + net_err)
            min_vals.append(net_val - net_err)

    # Return global min and max
    if not max_vals:
        return 0.0, 0.0

    return min(min_vals), max(max_vals)


def sigmoid(x, x_start, x_end, y_start, y_end):
    x_norm = (x - x_start) / (x_end - x_start)
    s = 0.5 * (1 + np.tanh(6 * (x_norm - 0.5)))
    return y_start + s * (y_end - y_start)


def draw_panel(ax, df, title, fixed_ylim, show_scenario_labels, vars, palette, scenario_labels,
               shap_suffix_avg, is_small=False):
    # Constants
    SCENARIO_IDS = [1, 4, 5]  # The scenario IDs in dataframe column 'scenario'

    x_centers = [0, 1, 2]
    bar_width = 0.4 if not is_small else 0.4
    x_centers_shifted_left = np.array(x_centers) - bar_width / 3
    x_centers_shifted_right = np.array(x_centers) + bar_width / 2.5
    alpha_ribbon = 0.25

    fs_val = 12 if not is_small else 12
    fs_label = 12 if not is_small else 12
    fs_tick = 12 if not is_small else 12

    # Storage for ribbon coordinates and net lines
    node_pos = [{} for _ in range(len(SCENARIO_IDS))]
    net_vals = []
    net_errs = []
    net_counts = []

    # Iterate through scenarios to draw bars and collect Net data
    for i, scen_id in enumerate(SCENARIO_IDS):
        cx = x_centers[i]

        # Filter df for this scenario
        df_s = df[df['scenario'] == scen_id]
        if df_s.empty:
            # Handle empty data (add placeholders to keep alignment)
            net_vals.append(np.nan)
            net_errs.append(np.nan)
            net_counts.append(0)
            continue

        # Extract net data
        net_row = df_s[df_s['Variable'] == 'NET_SHAPVALS']
        if not net_row.empty:
            # Use .values[0] to safely get the scalar
            net_vals.append(net_row['mean'].values[0])
            net_errs.append(net_row['sem'].values[0])
            net_counts.append(int(net_row['n_sites'].values[0]))
        else:
            raise ValueError("Expected net row to exist in dataframe.")
            # net_vals.append(0)
            # net_errs.append(0)
            # net_counts.append(0)

        # Draw bars: positive stack
        current_y = 0.0
        for var in vars:
            # Find row for this variable
            mean_col = var + shap_suffix_avg
            row = df_s[df_s['Variable'] == mean_col]
            if row.empty:
                continue

            val = row['mean'].values[0]
            if val < 0:
                continue  # Skip negatives in this pass

            top = current_y + val
            bottom = current_y

            ax.bar(cx, val, width=bar_width, bottom=bottom, color=palette[var],
                   edgecolor='white', linewidth=0.5, zorder=10)

            if (abs(val) > 0.01) and not is_small:
                ax.text(x_centers_shifted_right[i], bottom + val / 2, f"+{val:.2f}", ha='right', va='center',
                        fontsize=fs_val - 1, color='white', fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=1.2, foreground=palette[var])], zorder=20)

            node_pos[i][var] = (bottom, top)
            current_y += val

        # Draw bars: negative stack
        current_y = 0.0
        for var in vars:
            mean_col = var + shap_suffix_avg
            row = df_s[df_s['Variable'] == mean_col]
            if row.empty:
                continue

            val = row['mean'].values[0]
            if val >= 0:
                continue  # Skip positives in this pass

            # Stack downwards
            top = current_y
            bottom = current_y + val

            ax.bar(cx, abs(val), width=bar_width, bottom=bottom, color=palette[var],
                   edgecolor='white', linewidth=0.5, zorder=10)

            if (abs(val) > 0.01) and not is_small:
                ax.text(x_centers_shifted_right[i], bottom + abs(val) / 2, f"{val:.2f}", ha='right', va='center',
                        fontsize=fs_val - 1, color='white', fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=1.2, foreground=palette[var])], zorder=20)

            node_pos[i][var] = (bottom, top)
            current_y += val

        # Scenario Labels
        if show_scenario_labels:
            ax.text(cx, fixed_ylim[0] + (abs(fixed_ylim[0]) * 0.07), scenario_labels[i],
                    ha='center', va='top', fontsize=fs_tick, fontweight='bold')

    # Draw ribbons
    for i in range(len(SCENARIO_IDS) - 1):
        x_start, x_end = x_centers[i] + bar_width / 2, x_centers[i + 1] - bar_width / 2
        x_curve = np.linspace(x_start, x_end, 100)

        for var in vars:
            # Check if var exists in both steps
            if var not in node_pos[i] or var not in node_pos[i + 1]:
                continue

            s_bot, s_top = node_pos[i][var]
            e_bot, e_top = node_pos[i + 1][var]

            # Skip if negligible height
            if abs(s_top - s_bot) < 0.005 and abs(e_top - e_bot) < 0.005:
                continue

            y_top = sigmoid(x_curve, x_start, x_end, s_top, e_top)
            y_bot = sigmoid(x_curve, x_start, x_end, s_bot, e_bot)
            ax.fill_between(x_curve, y_bot, y_top, color=palette[var],
                            alpha=alpha_ribbon, edgecolor='none', zorder=1)

    # Net values and error bars (SEM)

    # Connector line
    ax.plot(x_centers_shifted_left, net_vals, '-', color='#808080',
            linewidth=2, markersize=10, zorder=19)

    # Error bars (net effect)
    mew = 1.5 if is_small else 2
    elinewidth = 1.5 if is_small else 2
    ax.errorbar(x_centers_shifted_left, net_vals, yerr=net_errs, fmt='D', color='white',
                ecolor='black', elinewidth=elinewidth, capsize=4, zorder=23,
                ms=10, mec='black', mew=mew)

    # Net Labels (text boxes)
    for x, y in zip(x_centers_shifted_left, net_vals):
        offset = 0
        bbox = dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6)
        ax.text(x - 0.1, y + offset, f"{y:+.2f}", fontsize=fs_val, fontweight='bold',
                ha='right', va='center', bbox=bbox, zorder=25)

    # Counts (n=...)
    for x, y in zip(x_centers, net_counts):
        ypos = 0.42 if is_small else 0.37
        smaller = 1 if is_small else 0
        ax.text(x, ypos, f"n={y}", fontsize=fs_val - smaller, fontweight='normal',
                ha='center', va='center', zorder=25)

    # Styling
    ax.axhline(0, color='black', linewidth=1, linestyle='--', zorder=25)
    ax.set_title(title, fontsize=fs_label, fontweight='bold', loc='left', pad=10)
    ax.set_ylim(fixed_ylim)

    if not is_small:
        ax.set_xlim(-1, 2.3)
    else:
        ax.set_xlim(-0.3, 2.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def show_shap_thresholds(ax, x_fit, y_fit, max_ix, min_ix, idx, ydim_max, ydim_min, show_annotate, show_annotate_short,
                         fontsize, colors_symbols):
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
    ax.scatter(x_fit[max_ix], y_fit[max_ix], color='none', marker='^', edgecolor=colors_symbols[0], **_params2)
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
    ax.scatter(x_fit[idx], y_fit[idx], c="none", edgecolors=colors_symbols[1], **_params2)
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
    ax.scatter(x_fit[min_ix], y_fit[min_ix], color='none', marker='v', edgecolor=colors_symbols[2], **_params2)
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


def format(ax, fontsize, showxticklabels, showyticklabels, xtickdigits, ytickdigits,
           showbottomspine, showleftspine):
    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1) if showbottomspine else ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(1) if showleftspine else ax.spines['left'].set_visible(False)

    ax.tick_params(axis='y', which='major', left=False)

    # Set the tick width for both x and y axes
    ax.tick_params(axis='x', which='major', width=1, length=5, labelsize=fontsize)
    # ax.tick_params(axis='both', which='minor', width=1, length=2, labelsize=fontsize)

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


def add_fit(ax, x_fit, y_fit, pi_lower, pi_upper, poly_func, r_squared,
            show_annotate, fontsize, color: str = '#004e98'):
    # Plot fitted polynomial curve
    ax.plot(x_fit, y_fit, color=color, linewidth=3, zorder=99)
    # label=rf'$y = {poly_coeffs[0]:.4f}x^4 - {poly_coeffs[1]:.4f}x^3 + {poly_coeffs[2]:.4f}x^2 + {poly_coeffs[3]:.4f}x - {poly_coeffs[4]:.4f}$'

    # Plot prediction interval
    fillbetweenplot = ax.fill_between(x_fit, pi_lower, pi_upper, color=color, alpha=0.2,
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
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                    fontsize=fontsize, color=color, ha='left', va='center', zorder=100)
    return fillbetweenplot


def layout_1row_5panels(figsize, add_colorbar_ax: bool = False):
    """
    Creates a figure with 5 panels arranged in a single row.
    Optionally adds a 6th narrow column for a colorbar.
    """
    fig = plt.figure(figsize=figsize, dpi=150, facecolor="white")
    ncols = 6
    width_ratios = [1, 1, 1, 1, 1, 0.1]
    gs = gridspec.GridSpec(1, ncols, width_ratios=width_ratios)
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    if add_colorbar_ax:
        # Create the colorbar axis in the last column (index 5)
        cax = fig.add_subplot(gs[0, 5])
        return fig, gs, axes, cax
    else:
        return fig, gs, axes


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
    # Pivot for means (site x scenario)
    pivot = df.pivot(index='SITE', columns='SCENARIO', values=feature_col).reindex(columns=columns)

    # Pivot for SDs (site x scenario)
    sd_col = feature_col.replace('_SHAPVALS_OVR_AVG', '_SHAPVALS_OVR_SD')
    pivot_sd = df.pivot(index='SITE', columns='SCENARIO', values=sd_col).reindex(columns=columns)

    if pivot.dropna(how='all').empty:
        ax.set_visible(False)
        return pd.DataFrame()

    stats_list = []

    # Loop through scenarios to calculate stats
    for col in [1, 4, 5]:

        # Drop NaNs for the specific scenario
        data_vec = pivot[col].dropna()

        # Selects rows from data_sd where the index exists in data
        if col in pivot_sd:
            data_sd_vec = pivot_sd[col].loc[data_vec.index]
        else:
            raise ValueError(f'Scenario {col} not found in pivot_sd')

        if not data_vec.empty:
            # Calculation of total SD
            # Calculate total SD using the law of total variance (sqrt(mean of variances + variance of means)).
            # This approach treats sites as equally representative (macro-average), normalizing
            # differences in sample counts between sites.
            mean_of_variances = (data_sd_vec ** 2).mean()  # Average of within-site variances
            variance_of_means = data_vec.var(ddof=0)  # Variance of the site means
            total_sd = np.sqrt(mean_of_variances + variance_of_means)  # Global SD

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
        # ax.set_xticklabels(scenario_labels, rotation=45, ha='right', color='black')
        ax.set_xticklabels(scenario_labels, rotation=45, ha='right', color='black', rotation_mode='anchor')
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
