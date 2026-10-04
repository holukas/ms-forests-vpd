import diive as dv
import matplotlib as mpl
import matplotlib.colors as mcolors
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


def heatmap_style(**overrides) -> dv.plotting.FormatStyle:
    """
    diive's default heatmap style, with optional per-plot overrides.

    Passing a `format_style` to a diive heatmap replaces its default style
    entirely, so the defaults are restated here and `overrides` are merged on top.
    """
    style = dv.plotting.FormatStyle(
        show_grid=False, show_legend=False, show_zeroline=False,
        chrome_color='black', spine_linewidth=2,
        ticks_direction='out', ticks_length=4, ticks_width=2)
    return style.merged(**overrides) if overrides else style


def create_colormap(fig, ax, cmap, label, absmax, labelsize, step=None):
    norm = mpl.colors.Normalize(vmin=-absmax, vmax=absmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=ax, extend='both')
    # Tick spacing follows the range, so a colorbar for a small effect still carries
    # ticks. A fixed 0.2 left the soil water effect, about 0.1 sigma, with a bare
    # zero. The factor 1.5 was set from three cases: 0.66 keeps the
    # main-analysis 0.2, and 0.31 and 0.10 get a single tick each side, at 0.2 and 0.05.
    # A figure with several colorbars can pass one step for all of them.
    if step is None:
        step = next(s for s in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01) if absmax / s >= 1.5)
    cb.ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
    cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: cb_formatter(x, pos, step)))
    cb.set_label(label, size=labelsize, labelpad=20)
    cb.ax.tick_params(labelsize=labelsize)


def cb_formatter(x, pos, step: float = 0.2):
    """Tick label: 0 as '0', other values with as many decimals as the tick step needs."""
    if np.isclose(x, 0, atol=1e-5):
        return "0"
    decimals = max(1, -int(np.floor(np.log10(step))))
    return f"{x:.{decimals}f}"


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
    ymin, ymax = (0, 0.25) if direction == 'up' else (-0.25, 0)
    im = ax.imshow(gradient, interpolation='bicubic',
                   extent=[-1.9, -0.5, ymin, ymax],
                   cmap=plt.cm.colors.LinearSegmentedColormap.from_list('custom', [color_main, '#ffffff']),
                   aspect='auto', alpha=0.6, zorder=3)
    im.set_clip_path(patch)


def get_panel_limits(df: pd.DataFrame):
    max_vals = []
    min_vals = []

    # Process each bar (grouped by IGBP and Scenario)
    grouped = df.groupby(['igbp', 'stage'])

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


# What each stage holds, as printed above the bars of the stage figures. One entry per
# stage sequence of src/stages.py. "non-extreme" marks the driver that is only kept off
# its extreme cut-off until the last stage: VPD in the main order ('published'), soil
# water in the mirrored one.
STAGE_CONDITIONS = {
    'published': {
        'VPD_ZSCORE':
            ['moderate', 'non-extreme', 'non-extreme', 'non-extreme', 'non-extreme', 'non-extreme',
             'non-extreme', '↑↑↑ extreme'],
        'TA_ZSCORE':
            ['moderate', '↑ high', '↑ high', '↑↑ very high', '↑↑ very high', '↑↑↑ extreme',
             '↑↑↑ extreme', '↑↑↑ extreme'],
        'SWC_ZSCORE':
            ['moderate', 'moderate', '↓ low', '↓ low', '↓↓ very low', '↓↓ very low',
             '↓↓↓ extreme', '↓↓↓ extreme'],
        'SWIN_ZSCORE':
            ['all', 'all', 'all', 'all', 'all', 'all', 'all', 'all'],
    },
    'mirrored': {
        'VPD_ZSCORE':
            ['moderate', 'moderate', '↑ high', '↑ high', '↑↑ very high', '↑↑ very high',
             '↑↑↑ extreme', '↑↑↑ extreme'],
        'TA_ZSCORE':
            ['moderate', '↑ high', '↑ high', '↑↑ very high', '↑↑ very high', '↑↑↑ extreme',
             '↑↑↑ extreme', '↑↑↑ extreme'],
        'SWC_ZSCORE':
            ['moderate', 'non-extreme', 'non-extreme', 'non-extreme', 'non-extreme', 'non-extreme',
             'non-extreme', '↓↓↓ extreme'],
        'SWIN_ZSCORE':
            ['all', 'all', 'all', 'all', 'all', 'all', 'all', 'all'],
    },
}


def draw_panel(ax, df, title, fixed_ylim, show_stage_labels, vars, palette, stage_labels,
               stage_ids, shap_suffix_avg, fontsize, is_small=False, stage_conditions=None):
    """Draw one stage bar panel.

    stage_conditions: labels printed above the bars, one list per driver (see
    STAGE_CONDITIONS). Defaults to STAGE_CONDITIONS['published'], the main sequence."""
    x_centers = [x for x in range(0, len(stage_ids))]
    bar_width = 0.55 if not is_small else 0.4
    x_centers_shifted_left = np.array(x_centers) - bar_width / 3
    x_centers_shifted_right = np.array(x_centers) + bar_width / 2.5
    alpha_ribbon = 0.35

    fs_val = fontsize if not is_small else 10
    fs_label = fontsize * 1.2 if not is_small else fontsize * 1.2
    fs_tick = fontsize if not is_small else fontsize

    # Storage for ribbon coordinates and net lines
    node_pos = [{} for _ in range(len(stage_ids))]
    net_vals = []
    net_errs = []
    net_counts = []

    # Iterate through scenarios to draw bars and collect Net data
    for i, stage_id in enumerate(stage_ids):
        cx = x_centers[i]

        # Filter df for this scenario
        df_s = df[df['stage'] == stage_id]
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

            if (abs(val) >= 0.02) and not is_small:
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

            if (abs(val) >= 0.02) and not is_small:
                div = 3 if abs(val) >= 0.4 else 2  # Displays text below middle of bar, for looong bars
                ax.text(x_centers_shifted_right[i], bottom + abs(val) / div, f"{val:.2f}", ha='right', va='center',
                        fontsize=fs_val - 1, color='white', fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=1.2, foreground=palette[var])], zorder=20)

            node_pos[i][var] = (bottom, top)
            current_y += val

        # Scenario Labels, N-counts, and Condition Rectangles (ABOVE the bars)
        if show_stage_labels:
            scenario_symbols = stage_conditions or STAGE_CONDITIONS['published']

            y_range = fixed_ylim[1] - fixed_ylim[0]
            box_height = y_range * 0.035  # Slightly thinner to fit everything
            box_width = bar_width * 0.8

            # 1. Scenario Name (stage_labels, e.g. "Stage 1")
            y_scen_label = fixed_ylim[1] - (y_range * 0.01)
            ax.text(cx, y_scen_label, stage_labels[i],
                    ha='center', va='top', fontsize=fs_tick, fontweight='bold')

            # 2. Site count (n=...)
            y_n_label = y_scen_label - (y_range * 0.03)
            current_n = net_counts[-1] if len(net_counts) > 0 else 0
            ax.text(cx, y_n_label, f"n={current_n}",
                    ha='center', va='top', fontsize=fs_tick, color='black')

            # 3. Condition Boxes (Start drawing below the 'n=...' text)
            # Use exactly 0.04 here for the top of the boxes
            top_y_boxes = y_n_label - (y_range * 0.04)

            for var_idx, var in enumerate(vars):
                # Stack downwards
                box_y = top_y_boxes - (var_idx * box_height) - box_height

                color = palette[var]
                symbol = scenario_symbols[var][i]

                # Draw Rectangle
                rect = plt.Rectangle(
                    (cx - (box_width / 2), box_y), box_width, box_height,
                    facecolor=color, alpha=0, edgecolor=color,
                    lw=1, zorder=99, clip_on=False)
                ax.add_patch(rect)

                # Add Symbol Text
                if symbol:
                    fs = 12
                    ax.text(cx, box_y + (box_height / 2), symbol,
                            ha='center', va='center', fontsize=fs,
                            fontweight='bold', color=color, zorder=100)

        # Site counts for small panels
        if is_small:
            y_range_small = fixed_ylim[1] - fixed_ylim[0]
            y_scen_label_small = fixed_ylim[1] - (y_range_small * 0.01)
            y_n_label_small = y_scen_label_small - (y_range_small * 0.03)
            current_n_small = net_counts[-1] if len(net_counts) > 0 else 0
            txt = f"n={current_n_small}" if i == 0 else f"{current_n_small}"
            ax.text(cx, y_n_label_small, f"{txt}",
                    ha='center', va='top', fontsize=fs_tick, color='black')

    # Add row labels to the left of the condition boxes
    if show_stage_labels:
        y_range = fixed_ylim[1] - fixed_ylim[0]
        box_height = y_range * 0.035

        # Recalculate same starting position
        y_scen_label = fixed_ylim[1] - (y_range * 0.01)
        # Note: Changed from 0.06 to 0.03 to match the loop above
        y_n_label = y_scen_label - (y_range * 0.03)

        # Note: Changed from 0.02 to 0.04 to match the loop above EXACTLY
        top_y_boxes = y_n_label - (y_range * 0.04)

        var_display_names = {
            'VPD_ZSCORE': 'VPD conditions   ',
            'TA_ZSCORE': 'TA conditions   ',
            'SWC_ZSCORE': 'SM conditions   ',
            'SWIN_ZSCORE': 'SW conditions   '
            # 'VPD_ZSCORE': 'Vapor pressure deficit   ',
            # 'TA_ZSCORE': 'Air temperature   ',
            # 'SWC_ZSCORE': 'Soil moisture   ',
            # 'SWIN_ZSCORE': 'Radiation   '
        }

        for var_idx, var in enumerate(vars):
            box_y = top_y_boxes - (var_idx * box_height) - box_height
            label_text = var_display_names.get(var, var)
            ax.text(x_centers[0] - (bar_width * 0.5) - 0.1, box_y + (box_height / 2),
                    label_text, ha='right', va='center', fontsize=12,
                    fontweight='bold', color=palette[var], zorder=100)

    # Draw ribbons
    for i in range(len(stage_ids) - 1):
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
    ms = 8 if is_small else 11
    ax.errorbar(x_centers_shifted_left, net_vals, yerr=net_errs, fmt='D', color='white',
                ecolor='black', elinewidth=elinewidth, capsize=4, zorder=23,
                ms=ms, mec='black', mew=mew)

    # Net Labels (text boxes)
    counter = 0
    for x, y in zip(x_centers_shifted_left, net_vals):
        counter += 1
        offset = 0
        bbox = dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6)
        _x = x - 0.1 if not is_small else x - 0.2
        _y = y + offset if not is_small else y  # Show centered in subplots
        if not is_small or counter in [1, 8]:
            ax.text(_x, _y, f"{y:+.2f}", fontsize=fs_val, fontweight='bold',
                    ha='right', va='center', bbox=bbox, zorder=25)

    # Styling
    ax.axhline(0, color='black', linewidth=1, linestyle='--', zorder=25)
    ax.set_title(title, fontsize=fs_label, fontweight='bold', loc='left', pad=10)
    ax.set_ylim(fixed_ylim)

    if not is_small:
        ax.set_xlim(x_centers[0] - 2, x_centers[-1] + 0.3)
    else:
        ax.set_xlim(x_centers[0] - 0.4, x_centers[-1] + 0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def show_shap_thresholds(ax, x_fit, y_fit, max_ix, min_ix, threshold_main, show_annotate, show_annotate_short,
                         fontsize, colors_symbols, label_pos: str = "Stimulation",
                         label_neg: str = "Suppression", peak: tuple = None):
    """Draw the zero crossing, the extremes and the two zone labels.

    label_pos and label_neg name the zones above and below zero. The defaults fit a
    carbon uptake flux (positive SHAP means more uptake). For RECO and ET, pass
    neutral labels such as "Increase" and "Decrease".

    peak is an optional (x, y) of the maximum, e.g. from fit.peak_of_polynomial(). Without
    it the maximum is read from the grid at max_ix, which rounds to the grid spacing.
    """
    x_peak, y_peak = peak if peak is not None else (x_fit[max_ix], y_fit[max_ix])
    color_limzone = '#d6604d'
    color_facilzone = '#4393c3'

    # threshold_main = x_fit[threshold_main_idx]

    # Get axis limits
    y_top_axis = ax.get_ylim()[1]
    y_bottom_axis = ax.get_ylim()[0]

    # -------------------------------------
    # STIMULATION ZONE (left of threshold)
    # -------------------------------------
    mask_facil = x_fit <= threshold_main
    x_facil = x_fit[mask_facil]

    if len(x_facil) > 1:
        # Define rectangle vertices (upwards from 0 to top)
        verts_facil = [
            (x_facil[0], 0),  # Start at left (y=0)
            (x_facil[-1], 0),  # End at threshold (y=0)
            (x_facil[-1], y_top_axis),  # Up to axis top
            (x_facil[0], y_top_axis),  # Back to start X at axis top
            (x_facil[0], 0)  # Close loop
        ]

        # Create path patch
        path_facil = mpath.Path(verts_facil)
        patch_facil = mpatches.PathPatch(path_facil, facecolor='none', edgecolor='none')
        ax.add_patch(patch_facil)

        # Gradient (blue)
        # imshow plots left-to-right (Index 0 -> Index 255)
        # So index 0 (left) = higher alpha, index 255 (right) = lower alpha
        c_blue_start = mcolors.to_rgba(color_facilzone, alpha=0.05)
        c_blue_end = mcolors.to_rgba(color_facilzone, alpha=0.1)
        colors_facil = [c_blue_start, c_blue_end]
        cmap_grad_facil = mcolors.LinearSegmentedColormap.from_list('fader_blue', colors_facil)

        # Draw the gradient image
        # Extent fills from 0 UP to y_top_axis
        im_facil = ax.imshow(
            np.linspace(0, 1, 256).reshape(1, 256),
            aspect='auto',
            cmap=cmap_grad_facil,
            extent=[x_facil.min(), x_facil.max(), 0, y_top_axis],
            zorder=0
        )
        im_facil.set_clip_path(patch_facil)

    # ------------------------------------
    # SUPPRESSION ZONE (right of threshold)
    # ------------------------------------
    y_bottom_axis = ax.get_ylim()[0]  # Absolute bottom of current plot axis
    mask_limit = x_fit >= threshold_main
    x_limit = x_fit[mask_limit]
    if len(x_limit) > 1:
        # Define rectangle vertices (clockwise or counter-clockwise)
        # top-left -> top-right -> bottom-right -> bottom-left -> close
        verts = [
            (x_limit[0], 0),  # start at threshold (y=0)
            (x_limit[-1], 0),  # end at max x (y=0)
            (x_limit[-1], y_bottom_axis),  # down to axis bottom
            (x_limit[0], y_bottom_axis),  # back to start x at axis bottom
            (x_limit[0], 0)  # close loop
        ]

        # Create path patch
        path = mpath.Path(verts)
        patch = mpatches.PathPatch(path, facecolor='none', edgecolor='none')
        ax.add_patch(patch)

        # Gradient (red)
        c_start = mcolors.to_rgba(color_limzone, alpha=0.1)
        c_end = mcolors.to_rgba(color_limzone, alpha=0.05)
        colors = [c_start, c_end]
        cmap_grad = mcolors.LinearSegmentedColormap.from_list('fader', colors)

        # Draw gradient image
        # Extent corresponds to [left, right, bottom, top]
        # Fill from y_bottom_axis up to 0
        im = ax.imshow(
            np.linspace(0, 1, 256).reshape(1, 256),
            aspect='auto',
            cmap=cmap_grad,
            extent=[x_limit.min(), x_limit.max(), y_bottom_axis, 0],
            zorder=0
        )
        im.set_clip_path(patch)

    ax.plot([threshold_main, threshold_main], [y_bottom_axis, 0],
            color=color_limzone, linestyle='-', linewidth=1, zorder=1)
    color = 'black'

    # mid_point_x = threshold_x + (x_fit.max() - threshold_x) / 1.05
    if show_annotate and not show_annotate_short:
        ax.text(x_fit[min_ix] - 0.15, -0.05, label_neg,
                color=color_limzone, alpha=1, ha='right', va='top',
                fontsize=fontsize * 1.2, style='italic', weight='bold')
        # ax.text(x_fit[min_ix] - 0.1, -0.1, "reduced uptake\nincreased release",
        #         color=color_limzone, alpha=1, ha='right', va='top',
        #         fontsize=fontsize * 1.2, style='italic', weight='normal')
        ax.text(threshold_main - 0.15, y_top_axis * 0.92, label_pos,
                color=color_facilzone, alpha=1, ha='right', va='top',
                fontsize=fontsize * 1.2, weight='bold', style='italic', zorder=1)
        # ax.text(x_fit[idx] - 0.1, y_top_axis * 0.85, "increased uptake\nreduced release",
        #         color=color_facilzone, alpha=1, ha='right', va='top',
        #         fontsize=fontsize * 1.2, weight='normal', style='italic', zorder=1)

    # -----------------------
    # ANNOTATIONS AND MARKERS
    # -----------------------
    _params = dict(color='black', linestyle=':', linewidth=1, zorder=100)
    _params2 = dict(linewidth=2, zorder=100, s=200, alpha=1)
    _params3 = dict(color='black', fontsize=fontsize, linespacing=1.2)

    # Calculate y-offset for lines leading to annotations
    # Needs to be a bit larger for the smaller subplots
    # As relative_height goes down (0.2), the factor goes UP
    # As relative_height goes up (1.0), the factor goes DOWN
    bbox = ax.get_position()
    relative_height = bbox.height  # Height of ax as a fraction of figure (0.0 to 1.0)
    y_range = y_top_axis - y_bottom_axis
    scaling_factor = 0.02 + (0.04 / relative_height)
    scaling_factor = min(0.08, scaling_factor)  # Clamp so it does not get huge on small plots
    y_offset_topline = y_range * scaling_factor

    # Maximum facilitation
    ax.scatter(x_peak, y_peak, color='none', marker='^', edgecolor=colors_symbols[0], **_params2)

    # Threshold
    ax.scatter(threshold_main, 0, c="none", edgecolors=colors_symbols[1], **_params2)

    # Max. limitation
    ax.scatter(x_fit[min_ix], y_fit[min_ix], color='none', marker='v', edgecolor=colors_symbols[2], **_params2)

    if show_annotate:
        _fontsize = fontsize * 0.9 if show_annotate_short else fontsize

        if not show_annotate_short:
            ann_txt = f'Max. {label_pos.lower()}\nx={x_peak:.2f}'
            offx = 0.1
            offy = 0.06
        else:
            ann_txt = f'x={x_peak:.2f}'
            offx = 0.1
            offy = 0
        ax.annotate(ann_txt,
                    xy=(x_peak, y_peak),
                    xytext=(x_peak + offx, -0.15 + offy),  # Adjust text position as needed
                    # xy=(x_fit[max_ix], y_fit[max_ix]),
                    # xytext=(x_fit[max_ix] + offx, y_fit[max_ix] * -1),  # Adjust text position as needed
                    arrowprops=dict(arrowstyle="->", color=color, lw=2, shrinkB=10),
                    fontsize=_fontsize, color=color, ha='center', va='center', zorder=100)

        if not show_annotate_short:
            ann_txt = f'Threshold\nx={threshold_main:.2f}'
            offx = 0.8
            offy = 0.3
        else:
            ann_txt = f'x={threshold_main:.2f}'
            offx = 0.7
            offy = 0.6
        ax.annotate(ann_txt,
                    xy=(threshold_main, 0),
                    xytext=(threshold_main - offx, 0 - offy),  # Adjust text position as needed
                    arrowprops=dict(arrowstyle="->", color=color, lw=2, shrinkB=10),
                    fontsize=_fontsize, color=color, ha='center', va='center', zorder=100)

        if not show_annotate_short:
            ann_txt = f'Max. {label_neg.lower()}\nx={x_fit[min_ix]:.2f}'
            offx = 1
            offy = 0.05
        else:
            ann_txt = f'x={x_fit[min_ix]:.2f}'
            offx = 1.5
            offy = 0
        ax.annotate(ann_txt,
                    xy=(x_fit[min_ix], y_fit[min_ix]),
                    xytext=(x_fit[min_ix] - offx, y_bottom_axis * 0.9),  # Adjust text position as needed
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2, shrinkB=10),
                    fontsize=_fontsize, color=color, ha='center', va='center', zorder=100)


def format(ax, fontsize, showxticklabels, showyticklabels, xtickdigits, ytickdigits,
           showbottomspine, showleftspine, showymajorticks):
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1) if showbottomspine else ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(1) if showleftspine else ax.spines['left'].set_visible(False)

    # Ticks
    ax.tick_params(axis='y', which='major', width=1, length=5, labelsize=fontsize, left=showymajorticks)
    ax.tick_params(axis='x', which='major', width=1, length=5, labelsize=fontsize)

    # Ticklabels
    if showxticklabels:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
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
            show_annotate, fontsize, linewidth, color: str = '#004e98'):
    # Plot fitted polynomial curve
    # Plot prediction interval
    ax.plot(x_fit, y_fit, color=color, linewidth=linewidth, zorder=99)
    # label=rf'$y = {poly_coeffs[0]:.4f}x^4 - {poly_coeffs[1]:.4f}x^3 + {poly_coeffs[2]:.4f}x^2 + {poly_coeffs[3]:.4f}x - {poly_coeffs[4]:.4f}$'
    fillbetweenplot = ax.fill_between(x_fit, pi_lower, pi_upper, color=color, alpha=0.25,
                                      label='95% pred. interval', zorder=1)

    if show_annotate:
        # Add an arrow to the fitted line
        # Find a point on the line to place the arrow.
        arrow_x = 1.5
        arrow_y = poly_func(arrow_x)
        # Find a point slightly to the left to define the arrow direction
        tail_x = arrow_x - 0.1
        tail_y = poly_func(tail_x)
        # Calculate the angle of the line at this point to get the correct arrow orientation
        angle = np.arctan2(arrow_y - tail_y, arrow_x - tail_x) * 180 / np.pi
        supscript = r'$^{th}$'
        ax.annotate(f'Poly. fit (4{supscript} order)',
                    xy=(arrow_x, arrow_y),
                    xytext=(arrow_x - 1.1, arrow_y - 0.25),  # Adjust text position as needed
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2),
                    fontsize=fontsize, color=color, ha='left', va='center', zorder=100)
    return fillbetweenplot


def layout_1row_5panels(figsize, add_colorbar_ax: bool = False):
    """Create a figure with 5 panels in one row and, optionally, a narrow colorbar axis."""
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


def plot_markers(ax, df, xvals, yvals, zvals, flux_txt, ax_labels_fontsize, area_size, annotate=False,
                 show_only_max_marker: bool = False):
    """Find the max (and min) regions with `findpoi`, mark them, and optionally add arrows."""
    piv = df.pivot(index=xvals, columns=yvals, values=zvals)

    _showaggs = ['max', 'min'] if not show_only_max_marker else ['max']
    locs = {m: findpoi(df=piv, k=area_size, agg='mean', what=m)[0] for m in _showaggs}

    for m_type, (lx, ly) in locs.items():
        x, y = lx + 0.05, ly + 0.05
        marker = '+' if m_type == 'max' else '_'
        # clip_on=False: a marker at the edge of the data is drawn whole rather than
        # cut by the axes box.
        ax.scatter(x, y, c='k', marker=marker, lw=3, s=650, zorder=100, alpha=0.5, clip_on=False)
        ax.scatter(x, y, facecolors='none', edgecolors='k', marker='o', lw=3, s=650, zorder=100, alpha=0.5,
                   clip_on=False)

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


def plot_stage_panel(ax, df, feature_col, color, columns, n_scenarios, y_limits, is_top_row, group_name,
                     stage_labels, show_x=False, show_y=False, is_main=False, is_first=False):
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
    for col in [1, 2, 3, 4, 5, 6, 7, 8]:

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
    alpha_ghost = 0.2 if is_main else 0.25
    lw_ghost = 0.5
    ax.plot(x_coords, pivot.T.values, color='gray', alpha=alpha_ghost, linewidth=lw_ghost, zorder=1)

    # todo IQR ribbon
    # ax.fill_between(x_coords, p25, p75, color=color, alpha=0.25, linewidth=0, zorder=2)

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
        s_sina = 6 if is_main else 5
        alpha_sina = 0.4 if is_main else 0.5
        ax.scatter(x_i + jitter, data_vec, color=color, s=s_sina, alpha=alpha_sina, linewidth=0, zorder=3)

        n_sites = len(data_vec)

        # Sample size annotation, show in first row only
        if is_top_row:
            ax.text(x_coords[x_i], 0.9, f'n={n_sites}' if is_main else f"{n_sites}",
                    fontsize=7, color='#555555', ha='center', va='center')

        # Percentage of sites below zero (i.e., negatively affected)
        n_sites_below_zero = data_vec[data_vec < 0].count()
        perc_n_sites_below_zero = n_sites_below_zero / n_sites * 100

        # Also store in stats df
        featurestats_df.loc[featurestats_df['Scenario'] == scen, 'PERC_N_SITES_BELOW_ZERO'] = perc_n_sites_below_zero

        # Decide text label
        text = f'{perc_n_sites_below_zero:.0f}%' if is_main else f'{perc_n_sites_below_zero:.0f}'

        # Implement percentage pill background
        alpha = 0.9 if perc_n_sites_below_zero > 66.6 else 0.5
        ax.text(x_coords[x_i], -1.38, text,
                fontsize=9 if is_main else 8,
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
    ax.grid(False)

    # Format x-axis
    ax.set_xlim(-0.5, (n_scenarios - 1) + 0.5)
    ax.set_xticks(x_coords)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    # Tick Styling
    if show_x:
        # ax.set_xticklabels(scenario_labels, rotation=45, ha='right', color='black')
        ax.set_xticklabels(stage_labels, rotation=45, ha='right', color='black', rotation_mode='anchor')
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
    # diive >= 0.91 splits this in two: data and labels go to the constructor,
    # everything about the rendering goes to plot(). Chrome that used to be a
    # plain keyword (title, show_grid) now travels in a FormatStyle.
    hm = dv.plotting.HeatmapXYZ(
        x=df.iloc[:, 0],
        y=df.iloc[:, 1],
        z=df.iloc[:, 2],
        xlabel=xlabel,
        ylabel=ylabel,
        zlabel=zlabel,
    )
    hm.plot(
        ax=ax,
        format_style=heatmap_style(title=title, show_grid=show_grid),
        cb_digits_after_comma=cb_digits_after_comma,
        # show_values_n_dec_places=1,
        # show_values=True,
        # show_values_fontsize=4,
        figdpi=300,
        color_bad='white',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_colormap=show_colormap,
        cb_extend=cb_extend
    )

    return hm.p  # Return the pcolormesh object
