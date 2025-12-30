import diive as dv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from scipy.stats import gaussian_kde


def plot_scenario_panel(ax, df, feature_col, color, columns, n_scenarios, y_limits, is_top_row, group_name,
                        scenario_labels, show_x=False, show_y=False, is_main=False, is_first=False):
    pivot = df.pivot(index='SITE', columns='SCENARIO', values=feature_col).reindex(columns=columns)

    if pivot.dropna(how='all').empty:
        ax.set_visible(False)
        return

    # Stats: collect in table
    stats_list = []
    for col in [1, 4, 5]:
        # Drop NaNs for the specific scenario
        data = pivot[col].dropna()
        if not data.empty:
            stats_list.append({
                'IGBP': group_name,
                'Feature': feature_col,
                'n_sites': len(data),
                'Scenario': col,
                'Median': data.median(),
                'Min': data.min(),
                'Max': data.max(),
                'Sites < 0': (data < 0).sum(),
                '% < 0': (data < 0).mean() * 100
            })
    # Display as table
    featurestats_df = pd.DataFrame(stats_list)
    # print(stats_df.to_string(index=False))

    medians = pivot.median(axis=0)
    q1 = pivot.quantile(0.25, axis=0)
    q3 = pivot.quantile(0.75, axis=0)
    x_coords = np.arange(n_scenarios)

    # Ghost lines (faint)
    alpha_ghost = 0.1 if is_main else 0.15
    lw_ghost = 0.5
    ax.plot(x_coords, pivot.T.values, color='gray', alpha=alpha_ghost, linewidth=lw_ghost, zorder=1)

    # IQR ribbon
    ax.fill_between(x_coords, q1, q3, color=color, alpha=0.15, linewidth=0, zorder=2)

    # Sina / jitter points
    # Controlled jitter that respects density but stays tight
    for x_i, scen in enumerate(columns):
        if scen not in pivot:
            continue
        data = pivot[scen].dropna()
        if len(data) < 2:
            ax.scatter([x_i] * len(data), data, color=color, s=2, alpha=0.5, zorder=3)
            continue

        kde = gaussian_kde(data)
        density = kde(data)
        # Normalize width for jitter
        width_factor = 0.15
        width = (density / density.max()) * width_factor
        rng = np.random.RandomState(42 + x_i)
        jitter = rng.uniform(-1, 1, size=len(data)) * width

        # Plot points
        s_sina = 4 if is_main else 4
        alpha_sina = 0.4 if is_main else 0.5
        ax.scatter(x_i + jitter, data, color=color, s=s_sina, alpha=alpha_sina, linewidth=0, zorder=3)

        n_sites = len(data)

        # Sample size annotation, show in first row only
        if is_top_row:
            ax.text(x_coords[x_i], 0.9, f'n={n_sites}',
                    fontsize=7, color='#555555', ha='center', va='center')

        # Percentage of sites below zero (i.e., negatively affected)
        n_sites_below_zero = data[data < 0].count()
        perc_n_sites_below_zero = n_sites_below_zero / n_sites * 100

        # Decide text label
        text = f'{perc_n_sites_below_zero:.0f}%'
        # text = f'{perc_n_sites_below_zero:.0f}% negative' if is_first else f'{perc_n_sites_below_zero:.0f}%'

        # Implement percentage pill background
        # We use white text for high-impact visibility if the background is dark,
        # or keep the line color for the text and use a faint version for the pill.
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
        # Percentage of sites below zero (i.e., negatively affected)
        # n_sites_below_zero = data[data < 0].count()
        # perc_n_sites_below_zero = n_sites_below_zero / n_sites * 100
        # text = f'{perc_n_sites_below_zero:.0f}% negative' if is_first else f'{perc_n_sites_below_zero:.0f}%'
        # ax.text(x_coords[x_i], -1.3, text,
        #         fontsize=7, color=color, ha='center', va='center')
        # # ax.text(x_coords[x_i], -1.2, f'{n_sites_below_zero} ({perc_n_sites_below_zero:.0f}%)',
        # #         fontsize=6, color='#555555', ha='center', va='center')

    # Median trend line and nodes
    lw_trend = 2.0
    s_node = 25
    ax.plot(x_coords, medians, color=color, linewidth=lw_trend, alpha=1.0, zorder=5)
    ax.scatter(x_coords, medians, facecolor=color, edgecolor='white', linewidth=1.0, s=s_node, zorder=6)

    # Formatting
    ax.set_ylim(y_limits)
    # 1. Shade the Negative Region (add this before the zero line)
    # Use a very light gray to indicate the "constraint zone"
    ax.axhspan(y_limits[0], 0, facecolor='#f0f0f0', alpha=0.6, zorder=0)
    # 2. Zero line
    ax.axhline(0, color='black', linestyle='--', linewidth=0.6, alpha=0.6, zorder=0)
    # # Formatting
    # ax.set_ylim(y_limits)
    # # Zero line
    # ax.axhline(0, color='black', linestyle='--', linewidth=0.6, alpha=0.5, zorder=0)

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

    # # Highlight "All Sites" background
    # if is_main:
    #     ax.patch.set_facecolor('#f7f7f7')
    #     ax.patch.set_alpha(0.5)
    # else:
    #     ax.patch.set_alpha(0.0)

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
