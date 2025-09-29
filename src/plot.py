import diive as dv
import matplotlib.pyplot as plt
import pandas as pd


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

