import time
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from diive.core.io.files import load_parquet
from diive.pkgs.analyses.decoupling import SortingBinsMethod
from scipy.stats import binned_statistic

df = pd.read_csv('../OUT/04_bins_zscores_allsites.csv')

igbps = list(set(df['IGBP'].tolist()))
print(igbps)

# Keep specific IGBP
# ['EBF', 'ENF', 'MF', 'DBF', 'DNF']
# locs = df['IGBP'] == 'ENF'
# df = df[locs].copy()




# Variables
neecol = "NEE_VUT_50"
tacol = "TA_F"
vpdcol = "VPD_F"

# Used vars
xvar = vpdcol
zvar = tacol
yvar = neecol

from_zvar = f"FROM_{zvar}"
to_zvar = f"TO_{zvar}"

from_xvar = f"FROM_{xvar}"
to_xvar = f"TO_{xvar}"
from_yvar = f"FROM_{yvar}"
to_yvar = f"TO_{yvar}"

from_xerror_pos = f"FROM_xerror_pos"
to_xerror_pos = f"TO_xerror_pos"
from_xerror_neg = f"FROM_xerror_neg"
to_xerror_neg = f"TO_xerror_neg"

from_yerror_pos = f"FROM_yerror_pos"
to_yerror_pos = f"TO_yerror_pos"
from_yerror_neg = f"FROM_yerror_neg"
to_yerror_neg = f"TO_yerror_neg"

# Sort by z medians (TA)
df = df.sort_values(by=to_zvar, inplace=False)

# Group into 100 groups of z, add to df
group, bins = pd.qcut(df[to_zvar], q=100, retbins=True, precision=9, duplicates='drop')
df['AGG_GROUP'] = group

# Init figure
fig = plt.figure(figsize=(9, 9))
gs = gridspec.GridSpec(1, 1)  # rows, cols
# gs.update(wspace=.2, hspace=1, left=.1, right=.9, top=.85, bottom=.1)
ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])
colors = plt.cm.coolwarm(np.linspace(0.1, 1, 100))  # Colors for 100 bins

counter = -1
grouped = df.groupby(by='AGG_GROUP', observed=True, as_index=True, sort=True, group_keys=True)
for g, g_df in grouped:
    counter += 1
    g_df = g_df.set_index('AGG_GROUP')
    # Number of different sites in this group
    n_sites = len(set(g_df['SITE'].tolist()))
    print(f"Plotting group {g} ({n_sites} unique sites)")

    # Get group medians
    from_x_median = g_df[from_xvar].median()
    to_x_median = g_df[to_xvar].median()
    from_y_median = g_df[from_yvar].median()
    to_y_median = g_df[to_yvar].median()

    # todo
    delta_x = to_x_median - from_x_median
    delta_y = to_y_median - from_y_median
    sensitivity = delta_y / delta_x
    print(delta_x, delta_y, sensitivity)

    # z (TA) is used as the grouping variable and for the colors
    from_z_medians = g_df[from_zvar].to_list()
    to_z_medians = g_df[to_zvar].to_list()
    all_z_medians = from_z_medians + to_z_medians
    z_median_across_all = np.mean(all_z_medians)

    # Lower error: get group SD
    from_x_min = g_df[from_xvar].std()
    to_x_min = g_df[to_xvar].std()
    from_y_min = g_df[from_yvar].std()
    to_y_min = g_df[to_yvar].std()

    # Get group maxima
    # from_x_max = abs(g_df[from_xvar].quantile(0.84))
    from_x_max = g_df[from_xvar].std()
    to_x_max = g_df[to_xvar].std()
    from_y_max = g_df[from_yvar].std()
    to_y_max = g_df[to_yvar].std()

    # Re-arrange for plotting
    x = [from_x_median, to_x_median]
    y = [from_y_median, to_y_median]
    xerror_pos = [from_x_max, to_x_max]
    xerror_neg = [from_x_min, to_x_min]
    yerror_pos = [from_y_max, to_y_max]
    yerror_neg = [from_y_min, to_y_min]



    ax1.plot(x, y,
             ls='-', lw=5, ms=14, label="X", color=colors[counter],
             marker='o', mec='k', mew=1, mfc=colors[counter], alpha=1, zorder=99)
    ax1.errorbar(x=x, y=y,
                 xerr=[xerror_neg, xerror_pos],
                 yerr=[yerror_neg, yerror_pos],
                 elinewidth=8, ecolor=colors[counter], alpha=.3, lw=0)
    ax1.plot(x, y,
             ls='-', lw=2, ms=0, label=None, color='black',
             mec='k', mew=1, alpha=1, zorder=99)

ax1.axhline(0, color="black")
ax1.axvline(0, color="black")
fig.suptitle("All z-scores", fontsize=16)
fig.tight_layout()
fig.show()



# for k, row in df.iterrows():
    # x = [row[from_xvar], row[to_xvar]]
    # y = [row[from_yvar], row[to_yvar]]
    # xerror_pos = [row[from_xerror_pos], row[to_xerror_pos]]
    # xerror_neg = [row[from_xerror_neg], row[to_xerror_neg]]
    # yerror_pos = [row[from_yerror_pos], row[to_yerror_pos]]
    # yerror_neg = [row[from_yerror_neg], row[to_yerror_neg]]

    # ax.plot(x, y,
    #         ls='-', lw=3, ms=14, label="X", color="black",
    #         marker='o', mec='k', mew=1, alpha=1, zorder=99)
    # ax.errorbar(x=x, y=y,
    #             xerr=[xerror_neg, xerror_pos],
    #             yerr=[yerror_neg, yerror_pos],
    #             elinewidth=8, ecolor="red", alpha=.1, lw=0)

# colors = plt.cm.coolwarm(np.linspace(0.1, 1, self.n_bins_z))
# for ix, m in enumerate(self.binaggs.keys()):
#     lw = 5 if emphasize_lines else 3
#     ax.plot(self.binaggs[m][self.xvar], self.binaggs[m][self.yvar],
#             ls='-', lw=lw, ms=14, label=m, color=colors[ix],
#             mec='k', mew=1, alpha=1, zorder=99, **kwargs)
#     ax.errorbar(x=self.binaggs[m][self.xvar],
#                 y=self.binaggs[m][self.yvar],
#                 xerr=[
#                     self.binaggs[m]['xerror_neg'],
#                     self.binaggs[m]['xerror_pos']
#                 ],
#                 yerr=[
#                     self.binaggs[m]['yerror_neg'],
#                     self.binaggs[m]['yerror_pos']
#                 ],
#                 elinewidth=8, ecolor=colors[ix], alpha=.3, lw=0)
#     if emphasize_lines:
#         ax.plot(self.binaggs[m][self.xvar], self.binaggs[m][self.yvar],
#                 ls='-', lw=2, ms=0, label=None, color='black',
#                 mec='k', mew=1, alpha=1, zorder=99)
#
# n_vals_yvar = self.df[self.yvar].count()
# n_vals_datapoint = n_vals_yvar / self.n_bins_z
# n_vals_datapoint = int(n_vals_datapoint / self.n_bins_x)
#
# txt_perc = f" {self.conversion} " if self.conversion else " "
#
# # Check number of available bins
# n_bins_zvar = len(self.binaggs)
# if n_bins_zvar != self.n_bins_z:
#     n_not_generated = self.n_bins_z - n_bins_zvar
# else:
#     n_not_generated = 0
#
# txt = (f"showing {self.agg} with 16-84 percentile range\n"
#        f"{n_vals_yvar}{txt_perc}values of {self.yvar}\n"
#        f"in {self.n_bins_x}{txt_perc}classes of {self.xvar},\n"
#        f"separate for {n_bins_zvar}{txt_perc}classes of {self.zvar}\n"
#        f"= {n_vals_datapoint} values per data point")
# # n_vals = self.df.groupby(self.var1_group_col).count().mean()[self.var1_col]
# # n_vals = int(n_vals / self.n_subbins_var2)
# ax.text(0.98, 0.02, txt,
#         size=theme.AX_LABELS_FONTSIZE, color='k', backgroundcolor='none', transform=ax.transAxes,
#         alpha=1, horizontalalignment='right', verticalalignment='bottom')
# default_format(ax=ax,
#                ax_xlabel_txt=f"{self.xvar}{txt_perc}",
#                ax_ylabel_txt=f"{self.yvar}{txt_perc}")
#
# textsize = theme.FONTSIZE_TXT_LEGEND_SMALLER_14
#
# if legend:
#     default_legend(ax=ax, ncol=n_col,
#                    title=f"{n_bins_zvar}{txt_perc}classes of {self.zvar} ({self.agg}) "
#                          f"(not generated: {n_not_generated} classes)",
#                    loc='upper left',
#                    textsize=textsize,
#                    bbox_to_anchor=(1, 1.02))





