import time
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from diive.core.io.files import load_parquet
from diive.pkgs.analyses.decoupling import SortingBinsMethod
from scipy.stats import binned_statistic

df = pd.read_csv('../OUT/03_siteinfo.csv')
df = df.fillna(np.nan)

# Variables
neecol = "NEE_VUT_50"
neeqc = "NEE_VUT_50_QC"
tacol = "TA_F"
taqc = "TA_F_QC"
vpdcol = "VPD_F"
vpdqc = "VPD_F_QC"
swinpotcol = "SW_IN_POT"  # For daytime/nighttime

# Used vars
xvar = vpdcol
zvar = tacol
yvar = neecol

subsetcols = [
    neecol, neeqc,
    swinpotcol,
    tacol, taqc,
    vpdcol, vpdqc
]

# Dataframe for collecting data across all sites
coll = pd.DataFrame()

for ix, row in df.iterrows():

    # if ix >1:
    #     break

    site = row['SITE']
    igbp = row['IGBP']

    print(f"\nWorking on site #{ix + 1} {site} ({igbp}) ...")

    # Load data and required columns
    filepath = row['_FILEPATH_PARQUET']
    sitedata = load_parquet(filepath)
    sitedata = sitedata[subsetcols].copy()

    # Keep data for 6 warmest months
    sitedata['MONTH'] = sitedata.index.month
    monthly_avg = sitedata.groupby('MONTH').mean()
    monthly_avg = monthly_avg.sort_values(by='TA_F', ascending=False, inplace=False)
    warmest6 = list(monthly_avg.head(6).index)
    locs = \
        (sitedata.index.month == warmest6[0]) | (sitedata.index.month == warmest6[1]) | (
                sitedata.index.month == warmest6[2]) | (sitedata.index.month == warmest6[3]) | (
                sitedata.index.month == warmest6[4]) | (sitedata.index.month == warmest6[5])
    sitedata_warmest6 = sitedata.loc[locs].copy()

    # Keep directly measured fluxes and meteo, no gap-filled data
    locs_qc0 = (sitedata_warmest6[neeqc] == 0) & (sitedata_warmest6[taqc] == 0) & (sitedata_warmest6[vpdqc] == 0)
    sitedata_warmest6_qc0 = sitedata_warmest6.loc[locs_qc0].copy()

    # Keep daytime data
    locs_dt = sitedata_warmest6_qc0[swinpotcol] > 20
    sitedata_warmest6_qc0_dt = sitedata_warmest6_qc0.loc[locs_dt].copy()

    # Keep required columns only
    sitedata_warmest6_qc0_dt = sitedata_warmest6_qc0_dt[[xvar, yvar, zvar]].copy()

    # Calculate bins
    sbm = SortingBinsMethod(df=sitedata_warmest6_qc0_dt,
                            xvar=xvar,
                            yvar=yvar,
                            zvar=zvar,
                            n_bins_z=100,
                            n_bins_x=2,
                            conversion='z-score',
                            agg='median')
    sbm.calcbins()
    binaggs = sbm.get_binaggs()
    sbm.showplot_decoupling_sbm(marker='o', emphasize_lines=True, title=f"{site} ({igbp})", legend=True)

    # Testing: collect all z-scores for all sites in one df
    merged_rows = pd.DataFrame()
    for k, v in binaggs.items():
        from_row = v.loc[0].copy()
        index_from = v.loc[0].index.tolist()
        index_from = [f"FROM_{i}" for i in index_from]
        from_row.index = index_from

        to_row = v.loc[1].copy()
        index_to = v.loc[1].index.tolist()
        index_to = [f"TO_{i}" for i in index_to]
        to_row.index = index_to

        new_row = pd.concat([from_row, to_row], axis=0, ignore_index=False)
        new_row = pd.DataFrame(new_row).transpose()
        new_row.index = [k]
        merged_rows = pd.concat([merged_rows, new_row])

    # Add site info
    merged_rows.insert(0, 'SITE', site)
    merged_rows.insert(1, 'IGBP', igbp)

    # Merge this site data with collection across all sites
    coll = pd.concat([coll, merged_rows], axis=0)

coll.to_csv("../OUT/04_bins_zscores_allsites.csv", index=False)


# fig = plt.figure(figsize=(16, 9))
# gs = gridspec.GridSpec(1, 1)  # rows, cols
# # gs.update(wspace=.2, hspace=1, left=.1, right=.9, top=.85, bottom=.1)
#
# ax = fig.add_subplot(gs[0, 0])
#
# f"{coll.index}"
#
# from_xvar = f"FROM_{xvar}"
# to_xvar = f"TO_{xvar}"
# from_yvar = f"FROM_{yvar}"
# to_yvar = f"TO_{yvar}"
#
# from_xerror_pos = f"FROM_xerror_pos"
# to_xerror_pos = f"TO_xerror_pos"
# from_xerror_neg = f"FROM_xerror_neg"
# to_xerror_neg = f"TO_xerror_neg"
#
# from_yerror_pos = f"FROM_yerror_pos"
# to_yerror_pos = f"TO_yerror_pos"
# from_yerror_neg = f"FROM_yerror_neg"
# to_yerror_neg = f"TO_yerror_neg"
#
# for k, row in coll.iterrows():
#     x = [row[from_xvar], row[to_xvar]]
#     y = [row[from_yvar], row[to_yvar]]
#     xerror_pos = [row[from_xerror_pos], row[to_xerror_pos]]
#     xerror_neg = [row[from_xerror_neg], row[to_xerror_neg]]
#     yerror_pos = [row[from_yerror_pos], row[to_yerror_pos]]
#     yerror_neg = [row[from_yerror_neg], row[to_yerror_neg]]
#
#     ax.plot(x, y,
#             ls='-', lw=3, ms=14, label="X", color="black",
#             marker='o', mec='k', mew=1, alpha=1, zorder=99)
#     ax.errorbar(x=x, y=y,
#                 xerr=[xerror_neg, xerror_pos],
#                 yerr=[yerror_neg, yerror_pos],
#                 elinewidth=8, ecolor="red", alpha=.1, lw=0)

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




# fig.suptitle("All z-scores", fontsize=16)
# fig.tight_layout()
# fig.show()




    # # Collect deltas
    # infldf = pd.DataFrame()
    # cc = []
    # for _ix, v in binaggs.items():
    #     from_y = v.loc[0, yvar]
    #     from_y_p16 = v.loc[0, f"{yvar}_P16"]
    #     from_y_p84 = v.loc[0, f"{yvar}_P84"]
    #     infldf.loc[_ix, 'FROM_Y'] = from_y
    #
    #     to_y = v.loc[1, yvar]
    #     to_y_p16 = v.loc[1, f"{yvar}_P16"]
    #     to_y_p84 = v.loc[1, f"{yvar}_P84"]
    #     infldf.loc[_ix, 'TO_Y'] = to_y
    #     infldf.loc[_ix, 'TO_Y_P16'] = to_y_p16
    #     infldf.loc[_ix, 'TO_Y_P84'] = to_y_p84
    #
    #     delta_y = to_y - from_y
    #     delta_y_p16 = to_y_p16 - from_y_p16
    #     delta_y_p84 = to_y_p84 - from_y_p84
    #     infldf.loc[_ix, 'DELTA_Y'] = delta_y
    #     infldf.loc[_ix, 'DELTA_Y_P16'] = delta_y_p16
    #     infldf.loc[_ix, 'DELTA_Y_P84'] = delta_y_p84
    #
    #     infldf.loc[_ix, 'Z'] = float(_ix)
    #     infldf.loc[_ix, 'X'] = v.loc[1, xvar]
    #     infldf.loc[_ix, 'DELTA_X'] = v.loc[1, xvar] - v.loc[0, xvar]
    #
    #     infldf.loc[_ix, 'SPEED'] = infldf.loc[_ix, 'DELTA_Y'] / infldf.loc[_ix, 'DELTA_X']
    #     infldf.loc[_ix, 'SPEED_P16'] = infldf.loc[_ix, 'DELTA_Y_P16'] / infldf.loc[_ix, 'DELTA_X']
    #     infldf.loc[_ix, 'SPEED_P84'] = infldf.loc[_ix, 'DELTA_Y_P84'] / infldf.loc[_ix, 'DELTA_X']

        # NEE_VUT_USTAR50_P84

    # Speed, change of flux normalized to change of VPD

    # plt.scatter(infldf['Z'], infldf['SPEED'], label="agg")
    # # plt.scatter(infldf['Z'], infldf['SPEED_P16'], label="P16")
    # # plt.scatter(infldf['Z'], infldf['SPEED_P84'], label="P84")
    # # plt.plot(infldf['SPEED'])
    # plt.locator_params(axis='x', nbins=20)
    # plt.title("Speed")
    # plt.legend()
    # plt.show()

    # # Plot deltas
    # dy = infldf['DELTA_Y']
    # plt.scatter(infldf['X'], dy)
    # plt.axhline(0)
    # plt.title(f"{site} ({igbp}): delta {yvar}")
    # plt.show()

#     # Plot to y
#     # locs = (infldf['TO_Y'] > 0) & (infldf['FROM_Y'] < 0)
#     # _infldf = infldf[locs].copy()
#     dy = infldf['TO_Y']
#     dy2 = infldf['TO_Y_P84']
#     dy3 = infldf['TO_Y_P16']
#     plt.scatter(infldf['X'], dy)
#     plt.scatter(infldf['X'], dy2)
#     plt.scatter(infldf['X'], dy3)
#     plt.axhline(0)
#     plt.title(f"{site} ({igbp}): TO_Y_P16 / TO_Y / TO_Y_P84")
#     plt.show()
#
#     # Plot cumulative deltas, with minimum
#     dy = infldf['DELTA_Y'].cumsum()
#     dy_p16 = infldf['DELTA_Y_P16'].cumsum()
#     dy_p84 = infldf['DELTA_Y_P84'].cumsum()
#     dy_min = dy.min()
#     dy_min_p16 = dy_p16.min()
#     dy_min_p84 = dy_p84.min()
#     _min = infldf.loc[dy == dy_min, 'X'].values[0]
#     _min_p16 = infldf.loc[dy_p16 == dy_min_p16, 'X'].values[0]
#     _min_p84 = infldf.loc[dy_p84 == dy_min_p84, 'X'].values[0]
#     plt.axvline(_min)
#     plt.axvline(_min_p16)
#     plt.axvline(_min_p84)
#     plt.scatter(infldf['X'], dy, label=f"agg: min x = {_min}")
#     plt.scatter(infldf['X'], dy_p16, label=f"P16: min x = {_min_p16}")
#     plt.scatter(infldf['X'], dy_p84, label=f"P84: min x = {_min_p84}")
#     plt.scatter(_min, dy_min, color="red")
#     plt.scatter(_min_p16, dy_min_p16, color="red")
#     plt.scatter(_min_p84, dy_min_p84, color="red")
#     plt.title(f"{site} ({igbp}): cumulative delta {yvar}")
#     plt.legend()
#     plt.show()
#
#     _mins.append(_min)
#
# print(_mins)

# _signs = np.sign(cc)

# print(infldf)

# dy = infldf['DELTA_Y']
# dy = infldf['DELTA_Y'].cumsum()
# # dy = infldf['DELTA_Y'].cumsum().rolling(window=10, center=True).mean()
# dy_min = dy.min()
# _min = infldf.loc[dy == dy_min, 'X'].values[0]

# plt.scatter(infldf['X'], dy)
# plt.scatter(_min, dy_min, color="red")
# plt.title(f"{site} ({igbp})")
# infldf['DELTA_Y'].plot()
# plt.show()

# print(cc)

# infls = np.where(np.diff(np.sign(cc)))[0]
# i = infls[-1] + 1
# print(cc[i])


# # Plot daytime
# x = sitedata_qc0_dt[vpdcol]
# y = sitedata_qc0_dt[neecol]
# plt.scatter(x, y, alpha=0.1)
# plt.show()
#
# # Plot nighttime
# x = sitedata_qc0_nt[vpdcol]
# y = sitedata_qc0_nt[neecol]
# plt.scatter(x, y, alpha=0.1)
# plt.show()
#
# # sitedata.plot(x_compat=True, subplots=True)
# # plt.show()
