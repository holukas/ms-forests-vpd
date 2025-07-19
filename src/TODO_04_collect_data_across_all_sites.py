import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from diive.core.io.files import load_parquet
from diive.pkgs.analyses.decoupling import SortingBinsMethod

# Files
INFILE = "../OUT/03_siteinfo.csv"
OUTFILE = "../OUT/04_bins_zscores_allsites.csv"

df = pd.read_csv(INFILE)
df = df.fillna(np.nan)

# Variables
# fluxcol = "GPP_DT_VUT_50"
# fluxcol = "RECO_NT_VUT_50"
fluxcol = "NEE_VUT_50"
fluxqc = "NEE_VUT_50_QC"
tacol = "TA_F"
taqc = "TA_F_QC"
vpdcol = "VPD_F"
vpdqc = "VPD_F_QC"
swinpotcol = "SW_IN_POT"  # For daytime/nighttime

# Used vars
xvar = vpdcol
zvar = tacol
yvar = fluxcol

subsetcols = [
    fluxcol,
    fluxqc,
    swinpotcol,
    tacol, taqc,
    vpdcol, vpdqc
]

# Dataframe for collecting data across all sites
coll = pd.DataFrame()

for ix, row in df.iterrows():

    if ix != 1:
        continue

    site = row['SITE']
    igbp = row['IGBP']

    print(f"\nWorking on site #{ix + 1} {site} ({igbp}) ...")

    # Load data and required columns
    filepath = row['_FILEPATH_PARQUET']
    sitedata = load_parquet(filepath)
    # [print(c) for c in sitedata.columns if "GPP" in c];
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
    locs_qc0 = (sitedata_warmest6[fluxqc] == 0) & (sitedata_warmest6[taqc] == 0) & (sitedata_warmest6[vpdqc] == 0)
    sitedata_warmest6_qc0 = sitedata_warmest6.loc[locs_qc0].copy()

    # Keep daytime data
    locs_dt = sitedata_warmest6_qc0[swinpotcol] > 20
    sitedata_warmest6_qc0_dt = sitedata_warmest6_qc0.loc[locs_dt].copy()
    # locs_dt = sitedata_warmest6[swinpotcol] > 20
    # sitedata_warmest6_qc0_dt = sitedata_warmest6.loc[locs_dt].copy()

    # Keep required columns only
    sitedata_warmest6_qc0_dt = sitedata_warmest6_qc0_dt[[xvar, yvar, zvar]].copy()

    fig = plt.figure(figsize=(36, 18))
    gs = gridspec.GridSpec(2, 2)  # rows, cols
    # gs.update(wspace=.2, hspace=1, left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    # ax = self._plot_bins(ax=ax, n_col=n_col, emphasize_lines=emphasize_lines, **kwargs)

    # Calculate bins
    settings = dict(df=sitedata_warmest6_qc0_dt,
                    xvar=xvar, yvar=yvar, zvar=zvar,
                    n_bins_z=100, n_bins_x=2, )

    # Measured
    sbm_measured = SortingBinsMethod(conversion=None, agg='median', **settings)
    sbm_measured.calcbins()
    binaggs_medians = sbm_measured.get_binaggs()
    sbm_measured.showplot_decoupling_sbm(ax=ax1, marker='o', emphasize_lines=True, title=f"{site} ({igbp})",
                                         legend=True)
    ax1.set_title("Measured", size=20)

    # Measured: deltas
    infldf = pd.DataFrame()
    for _ix, v in binaggs_medians.items():
        from_y = v.loc[0, yvar]
        to_y = v.loc[1, yvar]
        delta_y = to_y - from_y
        infldf.loc[_ix, 'FROM_Y'] = from_y
        infldf.loc[_ix, 'TO_Y'] = to_y
        infldf.loc[_ix, 'DELTA_Y'] = delta_y
        infldf.loc[_ix, 'X'] = v.loc[1, xvar]
        infldf.loc[_ix, 'DELTA_X'] = v.loc[1, xvar] - v.loc[0, xvar]
        infldf.loc[_ix, 'SLOPE'] = infldf.loc[_ix, 'DELTA_Y'] / infldf.loc[_ix, 'DELTA_X']
    dy = infldf['DELTA_Y']

    ax3.scatter(infldf['X'], infldf['SLOPE'])
    ax3.axhline(0)
    ax3.set_title("SLOPE")

    # z-scores
    sbm_zscores = SortingBinsMethod(conversion='z-score', agg='median', **settings)
    sbm_zscores.calcbins()
    binaggs_zscores = sbm_zscores.get_binaggs()
    sbm_zscores.showplot_decoupling_sbm(ax=ax2, marker='o', emphasize_lines=True, title=f"{site} ({igbp})", legend=True)
    ax2.set_title("z-scores", size=20)

    # z-scores: deltas
    infldf = pd.DataFrame()
    for _ix, v in binaggs_zscores.items():
        from_y = v.loc[0, yvar]
        to_y = v.loc[1, yvar]
        delta_y = to_y - from_y
        infldf.loc[_ix, 'FROM_Y'] = from_y
        infldf.loc[_ix, 'TO_Y'] = to_y
        infldf.loc[_ix, 'DELTA_Y'] = delta_y
        infldf.loc[_ix, 'X'] = v.loc[1, xvar]
        infldf.loc[_ix, 'DELTA_X'] = v.loc[1, xvar] - v.loc[0, xvar]
        infldf.loc[_ix, 'SLOPE'] = infldf.loc[_ix, 'DELTA_Y'] / infldf.loc[_ix, 'DELTA_X']
    dy = infldf['DELTA_Y']

    ax4.scatter(infldf['X'], infldf['SLOPE'])
    ax4.axhline(0)
    ax4.set_title("SLOPE")

    # sbm.showplot_decoupling_sbm(marker='o', emphasize_lines=True, title=f"{site} ({igbp})", legend=True)

    fig.suptitle(f"{site} ({igbp})", fontsize=16)
    fig.tight_layout()
    fig.show()

    # Testing: collect all z-scores for all sites in one df
    merged_rows = pd.DataFrame()
    for k, v in binaggs_medians.items():
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

coll.to_csv(OUTFILE, index=False)

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
#     infldf.loc[_ix, 'SLOPE'] = infldf.loc[_ix, 'DELTA_Y'] / infldf.loc[_ix, 'DELTA_X']
#     infldf.loc[_ix, 'SPEED_P16'] = infldf.loc[_ix, 'DELTA_Y_P16'] / infldf.loc[_ix, 'DELTA_X']
#     infldf.loc[_ix, 'SPEED_P84'] = infldf.loc[_ix, 'DELTA_Y_P84'] / infldf.loc[_ix, 'DELTA_X']

# NEE_VUT_USTAR50_P84

# Speed, change of flux normalized to change of VPD

# plt.scatter(infldf['Z'], infldf['SLOPE'], label="agg")
# # plt.scatter(infldf['Z'], infldf['SPEED_P16'], label="P16")
# # plt.scatter(infldf['Z'], infldf['SPEED_P84'], label="P84")
# # plt.plot(infldf['SLOPE'])
# plt.locator_params(axis='x', nbins=20)
# plt.title("Speed")
# plt.legend()
# plt.show()

#
#
#
#
#
#

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
