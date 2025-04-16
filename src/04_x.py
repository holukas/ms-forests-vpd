import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from diive.core.io.files import load_parquet
from diive.pkgs.analyses.decoupling import SortingBinsMethod

df = pd.read_csv('../OUT/03_siteinfo.csv')
df = df.fillna(np.nan)

# Variables
neecol = "NEE_VUT_50"
neeqc = "NEE_VUT_50_QC"
# neecol = "NEE_CUT_50"
# neeqc = "NEE_CUT_50_QC"
# neecol = "NEE_VUT_USTAR50"
# neeqc = "NEE_VUT_USTAR50_QC"
tacol = "TA_F"
taqc = "TA_F_QC"
vpdcol = "VPD_F"
vpdqc = "VPD_F_QC"
swccol = "SWC_F_MDS_1"
swcqc = "SWC_F_MDS_1_QC"
swinpotcol = "SW_IN_POT"

subsetcols = [
    neecol, neeqc,
    swinpotcol,
    tacol, taqc,
    vpdcol, vpdqc,
    # swccol, swcqc,
]

_mins = []

_df = df.copy()
for ix, row in _df.iterrows():

    print(f"INDEX: {ix}")

    if ix != 0:
        continue

    site = row['SITE']
    igbp = row['IGBP']
    filepath = row['_FILEPATH_PARQUET']
    sitedata = load_parquet(filepath)
    # [print(c) for c in sitedata.columns if "NEE_" in c];
    sitedata = sitedata[subsetcols].copy()

    sitedata[vpdcol].describe()

    # Highest-quality fluxes and meteo
    locs_qc0 = (sitedata[neeqc] == 0) & (sitedata[taqc] == 0) & (sitedata[vpdqc] == 0)
    # locs_qc0 = (sitedata[neeqc] == 0) & (sitedata[swcqc] == 0) & (sitedata[taqc] == 0) & (sitedata[vpdqc] == 0)
    sitedata_qc0 = sitedata.loc[locs_qc0].copy()

    # Daytime data
    locs_dt = sitedata_qc0[swinpotcol] > 20
    sitedata_qc0_dt = sitedata_qc0.loc[locs_dt].copy()

    # Nighttime data
    locs_nt = sitedata_qc0[swinpotcol] <= 20
    sitedata_qc0_nt = sitedata_qc0.loc[locs_nt].copy()

    # Focus on 6 warmest months
    sitedata['MONTH'] = sitedata.index.month
    monthly_avg = sitedata.groupby('MONTH').mean()
    monthly_avg = monthly_avg.sort_values(by='TA_F', ascending=False, inplace=False)
    warmest6 = list(monthly_avg.head(6).index)
    locs = \
        (sitedata_qc0_dt.index.month == warmest6[0]) | (sitedata_qc0_dt.index.month == warmest6[1]) | (
                sitedata_qc0_dt.index.month == warmest6[2]) | (sitedata_qc0_dt.index.month == warmest6[3]) | (
                    sitedata_qc0_dt.index.month == warmest6[4]) | (sitedata_qc0_dt.index.month == warmest6[5])
    sitedata_qc0_dt_warmest6 = sitedata_qc0_dt.loc[locs].copy()

    # # Checking sum of z-scores
    # from scipy.stats import zscore
    # sitedata_qc0_dt_warmest6 = sitedata_qc0_dt_warmest6.apply(zscore)
    # zsum = sitedata_qc0_dt_warmest6[tacol].add(sitedata_qc0_dt_warmest6[vpdcol])
    # yy = sitedata_qc0_dt_warmest6[neecol]
    # plt.scatter(zsum, yy)
    # plt.scatter(zsum, sitedata_qc0_dt_warmest6[tacol])
    # plt.scatter(zsum, sitedata_qc0_dt_warmest6[vpdcol])
    # plt.title("Sum of z-scores")
    # plt.show()

    # Used vars
    xvar = vpdcol
    yvar = neecol
    zvar = tacol

    # Calculate bins
    start_time = time.time()
    sbm = SortingBinsMethod(df=sitedata_qc0_dt_warmest6,
                            xvar=xvar,
                            yvar=yvar,
                            zvar=zvar,
                            n_bins_z=100,
                            n_bins_x=2,
                            conversion=None,
                            # conversion='z-score',
                            agg='median')
    sbm.calcbins()
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"Execution time: {time_elapsed}")
    binaggs = sbm.get_binaggs()

    # sbm.showplot_decoupling_sbm(marker='o', emphasize_lines=True, title=site)

    # Collect deltas
    infldf = pd.DataFrame()
    cc = []
    for _ix, v in binaggs.items():

        from_y = v.loc[0, yvar]
        from_y_p16 = v.loc[0, f"{yvar}_P16"]
        from_y_p84 = v.loc[0, f"{yvar}_P84"]
        infldf.loc[_ix, 'FROM_Y'] = from_y

        to_y = v.loc[1, yvar]
        to_y_p16 = v.loc[1, f"{yvar}_P16"]
        to_y_p84 = v.loc[1, f"{yvar}_P84"]
        infldf.loc[_ix, 'TO_Y'] = to_y
        infldf.loc[_ix, 'TO_Y_P16'] = to_y_p16
        infldf.loc[_ix, 'TO_Y_P84'] = to_y_p84

        delta_y = to_y - from_y
        delta_y_p16 = to_y_p16 - from_y_p16
        delta_y_p84 = to_y_p84 - from_y_p84
        infldf.loc[_ix, 'DELTA_Y'] = delta_y
        infldf.loc[_ix, 'DELTA_Y_P16'] = delta_y_p16
        infldf.loc[_ix, 'DELTA_Y_P84'] = delta_y_p84

        infldf.loc[_ix, 'Z'] = _ix
        infldf.loc[_ix, 'X'] = v.loc[1, xvar]
        infldf.loc[_ix, 'DELTA_X'] = v.loc[1, xvar] - v.loc[0, xvar]

        infldf.loc[_ix, 'SPEED'] = infldf.loc[_ix, 'DELTA_Y'] / infldf.loc[_ix, 'DELTA_X']
        infldf.loc[_ix, 'SPEED_P16'] = infldf.loc[_ix, 'DELTA_Y_P16'] / infldf.loc[_ix, 'DELTA_X']
        infldf.loc[_ix, 'SPEED_P84'] = infldf.loc[_ix, 'DELTA_Y_P84'] / infldf.loc[_ix, 'DELTA_X']

        # NEE_VUT_USTAR50_P84

    # Speed, change of flux normalized to change of VPD

    plt.scatter(infldf['Z'], infldf['SPEED'])
    plt.scatter(infldf['Z'], infldf['SPEED_P16'])
    plt.scatter(infldf['Z'], infldf['SPEED_P84'])
    # plt.plot(infldf['SPEED'])
    # plt.locator_params(axis='x', nbins=5)
    plt.title("Speed")
    plt.show()

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
