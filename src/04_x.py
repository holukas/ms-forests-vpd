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

_mins = []

_df = df.copy()
for ix, row in _df.iterrows():

    print(f"INDEX: {ix}")

    if ix != 4:
        continue

    site = row['SITE']
    igbp = row['IGBP']
    filepath = row['_FILEPATH_PARQUET']
    sitedata = load_parquet(filepath)

    # [print(c) for c in sitedata.columns if "NEE_" in c];

    subsetcols = [
        neecol, neeqc,
        swinpotcol,
        tacol, taqc,
        vpdcol, vpdqc,
        # swccol, swcqc,
    ]

    sitedata = sitedata[subsetcols].copy()

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
    sbm.showplot_decoupling_sbm(marker='o', emphasize_lines=True, title=site)

    # Collect deltas
    infldf = pd.DataFrame()
    cc = []
    for _ix, v in binaggs.items():

        from_y = v.loc[0, yvar]
        from_y_p25 = v.loc[0, f"{yvar}_P25"]
        from_y_p75 = v.loc[0, f"{yvar}_P75"]
        infldf.loc[_ix, 'FROM_Y'] = from_y

        to_y = v.loc[1, yvar]
        to_y_p25 = v.loc[1, f"{yvar}_P25"]
        to_y_p75 = v.loc[1, f"{yvar}_P75"]
        infldf.loc[_ix, 'TO_Y'] = to_y
        infldf.loc[_ix, 'TO_Y_P25'] = to_y_p25
        infldf.loc[_ix, 'TO_Y_P75'] = to_y_p75

        delta_y = to_y - from_y
        delta_y_p25 = to_y_p25 - from_y_p25
        delta_y_p75 = to_y_p75 - from_y_p75
        infldf.loc[_ix, 'DELTA_Y'] = delta_y
        infldf.loc[_ix, 'DELTA_Y_P25'] = delta_y_p25
        infldf.loc[_ix, 'DELTA_Y_P75'] = delta_y_p75

        infldf.loc[_ix, 'Z'] = _ix
        infldf.loc[_ix, 'X'] = v.loc[1, xvar]




        # NEE_VUT_USTAR50_P75

    # Plot deltas
    dy = infldf['DELTA_Y']
    plt.scatter(infldf['X'], dy)
    plt.axhline(0)
    plt.title(f"{site} ({igbp}): delta {yvar}")
    plt.show()

    # Plot to y
    # locs = (infldf['TO_Y'] > 0) & (infldf['FROM_Y'] < 0)
    # _infldf = infldf[locs].copy()
    dy = infldf['TO_Y']
    dy2 = infldf['TO_Y_P75']
    dy3 = infldf['TO_Y_P25']
    plt.scatter(infldf['X'], dy)
    plt.scatter(infldf['X'], dy2)
    plt.scatter(infldf['X'], dy3)
    plt.axhline(0)
    plt.title(f"{site} ({igbp}): TO_Y_P25 / TO_Y / TO_Y_P75")
    plt.show()

    # Plot cumulative deltas, with minimum
    dy = infldf['DELTA_Y'].cumsum()
    dy_p25 = infldf['DELTA_Y_P25'].cumsum()
    dy_p75 = infldf['DELTA_Y_P75'].cumsum()
    dy_min = dy.min()
    dy_min_p25 = dy_p25.min()
    dy_min_p75 = dy_p75.min()
    _min = infldf.loc[dy == dy_min, 'X'].values[0]
    _min_p25 = infldf.loc[dy_p25 == dy_min_p25, 'X'].values[0]
    _min_p75 = infldf.loc[dy_p75 == dy_min_p75, 'X'].values[0]
    plt.axvline(_min)
    plt.axvline(_min_p25)
    plt.axvline(_min_p75)
    plt.scatter(infldf['X'], dy, label=f"agg: min x = {_min}")
    plt.scatter(infldf['X'], dy_p25, label=f"p25: min x = {_min_p25}")
    plt.scatter(infldf['X'], dy_p75, label=f"p75: min x = {_min_p75}")
    plt.scatter(_min, dy_min, color="red")
    plt.scatter(_min_p25, dy_min_p25, color="red")
    plt.scatter(_min_p75, dy_min_p75, color="red")
    plt.title(f"{site} ({igbp}): cumulative delta {yvar}")
    plt.legend()
    plt.show()

    _mins.append(_min)

print(_mins)

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
