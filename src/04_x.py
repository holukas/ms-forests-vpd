import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from diive.core.io.files import load_parquet
from diive.pkgs.analyses.decoupling import SortingBinsMethod

df = pd.read_csv('../OUT/03_siteinfo.csv')
df = df.fillna(np.nan)
# print(df)

# Variables
# neecol = "NEE_CUT_50"
neecol = "NEE_VUT_USTAR50"
# neeqc = "NEE_CUT_50_QC"
neeqc = "NEE_VUT_USTAR50_QC"
tacol = "TA_F"
taqc = "TA_F_QC"
vpdcol = "VPD_F"
vpdqc = "VPD_F_QC"
# swccol = "SWC_F_MDS_1"
# swcqc = "SWC_F_MDS_1_QC"
swinpotcol = "SW_IN_POT"

_df = df.copy()
for ix, row in _df.iterrows():

    print(f"INDEX: {ix}")

    if ix != 2:
        continue

    site = row['SITE']
    igbp = row['IGBP']
    filepath = row['_FILEPATH_PARQUET']
    sitedata = load_parquet(filepath)

    # [print(c) for c in sitedata.columns if "SW_" in c];

    subsetcols = [
        neecol, neeqc,
        swinpotcol,
        tacol, taqc,
        vpdcol, vpdqc,
        # swccol, swcqc,
    ]

    sitedata = sitedata[subsetcols].copy()

    locs_qc0 = (sitedata[neeqc] == 0) & (sitedata[taqc] == 0) & (sitedata[vpdqc] == 0)
    # locs_qc0 = (sitedata[neeqc] == 0) & (sitedata[swcqc] == 0) & (sitedata[taqc] == 0) & (sitedata[vpdqc] == 0)
    sitedata_qc0 = sitedata.loc[locs_qc0].copy()

    locs_dt = sitedata_qc0[swinpotcol] > 20
    sitedata_qc0_dt = sitedata_qc0.loc[locs_dt].copy()

    locs_nt = sitedata_qc0[swinpotcol] <= 20
    sitedata_qc0_nt = sitedata_qc0.loc[locs_nt].copy()

    # Focus on 4 warmest months
    sitedata['MONTH'] = sitedata.index.month
    monthly_avg = sitedata.groupby('MONTH').mean()
    monthly_avg = monthly_avg.sort_values(by='TA_F', ascending=False, inplace=False)
    warmest4 = list(monthly_avg.head(4).index)

    locs = \
        (sitedata_qc0_dt.index.month == warmest4[0]) | (sitedata_qc0_dt.index.month == warmest4[1]) | (
                sitedata_qc0_dt.index.month == warmest4[2]) | (sitedata_qc0_dt.index.month == warmest4[3])

    sitedata_qc0_dt_warmest4 = sitedata_qc0_dt.loc[locs].copy()

    sbm = SortingBinsMethod(df=sitedata_qc0_dt_warmest4,
                            var2_col=tacol,
                            var1_col=vpdcol,
                            var3_col=neecol,
                            n_bins_var1=100,
                            n_subbins_var2=2,
                            convert_to_percentiles=False)
    sbm.calcbins()

    binmedians = sbm.get_binmedians()

    infldf = pd.DataFrame()
    cc = []
    for ix, v in binmedians.items():
        a = v.loc[0, neecol]
        b = v.loc[1, neecol]
        c = b - a
        infldf.loc[ix, 'Z'] = ix
        infldf.loc[ix, 'X'] = v['VPD_F'].median()  #TODO
        infldf.loc[ix, 'DELTA_Y'] = c
        cc.append(c)
        # print(c)

    # _signs = np.sign(cc)

    # print(infldf)

    dy = infldf['DELTA_Y'].cumsum().rolling(window=10, center=True).mean()
    dy_min = dy.min()
    _min = infldf.loc[dy == dy_min, 'X'].values[0]

    plt.scatter(infldf['X'], dy)
    plt.scatter(_min, dy_min, color="red")
    plt.title(f"{site} ({igbp})")
    # infldf['DELTA_Y'].plot()
    plt.show()

    # print(cc)

    # infls = np.where(np.diff(np.sign(cc)))[0]
    # i = infls[-1] + 1
    # print(cc[i])

    sbm.showplot_decoupling_sbm(marker='o', emphasize_lines=True, title=site)

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
