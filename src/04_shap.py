import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from diive.core.io.files import load_parquet

df = pd.read_csv('../OUT/03_siteinfo.csv')
df = df.fillna(np.nan)

tacol = 'TA_F'
taqc = 'TA_F_QC'
vpdcol = 'VPD_F'
vpdqc = 'VPD_F_QC'
preccol = 'P_F'
swccol = 'SWC_F_MDS_1'
fluxcol = 'GPP_NT_VUT_50'
# fluxcol = 'NEE_VUT_50'
fluxqc = "NEE_VUT_50_QC"
swinpotcol = "SW_IN_POT"  # For daytime/nighttime
swincol = "SW_IN_F"

_df = df.copy()
for ix, row in _df.iterrows():

    # if row['SITE'] != 'CH-Dav':
    if row['SITE'] != 'BE-Bra':
        continue

    print(f"\nLoading data for site {row['SITE']} ...")

    filepath = row['_FILEPATH_PARQUET']
    sitedata = load_parquet(filepath)

    # [print(c) for c in sitedata.columns if "GPP" in c];

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

    subset = sitedata_warmest6_qc0_dt[[fluxcol, tacol, vpdcol, swccol, swincol]].copy()
    subset = subset.dropna()
    # subset = subset.loc[subset.index.year == 2019].copy()
    y = np.array(subset[fluxcol])
    X = subset[[tacol, vpdcol, swccol, swincol]].copy()
    # X = np.array(subset.drop(fluxcol, axis=1))
    model = xgb.XGBRegressor()
    model = model.fit(X=X, y=y)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    # shap_interaction_values = explainer.shap_interaction_values(X)
    # shap.interaction_plot(shap_interaction_values[i], X.iloc[[i]], feature_names=X.columns)
    # shap.plots.waterfall(shap_values[1])
    shap.plots.scatter(shap_values[:, tacol], color=shap_values)
    shap.plots.scatter(shap_values[:, vpdcol], color=shap_values)
    shap.plots.scatter(shap_values[:, swccol], color=shap_values)
    shap.plots.scatter(shap_values[:, swincol], color=shap_values)
    shap.plots.bar(shap_values)
    # shap.summary_plot(shap_values)
# df.to_csv("../OUT/03_siteinfo.csv", index=False)
