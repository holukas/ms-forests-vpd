"""

Example:
https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Front%20page%20example%20%28XGBoost%29.html
https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html

"""

from pathlib import Path
import shap
import xgboost
import matplotlib.pyplot as plt
import pandas as pd
from diive.core.io.filereader import search_files
from diive.core.io.files import load_parquet

pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

SEARCHDIRS = r"F:\Sync\luhk_work\40 - DATA\Datasets\2024 - ICOS - Ecosystem final quality (L2) product in ETC-Archive format - release 2024-1\2-FLUXNET_HH_PARQUET"
IDENTIFIERS = ['ICOSETC_', '_FLUXNET_HH_L2', '.csv.parquet']

filepaths = search_files(searchdirs=SEARCHDIRS, pattern=r"ICOSETC_*_FLUXNET_HH_L2.csv.parquet")

sites_df = pd.DataFrame()
for ix, filepath in enumerate(filepaths):
    if ix == 1:
        _filename = Path(filepath).name
        splits = _filename.split('_')
        site = splits[1]
        print(f"Reading {site} ...")
        df = load_parquet(filepath=filepath)
        df = df.loc[df['SW_IN_F'] > 20, :].copy()
        # df = df.loc[df['NEE_VUT_50_QC'] == 0, :].copy()
        locs = (df.index.year == 2022)
        # locs = (df.index.year >= 2020) & (df.index.month == 7)
        # df = df.loc[locs, :].copy()
        # [print(c) for c in df.columns if "LE" in c];
        break

# print(df['NEE_VUT_50_QC'].describe())

# means = df.resample('D').mean()
# means.index = means.index.date

X = df[['TA_F', 'VPD_F', 'SW_IN_F', 'SWC_F_MDS_1']].copy()
# X = df[['LE_F_MDS', 'TA_F', 'VPD_F', 'SW_IN_F', 'PPFD_IN', 'PPFD_OUT', 'P_F', 'SWC_F_MDS_1']].copy()
y = df[['NEE_VUT_50']].copy()

X['SWC_F_MDS_1'].plot()
y['NEE_VUT_50'].plot()
plt.show()

# train an XGBoost model
model = xgboost.XGBRegressor().fit(X, y)
# model = CatBoostRegressor(iterations=300, learning_rate=0.1, random_seed=123)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)
expected_value = explainer.expected_value
_shap_values = shap_values.values


shapdf = pd.DataFrame(data=_shap_values, index=X.index, columns=X.columns)
shapdf.index = pd.to_datetime(shapdf.index)
shapdf['SUM_SHAP'] = shapdf.sum(axis=1)
shapdf['NEE_VUT_50'] = df['NEE_VUT_50'].copy()
shapdf['EXPECTED'] = expected_value
shapdf['SUM+EXPECTED'] = shapdf['EXPECTED'].add(shapdf['SUM_SHAP'])

# shapdf['NEE_VUT_50'].plot()
# shapdf['SUM+EXPECTED'].plot()
# shapdf['EXPECTED'].plot()
# plt.legend()
# plt.show()

# means = shapdf.resample('ME').sum()
means = shapdf.groupby(shapdf.index.month).sum()
# means.index = means.index.month
means[['TA_F', 'VPD_F', 'SW_IN_F', 'SWC_F_MDS_1']].plot.bar(stacked=True, figsize=(14, 5))
# means['SUM+EXPECTED'].plot.bar()
# means['EXPECTED'].plot.bar()
# means['SUM_SHAP'].plot()
# plt.legend()
plt.show()

# means.plot()
# means.plot.bar(stacked=True, figsize=(20, 5))
# means.plot.bar(stacked=True, subplots=True)
# plt.show()
# # visualize the first prediction's explanation
# fig = plt.figure()
# # shap.plots.waterfall(shap_values[666], show=False)
# shap.plots.waterfall(shap_values[0])
# plt.savefig(f'my_plot.png', bbox_inches='tight')

# shap.plots.initjs()

# # https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/decision_plot.html
# # features_display = X_display.loc[features.index]
# shap.decision_plot(expected_value, _shap_values)

# # visualize the first prediction's explanation with a force plot
# # shap.plots.force(shap_values[0], show=False, matplotlib=False)
# fig = plt.figure()
# shap.plots.force(shap_values[0], show=False, matplotlib=True)
# plt.savefig(f'my_plot.png', bbox_inches='tight')

# # visualize all the training set predictions
# fig = plt.figure()
# shap.plots.force(shap_values[:500], show=False, matplotlib=True)
# plt.savefig(f'my_plot.png', bbox_inches='tight')

# # shap.plots.heatmap(shap_values[:1000])
# shap.plots.scatter(shap_values[:, "TA_F"])
# shap.plots.scatter(shap_values[:, "PPFD_IN"], color=shap_values)
# shap.plots.scatter(shap_values[:, "VPD_F"], color=shap_values[:, "SW_IN_F"])
# shap.plots.scatter(shap_values[:, "SWC_F_MDS_1"], color=shap_values)

# # create a dependence scatter plot to show the effect of a single feature across the whole dataset
# shap.plots.scatter(shap_values[:, "LE_F_MDS"], color=shap_values[:, "VPD_F"])
# # shap.plots.scatter(shap_values[:, "VPD_F"], color=shap_values[:, "LE_F_MDS"])
# # shap.plots.scatter(shap_values[:, "TA_F"], color=shap_values[:, "VPD_F"])

# # summarize the effects of all the features
# shap.plots.beeswarm(shap_values, plot_size=(16, 9))

# shap.plots.bar(shap_values)



