"""

Example:
https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Front%20page%20example%20%28XGBoost%29.html
https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html

"""

import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from diive.core.io.files import load_parquet
from diive.core.plotting.scatter import ScatterXY

pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
pd.set_option('display.width', 1500)


class ShapAnalysis:
    TACOL = 'TA_F'
    SW_IN_COL = 'SW_IN_F'
    VPD_COL = 'VPD_F'
    SWC_COL = 'SWC_F_MDS_1'
    # LE_COL = 'LE_F_MDS'

    def __init__(self, siteinfo: pd.DataFrame, targetcol: str):
        self.siteinfo = siteinfo
        self.targetcol = targetcol

        self.sitelist = siteinfo['SITE'].tolist()

    def run(self):
        print(self.sitelist)
        self.siteinfo.apply(self._run_site, axis=1)

    def _run_site(self, row):
        # print(row)
        filepath = row['.filepath']
        # filepath = str(filepath).replace('L:', 'F:')
        df = load_parquet(filepath)
        featurecols = [self.TACOL, self.SW_IN_COL, self.VPD_COL, self.SWC_COL]
        subsetcols = featurecols.copy()
        subsetcols.append(self.targetcol)
        subset = df[subsetcols].copy()
        subset = subset.dropna()
        # [print(v) for v in df.columns if "LE" in v]

        X = subset[featurecols].copy()
        y = subset[self.targetcol].copy()
        model = xgb.XGBRegressor().fit(X, y)
        # explain the model's predictions using SHAP
        # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        expected_value = explainer.expected_value
        _shap_values = shap_values.values

        # Add absolute values to shapdf
        shapdf = X.copy()
        shapcolnames = [f"SHAP_{c}" for c in X.columns]
        _tempdf = pd.DataFrame(data=_shap_values, index=X.index, columns=shapcolnames)
        shapdf = pd.concat([shapdf, _tempdf], axis=1)
        shapdf.index = pd.to_datetime(shapdf.index)
        shapdf['SUM_SHAP'] = shapdf[shapcolnames].sum(axis=1)
        shapdf[FLUXCOL] = subset[FLUXCOL].copy()
        shapdf['EXPECTED'] = expected_value
        shapdf['SUM+EXPECTED'] = shapdf['EXPECTED'].add(shapdf['SUM_SHAP'])

        # ScatterXY(x=shapdf[TACOL], y=shapdf[f"SHAP_{TACOL}"], nbins=20, binagg='mean').plot()
        # ScatterXY(x=shapdf[VPD_COL], y=shapdf[f"SHAP_{VPD_COL}"], nbins=20, binagg='mean').plot()

        x = TACOL
        y = f"SHAP_{x}"
        test = shapdf[[x, y]].copy()
        test = test.sort_values(by=f"{x}")
        test = test.reset_index(drop=True)
        from diive.pkgs.fits.fitter import BinFitterCP
        bf = BinFitterCP(
            df=test,
            # n_bootstraps=2,
            xcol=x,
            ycol=y,
            # predict_max_x=None,
            # predict_min_x=None,
            n_predictions=1000,
            n_bins_x=50,
            bins_y_agg='mean',
            fit_type='quadratic_offset'  # 'linear', 'quadratic_offset', 'quadratic'
        )
        bf.run()
        fit_results = bf.fit_results
        # bf.showplot_binfitter(highlight_year=None, xlabel=xlabel, ylabel=ylabel)

        bf.showplot(
            show_unbinned_data=False,
            show_bin_variation=False,
            showfit=True,
            # xlim=(0, 30),
            # ylim=(-1, 0)
        )

        from diive.core.plotting.heatmap_datetime import HeatmapDateTime
        HeatmapDateTime(series=shapdf[y]).show()


#         # todo fit to bins, binfitter?
#         from diive.core.dfun.fits import fit_to_bins_polyreg
#         _df, predicted_col, predicted_results = fit_to_bins_polyreg(df=test, x_col=VPD_COL, y_col=f"SHAP_{VPD_COL}", degree=1)
#         _df.columns
#         _df[[f"SHAP_{VPD_COL}", 'predicted']].plot()
#         from diive.core.plotting.scatter import ScatterXY
#         ScatterXY(x=test[VPD_COL], y=test[f"SHAP_{VPD_COL}"]).plot()
#         plt.show()
#
#         from diive.pkgs.analyses.optimumrange import FindOptimumRange
#         optrange = FindOptimumRange(df=shapdf, xcol=f"SHAP_{TACOL}", ycol=TACOL, define_optimum="max", rwinsize=0.3)
#         optrange.find_optimum()
#         optrange.plot_vals_in_optimum_range()
#
#         # means = shapdf.resample('ME').sum()
#         means = shapdf.groupby(shapdf.index.isocalendar().week).mean()
#         # means.index = means.index.month
#         shapcols = [f"SHAP_{f}" for f in featurecols]
#         means[shapcols].plot.bar(stacked=True, figsize=(14, 5))
#         # means['SUM+EXPECTED'].plot.bar()
#         # means['EXPECTED'].plot.bar()
#         # means['SUM_SHAP'].plot()
#         # plt.legend()
#         plt.show()
#
# # SUBSETCOLS = [FLUXCOL, TACOL, SW_IN_COL, VPD_COL, SWC_COL, FLUXQCCOL]
# # FEATURES = [TACOL, SW_IN_COL, VPD_COL, SWC_COL]
# #
# # # [print(f) for f in filepaths]
# #
# # for ix, f in enumerate(filepaths):
# #     if ix < 10:
# #         continue
# #     df = load_parquet(filepath=f)
# #     # [print(v) for v in df.columns if "GPP" in v]
# #     print("###")
# #
# #     subset = df[SUBSETCOLS].copy()
# #     # locs = (subset[FLUXQCCOL] == 0)
# #     # locs = (subset[FLUXQCCOL] == 0) & (subset[SW_IN_COL] > 20)
# #     # subset = subset[locs].copy()
# #     subset = subset.dropna()
# #     X = subset[FEATURES].copy()
# #     y = subset[FLUXCOL].copy()
# #
# #     # # todo Testing: normalization to z-scores
# #     # unique_yrs = flux.index.year.unique()
# #     # zflux = pd.Series()
# #     # for ix, unique_yr in enumerate(unique_yrs):
# #     #     locs = flux.index.year == unique_yr
# #     #     _zflux = flux[locs].copy()
# #     #     _zflux = zscore(series=_zflux, absolute=False)
# #     #     if ix == 0:
# #     #         zflux = _zflux.copy()
# #     #     else:
# #     #         zflux = pd.concat([zflux, _zflux], axis=0)
# #     # zflux = zscore(series=flux, absolute=False)
# #     # zflux.plot(title=f"{f.name}")
# #     # plt.show()
# #     # # df['NEE_VUT_REF'].cumsum().plot(title=f"{f.name}")
# #     # df['NEE_VUT_REF'].plot(title=f"{f.name}")
# #     # plt.show()
# #
# #     # train an XGBoost model
# #     model = xgboost.XGBRegressor().fit(X, y)
# #     # model = CatBoostRegressor(iterations=300, learning_rate=0.1, random_seed=123)
# #
# #     # explain the model's predictions using SHAP
# #     # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
# #     explainer = shap.TreeExplainer(model)
# #     shap_values = explainer(X)
# #     expected_value = explainer.expected_value
# #     _shap_values = shap_values.values
# #
# #     # todo Add absolute values to shapdf
# #     shapdf = X.copy()
# #
# #     shapcolnames = [f"SHAP_{c}" for c in X.columns]
# #     _tempdf = pd.DataFrame(data=_shap_values, index=X.index, columns=shapcolnames)
# #
# #     shapdf = pd.concat([shapdf, _tempdf], axis=1)
# #     shapdf.index = pd.to_datetime(shapdf.index)
# #     shapdf['SUM_SHAP'] = shapdf[shapcolnames].sum(axis=1)
# #     shapdf[FLUXCOL] = df[FLUXCOL].copy()
# #     shapdf['EXPECTED'] = expected_value
# #     shapdf['SUM+EXPECTED'] = shapdf['EXPECTED'].add(shapdf['SUM_SHAP'])
# #
# #     from diive.core.plotting.scatter import ScatterXY
# #
# #     ScatterXY(x=shapdf[TACOL], y=shapdf[f"SHAP_{TACOL}"], nbins=50).plot()
# #     ScatterXY(x=shapdf[VPD_COL], y=shapdf[f"SHAP_{VPD_COL}"], nbins=50).plot()
# #
# #     # shapdf['NEE_VUT_50'].plot()
# #     # shapdf['SUM+EXPECTED'].plot()
# #     # shapdf['EXPECTED'].plot()
# #     # plt.legend()
# #     # plt.show()
# #
# #     # means = shapdf.resample('ME').sum()
# #     means = shapdf.groupby(shapdf.index.isocalendar().week).mean()
# #     # means.index = means.index.month
# #     shapcols = [f"SHAP_{f}" for f in FEATURES]
# #     means[shapcols].plot.bar(stacked=True, figsize=(14, 5))
# #     # means['SUM+EXPECTED'].plot.bar()
# #     # means['EXPECTED'].plot.bar()
# #     # means['SUM_SHAP'].plot()
# #     # plt.legend()
# #     plt.show()
# #
# #     # means.plot()
# #     # means.plot.bar(stacked=True, figsize=(20, 5))
# #     # means.plot.bar(stacked=True, subplots=True)
# #     # plt.show()
# #     # # visualize the first prediction's explanation
# #     # fig = plt.figure()
# #     # # shap.plots.waterfall(shap_values[666], show=False)
# #     # shap.plots.waterfall(shap_values[0])
# #     # plt.savefig(f'my_plot.png', bbox_inches='tight')
# #
# #     # shap.plots.initjs()
# #
# #     # https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/decision_plot.html
# #     # features_display = X_display.loc[features.index]
# #     # shap.decision_plot(expected_value, _shap_values)
# #     # shap.decision_plot(expected_value, _shap_values, ignore_warnings=True)
# #
# #     # # visualize the first prediction's explanation with a force plot
# #     # # shap.plots.force(shap_values[0], show=False, matplotlib=False)
# #     # fig = plt.figure()
# #     # shap.plots.force(shap_values[0], show=False, matplotlib=True)
# #     # plt.savefig(f'my_plot.png', bbox_inches='tight')
# #
# #     # # visualize all the training set predictions
# #     # fig = plt.figure()
# #     # shap.plots.force(shap_values[:500], show=False, matplotlib=True)
# #     # plt.savefig(f'my_plot.png', bbox_inches='tight')
# #
# #     # shap.plots.heatmap(shap_values[:1000])
# #     # shap.plots.scatter(shap_values[:, "TA_F"])
# #     # shap.plots.scatter(shap_values[:, "PPFD_IN"], color=shap_values)
# #     # shap.plots.scatter(shap_values[:, "VPD_F"], color=shap_values[:, "SW_IN_F"])
# #     # shap.plots.scatter(shap_values[:, "SWC_F_MDS_1"], color=shap_values)
# #
# #     # create a dependence scatter plot to show the effect of a single feature across the whole dataset
# #     # shap.plots.scatter(shap_values[:, "LE_F_MDS"], color=shap_values[:, "VPD_F"])
# #     # shap.plots.scatter(shap_values[:, "VPD_F"], color=shap_values[:, "LE_F_MDS"])
# #     # shap.plots.scatter(shap_values[:, "TA_F"], color=shap_values[:, "VPD_F"])
# #     # shap.plots.scatter(shap_values[:, "VPD_F"], color=shap_values[:, "TA_F"])
# #     # shap.plots.scatter(shap_values[:, "SW_IN_F"], color=shap_values[:, "TA_F"])
# #     # shap.plots.scatter(shap_values[:, SWC_COL], color=shap_values[:, "TA_F"])
# #
# #     # # summarize the effects of all the features
# #     # shap.plots.beeswarm(shap_values, plot_size=(16, 9))
# #
# #     # shap.plots.bar(shap_values)
# #     print("X")

if __name__ == '__main__':
    sitelist = pd.read_csv("OUT/11.2-sitelist.csv")
    FLUXCOL = 'GPP_NT_VUT_REF'
    # FLUXCOL = 'NEE_VUT_REF'
    FLUXQCCOL = 'NEE_VUT_REF_QC'
    TACOL = 'TA_F'
    SW_IN_COL = 'SW_IN_F'
    VPD_COL = 'VPD_F'
    SWC_COL = 'SWC_F_MDS_1'
    sa = ShapAnalysis(siteinfo=sitelist, targetcol=FLUXCOL)
    sa.run()
