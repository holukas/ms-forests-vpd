"""

Example:
https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Front%20page%20example%20%28XGBoost%29.html
https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html

"""
import statistics

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from diive.core.io.files import load_parquet
from diive.pkgs.fits.fitter import BinFitterCP

pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
pd.set_option('display.width', 1500)
from scipy.stats import zscore


class ShapAnalysis:

    def __init__(self, siteinfo: pd.DataFrame, targetcol: str, featurecols: list):
        self.siteinfo = siteinfo
        self.targetcol = targetcol
        self.featurecols = featurecols

        # todo testing
        # ix = 24
        # self.siteinfo = self.siteinfo.iloc[ix:ix + 4]

        self.sitelist = siteinfo['SITE'].tolist()

        self.test_xcols = []

    def run(self):
        print(self.sitelist)
        self.siteinfo.apply(self._run_site, axis=1)
        print(self.test_xcols)

    def _load_filter_data(self, row) -> pd.DataFrame:
        filepath = row['.filepath']
        # filepath = str(filepath).replace('L:', 'F:')
        df = load_parquet(filepath)
        # [print(v) for v in df.columns if "LE" in v]
        df = df.loc[df['SW_IN_POT'] > 20].copy()
        subsetcols = self.featurecols.copy()
        subsetcols.append(self.targetcol)
        subset = df[subsetcols].copy()
        subset = subset.dropna()
        # todo testing:
        subset = subset.iloc[0:5000].copy()
        return subset

    @staticmethod
    def _calc_shap(X, y):
        model = xgb.XGBRegressor().fit(X, y)
        # explain the model's predictions using SHAP
        # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
        explainer = shap.TreeExplainer(model)
        shap_arrays = explainer(X)
        expected_value = explainer.expected_value
        shap_values = shap_arrays.values
        return shap_values, expected_value

    @staticmethod
    def _collect_shap(X, shap_values, expected_value, subset) -> pd.DataFrame:
        shapdf = X.copy()
        shapcolnames = [f"SHAP_{c}" for c in X.columns]
        _tempdf = pd.DataFrame(data=shap_values, index=X.index, columns=shapcolnames)
        shapdf = pd.concat([shapdf, _tempdf], axis=1)
        shapdf.index = pd.to_datetime(shapdf.index)
        shapdf['SUM_SHAP'] = shapdf[shapcolnames].sum(axis=1)
        shapdf[FLUXCOL] = subset[FLUXCOL].copy()
        shapdf['EXPECTED'] = expected_value
        shapdf['SUM+EXPECTED'] = shapdf['EXPECTED'].add(shapdf['SUM_SHAP'])
        return shapdf

    def _bins_shap(self, xcol, shapdf, site, showplot: bool = True):
        y = f"SHAP_{xcol}"
        test = shapdf[[xcol, y]].copy()
        test = test.sort_values(by=f"{xcol}")
        test = test.reset_index(drop=True)

        bf = BinFitterCP(
            df=test,
            xcol=xcol,
            ycol=y,
            n_predictions=1000,
            n_bins_x=0,
            bins_y_agg='mean',
            fit_type='cubic'  # 'linear', 'quadratic_offset', 'quadratic', 'cubic'
        )
        bf.run()
        fit_results = bf.get_results()

        if showplot:
            bf.showplot(
                show_unbinned_data=True,
                show_bin_variation=False,
                showfit=True,
                title=f"{site}  {self.targetcol}"
                # xlim=(0, 30),
                # ylim=(-1, 0)
            )
        return fit_results

    def _detect_zerocrossing_y(self, x: pd.Series, y: pd.Series, thres_y_sign_change: str):
        # kudos: https://stackoverflow.com/questions/28766692/intersection-of-two-graphs-in-python-find-the-x-value

        n_zerocrossings = None
        zerocrossings_ix = None

        # Collect predicted vals in df
        zerocrossings_df = pd.DataFrame()
        zerocrossings_df['x_col'] = x.copy()
        zerocrossings_df['y_nom'] = y.copy()

        # Check values above/below zero
        _signs = np.sign(zerocrossings_df['y_nom'])
        _signs_max = _signs.max()
        _signs_min = _signs.min()

        if _signs_max == _signs_min:
            print("y does not cross zero-line.")
        else:
            zerocrossings_ix = np.argwhere(np.diff(_signs)).flatten()
            # n_zerocrossings = len(zerocrossings_ix)

        # Keep crossings
        valid_crossings_ix = []
        for i in zerocrossings_ix:
            val_before_crossing = zerocrossings_df.iloc[i]
            val_after_crossing = zerocrossings_df.iloc[i + 1]
            if thres_y_sign_change == '-':
                if val_after_crossing['y_nom'] < val_before_crossing['y_nom']:
                    valid_crossings_ix.append(i)
            elif thres_y_sign_change == '+':
                if val_after_crossing['y_nom'] > val_before_crossing['y_nom']:
                    valid_crossings_ix.append(i)

        n_zerocrossings = len(valid_crossings_ix)

        # There must be one single line crossing to accept result
        # If there is more than one line crossing, reject result
        if n_zerocrossings == 1:
            valid_crossings_ix = valid_crossings_ix[0]
        else:
            raise ("More than 1 zero-crossing detected. "
                   "There must be one single line crossing to accept result. "
                   "Stopping.")

        # Values at zero crossing needed
        # Found index is last element *before* zero-crossing, therefore + 1
        zerocrossing_vals = zerocrossings_df.iloc[valid_crossings_ix + 1]
        zerocrossing_vals = zerocrossing_vals.to_dict()

        # Value for y at zero crossing
        # i.e. reject if NEE after zero-crossing does not change to emission

        # # Check if the sign after the crossing is indeed negative as expected
        # if (thres_y_sign_change == '-') & (zerocrossing_vals['y_nom'] < 0):
        #     pass
        # # Check if the sign after the crossing is indeed positive as expected
        # elif (thres_y_sign_change == '+') & (zerocrossing_vals['y_nom'] > 0):
        #     pass
        # # If the sign after the crossing is positive against expectations, return None
        # elif (thres_y_sign_change == '-') & (zerocrossing_vals['y_nom'] > 0):
        #     return None
        # # If the sign after the crossing is negative against expectations, return None
        # elif (thres_y_sign_change == '+') & (zerocrossing_vals['y_nom'] < 0):
        #     return None

        # # x value must be above threshold to be somewhat meaningful, otherwise reject result
        # if (zerocrossing_vals['x_col'] < self.thres_min_x):
        #     # x is too low, must be at least 1 kPa for valid crossing
        #     return None

        return zerocrossing_vals

    def _run_site(self, row):
        # Make subset
        subset = self._load_filter_data(row)

        subset = subset.apply(zscore)

        # Set data
        X = subset[self.featurecols].copy()
        y = subset[self.targetcol].copy()

        # Calculate SHAP values
        shap_values, expected_value = self._calc_shap(X=X, y=y)

        # Collect SHAP data in dataframe
        shapdf = self._collect_shap(
            X=X,
            shap_values=shap_values,
            expected_value=expected_value,
            subset=subset)

        # ScatterXY(x=shapdf[TACOL], y=shapdf[f"SHAP_{TACOL}"], nbins=20, binagg='mean').plot()
        # ScatterXY(x=shapdf[VPD_COL], y=shapdf[f"SHAP_{VPD_COL}"], nbins=20, binagg='mean').plot()

        # todo testing:
        fit_results = self._bins_shap(xcol='VPD_F', shapdf=shapdf, site=row['SITE'],
                                      showplot=True)
        # for c in self.featurecols:
        #     self._bins_shap(xcol=c, shapdf=shapdf, site=row['SITE'])

        zerocrossing_vals = self._detect_zerocrossing_y(x=fit_results['fit_df']['fit_x'],
                                                        y=fit_results['fit_df']['nom'],
                                                        thres_y_sign_change='-')

        # print(zerocrossing_vals)
        self.test_xcols.append(zerocrossing_vals['x_col'])
        mean = statistics.mean(self.test_xcols)
        print(f"Mean: {mean}    added {zerocrossing_vals['x_col']}, now {len(self.test_xcols)} crossings")

        # from diive.core.plotting.heatmap_datetime import HeatmapDateTime
        # HeatmapDateTime(series=shapdf[y]).show()


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
    # FLUXCOL = 'GPP_DT_VUT_REF'
    FLUXCOL = 'GPP_NT_VUT_REF'
    # FLUXCOL = 'NEE_VUT_REF'
    # FLUXCOL = 'RECO_DT_VUT_REF'
    # FLUXCOL = 'RECO_NT_VUT_REF'
    FLUXQCCOL = 'NEE_VUT_REF_QC'
    TACOL = 'TA_F'
    SW_IN_COL = 'SW_IN_F'
    VPD_COL = 'VPD_F'
    SWC_COL = 'SWC_F_MDS_1'
    # LE_COL = 'LE_F_MDS'
    FEATURECOLS = [TACOL, SW_IN_COL, VPD_COL, SWC_COL]
    sa = ShapAnalysis(siteinfo=sitelist, targetcol=FLUXCOL, featurecols=FEATURECOLS)
    sa.run()
