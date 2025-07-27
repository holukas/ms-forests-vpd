import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from diive.core.io.files import load_parquet
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import src.files as files
import diive as dv

# Load site info
siteinfo_df = files.load_siteinfo(filename="03_siteinfo.csv")

tacol = 'TA_F'
taqc = 'TA_F_QC'
vpdcol = 'VPD_F'
vpdqc = 'VPD_F_QC'
preccol = 'P_F'
swccol = 'SWC_F_MDS_1'
# fluxcol = 'GPP_NT_VUT_50'
# fluxcol = 'LE_F_MDS'
fluxcol = 'NEE_VUT_50'
fluxqc = "NEE_VUT_50_QC"
swinpotcol = "SW_IN_POT"  # For daytime/nighttime
swincol = "SW_IN_F"

df_all = None
_df = siteinfo_df.copy()
for ix, row in _df.iterrows():

    # if ix != 0:
    #     continue

    print(f"\nLoading data for site {row['SITE']} ...")

    # Load site data
    filepath = row['_FILEPATH_PARQUET']
    sitedata = load_parquet(filepath)
    # [print(c) for c in sitedata.columns if "LE" in c];

    # Detect 6 warmest months
    ta = sitedata[[tacol]].copy()
    ta['MONTH'] = ta.index.month
    monthly_avg = ta.groupby('MONTH').mean()
    monthly_avg = monthly_avg.sort_values(by=tacol, ascending=False, inplace=False)
    warmest6 = monthly_avg.head(6).index.to_list()
    # locs = sitedata.index.month.isin(warmest6)

    # Make subset and convert to z-scores
    subset = sitedata[[fluxcol, tacol, vpdcol, swccol, swincol]].copy()

    # Convert to z-scores, ignoring NaNs
    # subset[fluxcol] = subset[fluxcol].apply(lambda x: zscore(x, nan_policy='omit'))
    subset[fluxcol] = subset[fluxcol].transform(lambda x: (x - x.mean()) / x.std())
    # subset = subset.apply(lambda x: zscore(x, nan_policy='omit'))

    # Add original flux quality flag and potential radiation
    subset[fluxqc] = sitedata[fluxqc].copy()
    subset[swinpotcol] = sitedata[swinpotcol].copy()

    # Filter subset
    # Keep data from 6 warmest months
    # Keep directly measured fluxes, no gap-filled data
    # Keep daytime records
    # Keep required variables only
    # Keep records where all vars available
    subset = subset.loc[
        subset.index.month.isin(warmest6) &
        (subset[fluxqc] == 0) &
        (subset[swinpotcol] > 20),
        [fluxcol, tacol, vpdcol, swincol, swccol]
    ].copy()
    subset = subset.dropna()
    print(f"Records: {len(subset)}")

    # TODO testing: Limit time range
    # subset = subset.loc[subset.index.year == 2019].copy()
    # subset = subset.loc[subset.index.month == 7].copy()

    # Target and features
    # features = [tacol, vpdcol, swincol]
    features = [tacol, vpdcol, swccol, swincol]
    X = subset[features]
    y = subset[fluxcol]

    # Initialize XGBoost Regressor
    # objective='reg:squarederror' for standard regression
    # n_estimators: number of boosting rounds (trees)
    # learning_rate: step size shrinkage to prevent overfitting
    # max_depth: maximum depth of a tree
    # Initialize and train the XGBoost Regressor model
    model = xgb.XGBRegressor(objective='reg:squarederror',  # For regression tasks
                             n_estimators=1000,  # Number of boosting rounds
                             learning_rate=0.5,  # Step size shrinkage to prevent overfitting
                             max_depth=6,  # Maximum depth of a tree
                             subsample=.9,  # Subsample ratio of the training instance
                             colsample_bytree=.9,  # Subsample ratio of columns when constructing each tree
                             random_state=42,
                             early_stopping_rounds=50,  # Stop if validation metric doesn't improve for 50 rounds
                             n_jobs=-1)  # Use all available CPU cores

    # Train the model
    model.fit(X, y, eval_set=[(X, y)], verbose=False)

    # Evaluate model performance on the test set
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"R2: {model.score(X, y):.4f}")
    print(f"RMSE: {rmse:.2f}")

    # Create a SHAP TreeExplainer for the trained XGBoost model
    explainer = shap.TreeExplainer(model)
    print("SHAP Explainer created.")

    # Get SHAP values
    shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value

    # Collect SHAP values in dataframe
    shapcols = [f'{c}_SHAPVALS' for c in X]
    shapdf = pd.DataFrame(data=shap_values, index=X.index, columns=shapcols)
    shapdf.index = pd.to_datetime(shapdf.index)
    shapdf['SUM'] = shapdf.sum(axis=1)
    shapdf['EXPECTED'] = expected_value
    shapdf['SUM+EXPECTED'] = shapdf['EXPECTED'].add(shapdf['SUM'])
    shapdf[fluxcol] = y.copy()
    # print(shapdf)

    merged = pd.concat([X, shapdf], axis=1)
    # print(merged)

    # from diive.core.plotting.scatter import ScatterXY
    # ScatterXY(x=merged['VPD_F'], y=merged['VPD_F_SHAPVALS'], nbins=200,
    #           xlim=[-1, 10], binagg='mean').plot()

    # xvar = vpdcol
    # yvar = tacol
    # zvar = fluxcol
    # plotdf = shapdf[[xvar, yvar, zvar]].copy()

    q = dv.ga(
        x=merged['TA_F'],
        y=merged['VPD_F'],
        z=merged['VPD_F_SHAPVALS'],
        # binning_type='custom',
        # custom_x_bins=list(np.arange(-8, 10, .5)),
        # custom_y_bins=list(np.arange(-8, 10, .5)),
        binning_type='quantiles',
        n_bins=20,
        min_n_vals_per_bin=1,
        aggfunc='mean'
    )
    print(q.df_agg_wide)

    hm = dv.heatmapxyz(
        x=q.df_agg_long[f'BIN_TA_F'],
        y=q.df_agg_long[f'BIN_VPD_F'],
        z=q.df_agg_long['VPD_F_SHAPVALS'],
        cb_digits_after_comma=0,
        xlabel=r'x',
        ylabel=r'y',
        zlabel=r'z',
        # vmin=-3,
        # vmax=3
    )
    hm.show()

    if ix == 0:
        df_all = q.df_agg_long.copy()
    else:
        df_all = pd.concat([df_all, q.df_agg_long], axis=0)

df_all['BIN_COMBINED_STR'] = (df_all['BIN_TA_F'].astype(str) + "+" + df_all['BIN_VPD_F'].astype(str))
df_all.groupby('BIN_COMBINED_STR').mean()

hm = dv.heatmapxyz(
    x=df_all[f'BIN_TA_F'],
    y=df_all[f'BIN_VPD_F'],
    z=df_all['VPD_F_SHAPVALS'],
    cb_digits_after_comma=0,
    xlabel=r'x',
    ylabel=r'y',
    zlabel=r'z',
    # vmin=-3,
    # vmax=3
)
hm.show()
