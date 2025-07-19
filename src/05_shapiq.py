import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from diive.core.io.files import load_parquet
import pandas as pd
import numpy as np
import xgboost as xgb
from docs.source.examples.other.mlflow import X_test
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('../OUT/03_siteinfo.csv')
df = df.fillna(np.nan)

tacol = 'TA_F'
taqc = 'TA_F_QC'
vpdcol = 'VPD_F'
vpdqc = 'VPD_F_QC'
preccol = 'P_F'
swccol = 'SWC_F_MDS_1'
# fluxcol = 'GPP_NT_VUT_50'
fluxcol = 'NEE_VUT_50'
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

    df = sitedata_warmest6_qc0_dt[[fluxcol, tacol, vpdcol, swccol, swincol]].copy()
    df = df.dropna()
    df = df.loc[df.index.year == 2019].copy()

    # X, y = shapiq.load_bike_sharing()
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X.values,
    #     y.values,
    #     test_size=0.25,
    #     random_state=42,
    # )
    # n_features = X_train.shape[1]
    # X_train.shape, X_test.shape



    # Define features (X) and target (y)
    features = [tacol, vpdcol, swccol, swincol]
    target = fluxcol

    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")



    # Initialize and train the XGBoost Regressor model
    model = xgb.XGBRegressor(objective='reg:squarederror',  # For regression tasks
                                 n_estimators=1000,  # Number of boosting rounds
                                 learning_rate=0.05,  # Step size shrinkage to prevent overfitting
                                 max_depth=6,  # Maximum depth of a tree
                                 subsample=0.7,  # Subsample ratio of the training instance
                                 colsample_bytree=0.7,  # Subsample ratio of columns when constructing each tree
                                 random_state=42,
                                 early_stopping_rounds=50,  # Stop if validation metric doesn't improve for 50 rounds
                                 n_jobs=-1)  # Use all available CPU cores

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    print(f"Train R2: {model.score(X_train, y_train):.4f}")
    print(f"Test  R2: {model.score(X_test, y_test):.4f}")

    import shapiq
    # shapiq.load_bike_sharing()  # df
    # X, y = shapiq.load_bike_sharing()
    explainer = shapiq.TreeExplainer(model=model, index="k-SII", min_order=3, max_order=3)

    # x = X_test.iloc[121]
    # interaction_values = explainer.explain(x)
    # print(interaction_values)
    # shapiq.network_plot(
    #     interaction_values=interaction_values,
    #     feature_names=X.columns,
    #     show=True
    # )

    X_test_np = X_test.to_numpy()
    X_train_np = X_train.to_numpy()

    list_of_interaction_values = explainer.explain_X(X_test_np[:4, ])
    shapiq.plot.bar_plot(list_of_interaction_values, feature_names=X.columns, max_display=20, show=True)



