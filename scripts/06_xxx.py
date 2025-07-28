"""
Train XGBoost model for each site and save SHAP values to file.
"""
import diive as dv
import pandas as pd

import src.files as files

# Load settings
settings = files.read_settings_file("../config/settings.yaml")

# Load site info
siteinfo_df = files.load_siteinfo(filename="05_siteinfo.csv")

tacol = 'TA_F'
vpdcol = 'VPD_F'
swccol = 'SWC_F_MDS_1'
fluxcol = 'NEE_VUT_50'
swincol = "SW_IN_F"

x = tacol
y = vpdcol
z = f"{vpdcol}_SHAPVALS"

binx = f"BIN_{x}"
biny = f"BIN_{y}"
aggfunc = 'mean'

df_all = None

for ix, site in siteinfo_df.iterrows():
    print(f"\nLoading data for site #{ix + 1} {site['SITE']} ...")
    filepath = site['_FILEPATH_SHAP_VALUES']

    if filepath == '-MISSING-':
        continue

    shapvals_df = dv.load_parquet(filepath)
    print(shapvals_df)

    q = dv.ga(
        x=shapvals_df[x],
        y=shapvals_df[y],
        z=shapvals_df[z],
        # binning_type='custom',
        # custom_x_bins=list(np.arange(-8, 10, .5)),
        # custom_y_bins=list(np.arange(-8, 10, .5)),
        binning_type='quantiles',
        n_bins=20,
        min_n_vals_per_bin=1,
        aggfunc=aggfunc
    )
    print(q.df_agg_wide)

    hm = dv.heatmapxyz(
        x=q.df_agg_long[binx],
        y=q.df_agg_long[biny],
        z=q.df_agg_long[z],
        title=site['SITE'],
        cb_digits_after_comma=0,
        xlabel=f'{binx} (percentile)',
        ylabel=f'{biny} (percentile)',
        zlabel=f'{aggfunc} {z} (z-score)',
        # vmin=-3,
        # vmax=3
    )
    hm.show()

    if ix == 0:
        df_all = q.df_agg_long.copy()
    else:
        df_all = pd.concat([df_all, q.df_agg_long], axis=0)

df_all['BIN_COMBINED_STR'] = (df_all[binx].astype(str) + "+" + df_all[biny].astype(str))
df_all.groupby('BIN_COMBINED_STR').mean()

print(df_all)

hm = dv.heatmapxyz(
    title="All sites",
    x=df_all[binx],
    y=df_all[biny],
    z=df_all[z],
    cb_digits_after_comma=0,
    xlabel=f'{binx} (percentile)',
    ylabel=f'{biny} (percentile)',
    zlabel=f'{aggfunc} {z} (z-score)',
    # vmin=-3,
    # vmax=3
)
hm.show()
