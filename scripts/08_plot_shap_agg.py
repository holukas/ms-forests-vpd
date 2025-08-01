"""
Flame plot.
"""
from pathlib import Path

import diive as dv

import src.files as files
from common import get_variable_names


settings = files.read_settings_file("../config/settings.yaml")

swincol = 'SW_IN_F'
tacol = 'TA_F'
vpdcol = 'VPD_F'

x = tacol
y = vpdcol
z = f"{vpdcol}_SHAPVALS"

binx = f"BIN_{x}"
biny = f"BIN_{y}"
aggfunc = 'median'

filepath = Path(settings['DIR_DATA_OUT_SHAPVALS_SITE']) / "2_ALLSITES_shap_values_median.parquet"
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)

# import pandas as pd
# pivot_df_median = pd.pivot_table(
#     shapvals_df,
#     values='VPD_F_SHAPVALS',
#     index='BIN_TA_F',    # Rows will be indexed by BIN_TA_F
#     columns='BIN_VPD_F',   # Columns will be taken from BIN_VPD_F
#     aggfunc='count'       # Aggregate with the median
# )

keeplocs = shapvals_df['VPD_F_SHAPVALS_COUNTS'] > 30
shapvals_df = shapvals_df[keeplocs].copy()

hm = dv.heatmapxyz(
    title="All sites",
    x=shapvals_df[binx],
    y=shapvals_df[biny],
    z=shapvals_df[z],
    # z=shapvals_df['VPD_F_SHAPVALS_COUNTS'],
    cb_digits_after_comma=1,
    xlabel=f'{binx} (z-score)',
    ylabel=f'{biny} (z-score)',
    zlabel=f'{aggfunc} {z} (z-score)',
    # vmin=-3,
    # vmax=3
)
hm.show()
