from pathlib import Path

import diive as dv

import src.files as files
from src.aggregation import aggregate_shap_values_for_site

# VARIABLES (site-level)
# ----------------------
# 'ta_var', 'vpd_var', 'swc_var', 'swin_var'
xvar = 'ta_var'
yvar = 'vpd_var'
zvar = 'vpd_var'
aggfunc = 'median'
conditional = False  # SHAP

# Load settings
settings = files.read_settings_file("../config/settings.yaml")

# Load site info
siteinfo_df = files.load_siteinfo(settings)

# Aggregate SHAP values for each site and collect in dataframe
shapvals_sites_agg_long_df = None
for ix, siteconfig in siteinfo_df.iterrows():
    shapvals_sites_agg_long_df = aggregate_shap_values_for_site(
        siteconfig=siteconfig, ix=ix,
        shapvals_sites_agg_long_df=shapvals_sites_agg_long_df,
        xvar=xvar, yvar=yvar, zvar=zvar, aggfunc=aggfunc,
        conditional=conditional
    )
    # # ---todo testing
    # biny = f"BIN_VPD_F"
    # binx = f"BIN_TA_F"
    # z = f"VPD_F_SHAPVALS"
    # if ix == 0:
    #     dummydf = shapvals_sites_agg_long_df.copy()
    # else:
    #     dummydf = pd.concat([dummydf, shapvals_sites_agg_long_df], axis=0)
    # dummydf2 = aggregate_shap_values_across_all_sites(df=dummydf, binx=binx, biny=biny, z=z)
    # hm = dv.heatmapxyz(
    #     title=f"All sites + {siteconfig['SITE']}",
    #     x=dummydf2[binx], y=dummydf2[biny], z=dummydf2[z],
    #     cb_digits_after_comma=1, xlabel=f'{binx} (z-score)', ylabel=f'{biny} (z-score)',
    #     zlabel=f'{aggfunc} {z} (z-score)',
    #     # show_values_n_dec_places=1,
    #     # show_values=True,
    #     # show_values_fontsize=4,
    #     figdpi=300,
    #     # vmin=-3,
    #     # vmax=3
    # )
    # hm.show()
    # # ---todo testing

# Save SHAP values aggregated per site to Parquet and CSV
outfilepath = dv.save_parquet(
    filename=f"2_PerSite_Aggregated_SHAPValues",
    data=shapvals_sites_agg_long_df,
    outpath=Path(settings['DIR_DATA_OUT_SHAPVALS_STANDARD']))
shapvals_sites_agg_long_df.to_csv(outfilepath.replace('.parquet', '.csv'))
