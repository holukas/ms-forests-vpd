from pathlib import Path

import diive as dv

import src.files as files
from src.aggregation import aggregate_shap_values_across_sites

# ------------------------------
# Variables
# NEP, NEE, LE, GPP, RECO, TA, VPD, SWIN, SWC
FLUX = 'NEP'  # Go to FLUX folder
# FLUX = 'GPP'  # Go to FLUX folder
# FLUX = 'RECO'  # Go to FLUX folder
# FLUX = 'LE'  # Go to FLUX folder
xvar = 'TA'
yvar = 'SWC'
aggfunc = 'median'
CONDITIONAL = True  # SHAP
# ------------------------------

binx = f"BIN_{xvar}"
biny = f"BIN_{yvar}"

# Load settings
settings = files.read_settings_file("../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type
filepath = Path(results_outdir) / f"2_PerSite_Aggregated_SHAPValues-{shap_type}_BIN-{xvar}_BIN-{yvar}_{FLUX}.parquet"
shapvals_sites_agg_long_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
igbps = shapvals_sites_agg_long_df['IGBP'].unique()
print(f"TOTAL: {shapvals_sites_agg_long_df['SITE'].nunique()} sites")

found_n = []
for i in igbps:
    subset = shapvals_sites_agg_long_df.loc[shapvals_sites_agg_long_df['IGBP'] == i].copy()
    n_sites = subset['SITE'].nunique()
    found_n.append(n_sites)
    print(f"{i}: {n_sites} sites")



# # Save to Parquet
# outfilepath = dv.save_parquet(
#     filename=f"4_All-{i}_Aggregated_SHAPValues-{shap_type}_BIN-{xvar}_BIN-{yvar}_{FLUX}",
#     data=subset_agg_df,
#     outpath=results_outdir)
# # print(f"Saved SHAP values across all files as mean to file {outfilepath}.")
# # shapvals_sites_grouped_agg_df.to_csv(outfilepath.replace('.parquet', '.csv'))
