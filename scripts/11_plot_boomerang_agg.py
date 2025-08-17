"""
"""
from pathlib import Path

import diive as dv
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import src.files as files
from diive.pkgs.analyses.decoupling import SortingBinsMethod

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Latin Modern Roman'] + plt.rcParams['font.serif']

# ------------------------------
# Variables
# NEP, NEE, LE, GPP, RECO, TA, VPD, SWIN, SWC
FLUX = 'NEP'
xvar = 'TA'
yvar = 'VPD_SHAPVALS'
zvar = 'SWC'  # Classes
aggfunc = 'median'
CONDITIONAL = True  # SHAP

# Heatmap settings
n_sites_min = 20
# ------------------------------

# Load settings
settings = files.read_settings_file("../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type
filepath = Path(results_outdir) / f"2_PerSite_Aggregated_SHAPValues-{shap_type}_{FLUX}.parquet"
shapvals_sites_agg_long_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)

subset = shapvals_sites_agg_long_df[[xvar, yvar, zvar]].copy()
subset = subset.dropna()

sbm = SortingBinsMethod(df=subset,
                        zvar=zvar,
                        xvar=xvar,
                        yvar=yvar,
                        n_bins_z=100,
                        n_bins_x=2,
                        conversion=None)
sbm.calcbins()
sbm.showplot_decoupling_sbm(marker='o', emphasize_lines=True)

binaggs = sbm.get_binaggs()
first = next(iter(binaggs))
print(binaggs[first])

