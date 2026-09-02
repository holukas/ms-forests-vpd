"""
Figure 5: the VPD response of four fluxes on one axis.

Figure 4 shows the VPD response of NEP. This figure puts the same curve beside the ones for
GPP, RECO and ET, fitted the same way from the same sites, months, hours and model settings,
so a reader sees what the NEP threshold is made of. GPP turns over where NEP does, RECO has
no resolvable turning point, and ET keeps responding to VPD long after carbon uptake has
stopped benefiting. The threshold is therefore a limit on photosynthesis, and water loss
continues past it.

**It computes nothing new.** Each curve is the stage 42 output for its flux, read through
`src.files.load_data` exactly as script 54 reads it for Figure 4, and fitted with
`src.fit.fit_polynomial`, so the NEP curve here is the Figure 4 curve and the thresholds
match the coefficient table script 54 writes.

**RECO is drawn without a threshold marker.** Its fit is too weak to carry a number: R2 about
0.75 over all sites, and the prediction band never crosses zero on one side, so the crossing
of the central curve alone would overstate what the data support. The curve is shown, the
crossing is not marked, and the threshold table written here records it as not resolved.

**All sites only.** The per-biome ET fits return reversed intervals for DBF and MF, meaning the
fit did not constrain them, so this figure does not split by forest type.

**Two things the caption has to say.** The peak season is the four highest-GPP months for
every flux, so RECO and ET are shown in a window defined by carbon uptake rather than by
their own seasonality. And a positive value means more of the flux, which is more uptake for
NEP and GPP but more respiration for RECO and more water loss for ET; the zone labels are
neutral for that reason.

Reads, for each of the four fluxes:
    40_aggregation/{FLUX}/conditional/42_SHAPVALUES-conditional_meanAggregatedAcrossSites
        _BIN-TA_ZSCORE+BIN-VPD_ZSCORE+{FLUX}.parquet

Writes, into the NEP plot folder, since this is a display item of the NEP paper:
    58_FIG-5_VpdResponseByFlux_NEP+GPP+RECO+ET.png
    58_FIG-5_VpdResponseByFlux_NEP+GPP+RECO+ET_CURVES.csv      fitted curve and band per flux
    58_FIG-5_VpdResponseByFlux_NEP+GPP+RECO+ET_THRESHOLDS.csv  crossing per flux, in sigma
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.files as files
import src.fit as fit
from src.paths import load_settings

SHOW_PLOT = False

CONDITIONAL = True
VARIANT = ""
SITE_SUBSET = ""

# Order is the order of the legend and of the threshold labels.
FLUXES = ['NEP_ZSCORE', 'GPP_ZSCORE', 'RECO_ZSCORE', 'ET_ZSCORE']
NAMES = {'NEP_ZSCORE': 'NEP', 'GPP_ZSCORE': 'GPP', 'RECO_ZSCORE': 'RECO', 'ET_ZSCORE': 'ET'}
# Okabe and Ito, distinguishable for every common colour vision deficiency.
COLORS = {'NEP_ZSCORE': '#000000', 'GPP_ZSCORE': '#009E73',
          'RECO_ZSCORE': '#D55E00', 'ET_ZSCORE': '#0072B2'}
# Fluxes whose fit is too weak to carry a crossing. Drawn, not marked.
NO_THRESHOLD = {'RECO_ZSCORE'}
# Threshold labels sit in a band above the curves, each with a dotted guide down to its
# crossing, because the curves are steep where they cross and a label at the marker prints
# on top of them. NEP and GPP cross 0.02 sigma apart, so their labels stack.
LABEL_ROW = {'NEP_ZSCORE': 0, 'GPP_ZSCORE': 1, 'ET_ZSCORE': 0}

AX_LABELS_FONTSIZE = 12

xvar, yvar, zvar = 'BIN_VPD_ZSCORE', 'VPD_ZSCORE_SHAPVALS', 'TA_ZSCORE'
aggfunc = 'mean'
shap_type = 'conditional' if CONDITIONAL else 'interventional'
settings = load_settings()
agg_base = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG'])
dir_out = Path(settings['DIR_PLOTS_OUT']) / 'NEP_ZSCORE' / shap_type / VARIANT / SITE_SUBSET
dir_out.mkdir(parents=True, exist_ok=True)
stem = dir_out / '58_FIG-5_VpdResponseByFlux_NEP+GPP+RECO+ET'


def load_curve(flux):
    """The all-sites cells Figure 4 fits, for one flux: VPD bin, VPD SHAP, site count."""
    dir_res = agg_base / flux / shap_type / VARIANT / SITE_SUBSET
    _, subsetdf, _, n_sites = files.load_data(
        suffix='Sites', shap_type=shap_type, dir_res=dir_res, flux=flux, aggfunc=aggfunc,
        subsetcols=[(xvar, 'median'), (yvar, 'median'), (zvar, aggfunc), (yvar, 'sem')],
        count_vals_col=(zvar, 'count'), site_filter=None,
        x_in_filename='BIN-TA_ZSCORE', y_in_filename='BIN-VPD_ZSCORE')
    return subsetdf.iloc[:, 0].to_numpy(float), subsetdf.iloc[:, 1].to_numpy(float), n_sites


fig, ax = plt.subplots(figsize=(8.5, 5.6), dpi=150)
fig.subplots_adjust(left=0.11, right=0.97, top=0.93, bottom=0.13)

curves, thresholds, marked_thresholds = [], [], []
for flux in FLUXES:
    x, y, n_sites = load_curve(flux)
    poly_func, _, x_fit, y_fit, r2, pi_upper, pi_lower = fit.fit_polynomial(x, y)
    thr, thr_lo, thr_hi = fit.calc_threshold(x_fit, y_fit, pi_lower, pi_upper)
    name, colour = NAMES[flux], COLORS[flux]

    ax.fill_between(x_fit, pi_lower, pi_upper, color=colour, alpha=0.10, lw=0, zorder=1)
    ax.plot(x_fit, y_fit, color=colour, lw=2.6, zorder=3,
            label=f"{name}  (n = {n_sites}, R² = {r2:.2f})")

    marked = flux not in NO_THRESHOLD and np.isfinite(thr)
    if marked:
        ax.scatter([thr], [0], s=64, color=colour, edgecolor='white', linewidth=1.2, zorder=5)
        marked_thresholds.append((flux, thr))

    curves.append(pd.DataFrame({'flux': name, 'vpd_z': x_fit, 'effect_z': y_fit,
                                'band_lower': pi_lower, 'band_upper': pi_upper}))
    thresholds.append({'flux': name, 'n_sites': n_sites, 'fit_r2': round(r2, 3),
                       'threshold_z': thr, 'lower_z': thr_lo, 'upper_z': thr_hi,
                       'marked_in_figure': marked,
                       'note': '' if marked else 'fit too weak to resolve a crossing'})

ax.axhline(0, color='#555555', lw=1.0, zorder=2)

# Label band above the curves, guides down to the crossings.
ymin, ymax = ax.get_ylim()
span = ymax - ymin
ax.set_ylim(ymin, ymax + 0.16 * span)
for flux, thr in marked_thresholds:
    y_text = ymax + (0.12 - 0.065 * LABEL_ROW[flux]) * span
    ax.plot([thr, thr], [0, y_text], ls=(0, (1, 3)), color=COLORS[flux], lw=1.2, zorder=2)
    ax.text(thr, y_text, f"{NAMES[flux]} {thr:.2f}", color=COLORS[flux], ha='center',
            va='bottom', fontsize=AX_LABELS_FONTSIZE * 0.9, fontweight='bold', zorder=6,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5))

# Neutral zone wording, since a positive value is uptake for two fluxes and loss for two.
ax.text(0.015, 0.97, 'VPD raises the flux', transform=ax.transAxes, ha='left', va='top',
        color='#777777', fontsize=AX_LABELS_FONTSIZE * 0.85)
ax.text(0.015, 0.03, 'VPD lowers the flux', transform=ax.transAxes, ha='left', va='bottom',
        color='#777777', fontsize=AX_LABELS_FONTSIZE * 0.85)

ax.set_xlabel(r'VPD ($\sigma$)', fontsize=AX_LABELS_FONTSIZE)
ax.set_ylabel(r'VPD effect on daytime flux ($\sigma$)', fontsize=AX_LABELS_FONTSIZE)
ax.tick_params(axis='both', labelsize=AX_LABELS_FONTSIZE, length=5, width=1)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
ax.legend(loc='lower left', bbox_to_anchor=(0.015, 0.09), frameon=False,
          fontsize=AX_LABELS_FONTSIZE * 0.9, handlelength=2.2)

fig.savefig(f'{stem}.png', dpi=300, facecolor='white')
pd.concat(curves, ignore_index=True).to_csv(f'{stem}_CURVES.csv', index=False)
thresholds = pd.DataFrame(thresholds)
thresholds.to_csv(f'{stem}_THRESHOLDS.csv', index=False)

print()
print(thresholds.round(3).to_string(index=False))
print(f"\nSaved {stem}.png, _CURVES.csv and _THRESHOLDS.csv")

if SHOW_PLOT:
    plt.show()
