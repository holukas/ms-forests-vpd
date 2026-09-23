"""
Supplementary Fig. 6: the VPD response of NEP, GPP and ET on one axis, with RECO as a control.

Each curve is the all-sites output of script 42 for its flux, read and fitted as in script
54, so the NEP curve is the Figure 4 curve. Only NEP is measured; the legend gives the
source of the other fluxes. SHOW_RECO = False leaves out RECO. Fluxes in NO_THRESHOLD get
an open marker and no interval. There is no split by forest type, because the per-biome ET
fits are not constrained. The top axis converts sigma to kPa with the mean over sites of
the site VPD means and standard deviations, as script 54 does.

Reads the script 42 output of each flux on the TA by VPD grid and the stage 21 subsets
table. Writes to the NEP plot folder, with the fluxes shown in the file name:
- 58_SUPPFIG-6_VpdResponseByFlux_NEP+GPP+RECO+ET.png
- the same name with _CURVES.csv (fitted curve and band per flux) and _THRESHOLDS.csv
  (crossing per flux in sigma and kPa)
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

# RECO on the same axis as the control, see the docstring. False leaves it out.
SHOW_RECO = True

# Order is the order of the legend and of the threshold labels.
FLUXES = ['NEP_ZSCORE', 'GPP_ZSCORE', 'RECO_ZSCORE', 'ET_ZSCORE']
if not SHOW_RECO:
    FLUXES = [f for f in FLUXES if f != 'RECO_ZSCORE']
NAMES = {'NEP_ZSCORE': 'NEP', 'GPP_ZSCORE': 'GPP', 'RECO_ZSCORE': 'RECO', 'ET_ZSCORE': 'ET'}
# Where each flux comes from. Only NEP is a measurement, and the legend says so.
SOURCE = {'NEP_ZSCORE': 'measured', 'GPP_ZSCORE': 'partitioned',
          'RECO_ZSCORE': 'partitioned', 'ET_ZSCORE': 'from gap-filled LE'}
# Okabe and Ito, distinguishable for every common color vision deficiency.
COLORS = {'NEP_ZSCORE': '#000000', 'GPP_ZSCORE': '#009E73',
          'RECO_ZSCORE': '#D55E00', 'ET_ZSCORE': '#0072B2'}
# Fluxes whose crossing interval is not resolved. Drawn with an open marker and no band.
NO_THRESHOLD = {'RECO_ZSCORE'}
# Threshold labels sit in a band above the curves, each with a dotted guide down to its
# crossing, because the curves are steep where they cross and a label at the marker prints
# on top of them. NEP and GPP cross 0.02 sigma apart, so their labels stack.
LABEL_ROW = {'NEP_ZSCORE': 0, 'GPP_ZSCORE': 1, 'RECO_ZSCORE': 2, 'ET_ZSCORE': 0}
# Labels that end at their guide rather than sitting centered on it, so that the two
# labels near 0 sigma do not collide with the NEP label above them.
LABEL_HA = {'GPP_ZSCORE': 'right', 'RECO_ZSCORE': 'right'}

AX_LABELS_FONTSIZE = 12

xvar, yvar, zvar = 'BIN_VPD_ZSCORE', 'VPD_ZSCORE_SHAPVALS', 'TA_ZSCORE'
aggfunc = 'mean'
shap_type = 'conditional' if CONDITIONAL else 'interventional'
settings = load_settings()
agg_base = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG'])
dir_out = Path(settings['DIR_PLOTS_OUT']) / 'NEP_ZSCORE' / shap_type / VARIANT / SITE_SUBSET
dir_out.mkdir(parents=True, exist_ok=True)
stem = dir_out / f"58_SUPPFIG-6_VpdResponseByFlux_{'+'.join(NAMES[f] for f in FLUXES)}"

# Sigma to kPa, the mapping of script 54. CD-Ygb records VPD in Pa, every other site in hPa,
# so its two statistics are divided by 100 before the mean over sites is taken.
PA_UNIT_SITES = ['CD-Ygb']
PA_TO_HPA = 100
_sites = pd.read_csv(Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
                     / "21_SUBSETS_parquet_vars_stats_subsets.csv")
_converted = _sites['SITE'].isin(PA_UNIT_SITES)
_sites.loc[_converted, ['VPD_Z0', 'VPD_SD']] /= PA_TO_HPA
KPA_MEAN = _sites['VPD_Z0'].mean() / 10
KPA_SD = _sites['VPD_SD'].mean() / 10


def sigma_to_kpa(z):
    return KPA_MEAN + z * KPA_SD


def kpa_to_sigma(k):
    return (k - KPA_MEAN) / KPA_SD


def load_curve(flux):
    """The all-sites cells Figure 4 fits, for one flux: VPD bin, VPD SHAP, site count."""
    dir_res = agg_base / flux / shap_type / VARIANT / SITE_SUBSET
    _, subsetdf, _, n_sites = files.load_data(
        suffix='Sites', shap_type=shap_type, dir_res=dir_res, flux=flux, aggfunc=aggfunc,
        subsetcols=[(xvar, 'median'), (yvar, 'median'), (zvar, aggfunc), (yvar, 'sem')],
        count_vals_col=(zvar, 'count'), site_filter=None,
        x_in_filename='BIN-TA_ZSCORE', y_in_filename='BIN-VPD_ZSCORE')
    return subsetdf.iloc[:, 0].to_numpy(float), subsetdf.iloc[:, 1].to_numpy(float), n_sites


def style_axes(ax, xlabel=True):
    ax.axhline(0, color='#555555', lw=1.0, zorder=2)
    if xlabel:
        ax.set_xlabel(r'VPD ($\sigma$)', fontsize=AX_LABELS_FONTSIZE)
    ax.tick_params(axis='both', labelsize=AX_LABELS_FONTSIZE, length=5, width=1)
    ax.spines['right'].set_visible(True)
    ax.tick_params(axis='y', right=False, labelright=False)
    top = ax.secondary_xaxis('top', functions=(sigma_to_kpa, kpa_to_sigma))
    top.set_xlabel('VPD (kPa, mean over sites)', fontsize=AX_LABELS_FONTSIZE * 0.9, color='#555555')
    top.tick_params(axis='x', labelsize=AX_LABELS_FONTSIZE * 0.9, length=4, width=1, colors='#555555')
    top.spines['top'].set_color('#555555')


fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=150)
fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.10)

curves, thresholds, marked_thresholds = [], [], []
control_threshold = None
for flux in FLUXES:
    x, y, n_sites = load_curve(flux)
    poly_func, _, x_fit, y_fit, r2, pi_upper, pi_lower = fit.fit_polynomial(x, y)
    thr, thr_lo, thr_hi = fit.calc_threshold(x_fit, y_fit, pi_lower, pi_upper)
    name, colour = NAMES[flux], COLORS[flux]
    control = flux in NO_THRESHOLD
    note = ''

    ax.fill_between(x_fit, pi_lower, pi_upper, color=colour, alpha=0.10, lw=0, zorder=1)
    ax.plot(x_fit, y_fit, color=colour, lw=2.6, zorder=3, alpha=0.75 if control else 1.0,
            label=f"{name}, {SOURCE[flux]}  (n = {n_sites}, fit R² = {r2:.2f}{note})")

    marked = not control and np.isfinite(thr)
    if marked:
        ax.scatter([thr], [0], s=64, color=colour, edgecolor='white', linewidth=1.2, zorder=5)
        marked_thresholds.append((flux, thr))
    elif np.isfinite(thr):
        # The central curve does cross, so the crossing is drawn as an open marker. Its
        # interval is open on one side, so no bar is drawn; the legend says it is not resolved.
        ax.scatter([thr], [0], s=64, facecolor='white', edgecolor=colour, linewidth=1.6, zorder=5)
        marked_thresholds.append((flux, thr))
        control_threshold = (flux, thr, thr_lo, thr_hi)

    curves.append(pd.DataFrame({'flux': name, 'vpd_z': x_fit, 'vpd_kpa_network': sigma_to_kpa(x_fit),
                                'effect_z': y_fit, 'band_lower': pi_lower, 'band_upper': pi_upper}))
    thresholds.append({'flux': name, 'source': SOURCE[flux], 'n_sites': n_sites, 'fit_r2': round(r2, 3),
                       'threshold_z': thr, 'lower_z': thr_lo, 'upper_z': thr_hi,
                       'threshold_kpa_network': sigma_to_kpa(thr) if np.isfinite(thr) else np.nan,
                       'marked_in_figure': marked or np.isfinite(thr),
                       'note': '' if not control else 'interval open on one side, crossing not resolved'})

# Label band above the curves, guides down to the crossings.
ymin, ymax = ax.get_ylim()
span = ymax - ymin
# The extra 0.3 of span above the curves holds the labels clear of the kPa axis on top.
ax.set_ylim(ymin, ymax + 0.36 * span)
for flux, thr in marked_thresholds:
    y_text = ymax + (0.14 - 0.07 * LABEL_ROW[flux]) * span
    ax.plot([thr, thr], [0, y_text], ls=(0, (1, 3)), color=COLORS[flux], lw=1.2, zorder=2)
    ha = LABEL_HA.get(flux, 'center')
    ax.text(thr + {'left': 0.04, 'right': -0.04}.get(ha, 0), y_text, f"{NAMES[flux]} {thr:.2f}$\sigma$", color=COLORS[flux], ha=ha,
            va='bottom', fontsize=AX_LABELS_FONTSIZE * 0.9, fontweight='bold', zorder=6,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5))

# Zone wording names the attribution, not a cause, and stays neutral on the sign, since a
# positive value is uptake for two fluxes and loss for two.
ax.text(0.015, 0.97, 'Flux increase', transform=ax.transAxes, ha='left', va='top',
        color='#777777', fontsize=AX_LABELS_FONTSIZE * 0.85)
ax.text(0.015, 0.03, 'Flux decrease', transform=ax.transAxes, ha='left', va='bottom',
        color='#777777', fontsize=AX_LABELS_FONTSIZE * 0.85)

style_axes(ax)
ax.set_ylabel(r'VPD effect on daytime flux ($\sigma$)', fontsize=AX_LABELS_FONTSIZE)
ax.legend(loc='lower left', bbox_to_anchor=(0.015, 0.11), frameon=False,
          fontsize=AX_LABELS_FONTSIZE * 0.85, handlelength=2.2)

fig.savefig(f'{stem}.png', dpi=300, facecolor='white')
pd.concat(curves, ignore_index=True).to_csv(f'{stem}_CURVES.csv', index=False)
thresholds = pd.DataFrame(thresholds)
thresholds.to_csv(f'{stem}_THRESHOLDS.csv', index=False)

print()
print(thresholds.round(3).to_string(index=False))
print(f"\nkPa axis: {KPA_MEAN:.3f} + z * {KPA_SD:.3f}, mean over sites of the site statistics")
print(f"\nSaved {stem}.png, _CURVES.csv and _THRESHOLDS.csv")

if SHOW_PLOT:
    plt.show()
