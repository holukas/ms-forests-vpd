"""
Supplementary figure: the VPD response of NEP, GPP and ET on one axis, with RECO as the control.

Figure 4 shows the VPD response of NEP. This figure puts the same curve beside the ones for
GPP and ET, fitted the same way from the same sites, months, hours and model settings, so a
reader sees what the NEP threshold is made of. GPP turns over where NEP does, and ET keeps
responding to VPD long after carbon uptake has stopped benefiting. The threshold is therefore
a limit on photosynthesis, and water loss continues past it.

**Supplementary, decided 5 September 2026.** Only NEP is measured. GPP and RECO are
partitioned from the night-time NEE with a temperature model, and ET comes from gap-filled LE,
so the figure rests on modelled quantities except for its black curve. A main-text figure
built on them would widen the paper past the measured net flux it is about. The legend says
where each flux comes from, and the caption has to say it again.

**RECO has its own panel, decided 5 September 2026, and its crossing is drawn.** The claim of
panel a is that the NEP threshold is a GPP threshold, and the question a reader asks next is
whether the NEP decline could instead be respiration rising with VPD. Panel b answers it: RECO
does not rise, it falls, gently and monotonically, to about a quarter of the GPP decline at
3 sigma. That is the one thing the panel has to show, and it is why the panel is a control for
the NEP claim without being a null result.

The central curve does cross zero, at -0.18 sigma, so the figure draws that crossing as an open
marker and gives it the interval the band supports, which is open on the low side and ends at
0.96 on the high side. An earlier version wrote "no resolvable crossing" over a curve that
visibly crossed, which was not honest; "not resolved" refers to the interval, and the figure
now shows it. The fit is the weakest of the four, R2 about 0.75.

Why RECO falls with VPD is not a stomatal effect, because respiration has no stomatal pathway.
Two things are in play and this figure cannot separate them. Daytime RECO is not measured: it
is the night-time temperature model extrapolated into the day with a reference respiration
refitted in moving windows, so a conditional VPD effect on it, with temperature held, can only
enter through those windows, and dry spells, which are high-VPD periods, carry lower
respiration because soil respiration falls as soil dries and substrate supply falls with GPP.
The remainder is credit shared with temperature, which correlates with VPD at 0.78. The
substrate pathway is the one real physiological route, less assimilate under high VPD means
less growth and maintenance respiration, and it is lagged, so it is weak at half-hourly
resolution, which is what a gentle decline looks like. Set SHOW_RECO = False for the
three-curve version without the panel.

**The top axis is in kPa.** Each site standardises VPD against its own mean and standard
deviation, so a sigma value is a different absolute VPD at every site. The axis uses the
network mean of the site means and of the site standard deviations, equal weight per site,
which is the same mapping script 54 uses for the kPa column of the coefficient table. It is a
guide for the reader, not a second measurement: the 1.26 kPa in the text is the site-wise mean
of the mapped thresholds, which is what script 54 reports.

**It computes nothing new.** Each curve is the stage 42 output for its flux, read through
`src.files.load_data` exactly as script 54 reads it for Figure 4, and fitted with
`src.fit.fit_polynomial`, so the NEP curve here is the Figure 4 curve and the thresholds
match the coefficient table script 54 writes.

**All sites only.** The per-biome ET fits return reversed intervals for DBF and MF, meaning the
fit did not constrain them, so this figure does not split by forest type.

**Two things the caption has to say.** The peak season is the four highest-GPP months for
every flux, so RECO and ET are shown in a window defined by carbon uptake rather than by
their own seasonality. And a positive value means more of the flux, which is more uptake for
NEP and GPP but more water loss for ET and more release for RECO; the zone labels are
neutral for that reason.

Reads, for each of the four fluxes:
    40_aggregation/{FLUX}/conditional/42_SHAPVALUES-conditional_meanAggregatedAcrossSites
        _BIN-TA_ZSCORE+BIN-VPD_ZSCORE+{FLUX}.parquet
and the stage 21 subsets table for the kPa axis.

Writes, into the NEP plot folder, since this is a display item of the NEP paper. The fluxes
shown are in the file name, so the version without RECO cannot overwrite the one with it.
X stands in until the figure number is assigned:
    58_SUPPFIG-X_VpdResponseByFlux_NEP+GPP+RECO+ET.png
    58_SUPPFIG-X_VpdResponseByFlux_NEP+GPP+RECO+ET_CURVES.csv      fitted curve and band per flux
    58_SUPPFIG-X_VpdResponseByFlux_NEP+GPP+RECO+ET_THRESHOLDS.csv  crossing per flux, sigma and kPa
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

# RECO in its own panel as the negative control, see the docstring. False gives the
# three-curve version on one panel.
SHOW_RECO = True

# Order is the order of the legend and of the threshold labels.
FLUXES = ['NEP_ZSCORE', 'GPP_ZSCORE', 'RECO_ZSCORE', 'ET_ZSCORE']
if not SHOW_RECO:
    FLUXES = [f for f in FLUXES if f != 'RECO_ZSCORE']
NAMES = {'NEP_ZSCORE': 'NEP', 'GPP_ZSCORE': 'GPP', 'RECO_ZSCORE': 'RECO', 'ET_ZSCORE': 'ET'}
# Where each flux comes from. Only NEP is a measurement, and the legend says so.
SOURCE = {'NEP_ZSCORE': 'measured', 'GPP_ZSCORE': 'partitioned',
          'RECO_ZSCORE': 'partitioned', 'ET_ZSCORE': 'from gap-filled LE'}
# Okabe and Ito, distinguishable for every common colour vision deficiency.
COLORS = {'NEP_ZSCORE': '#000000', 'GPP_ZSCORE': '#009E73',
          'RECO_ZSCORE': '#D55E00', 'ET_ZSCORE': '#0072B2'}
# Fluxes whose fit is too weak to carry a crossing. Drawn, not marked, in their own panel.
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
stem = dir_out / f"58_SUPPFIG-X_VpdResponseByFlux_{'+'.join(NAMES[f] for f in FLUXES)}"

# Sigma to kPa, the mapping of script 54. CD-Ygb records VPD in Pa, every other site in hPa,
# so its two statistics are divided by 100 before the network mean is taken.
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
    for side in ('right',):
        ax.spines[side].set_visible(False)
    top = ax.secondary_xaxis('top', functions=(sigma_to_kpa, kpa_to_sigma))
    top.set_xlabel('VPD (kPa, network mean)', fontsize=AX_LABELS_FONTSIZE * 0.9, color='#555555')
    top.tick_params(axis='x', labelsize=AX_LABELS_FONTSIZE * 0.9, length=4, width=1, colors='#555555')
    top.spines['top'].set_color('#555555')


if SHOW_RECO:
    fig, (ax, ax_reco) = plt.subplots(1, 2, figsize=(12.5, 5.6), dpi=150, sharey=True,
                                      gridspec_kw=dict(width_ratios=[1.7, 1], wspace=0.08))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.13)
else:
    fig, ax = plt.subplots(figsize=(8.5, 5.6), dpi=150)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.13)
    ax_reco = None

curves, thresholds, marked_thresholds = [], [], []
control_threshold = None
for flux in FLUXES:
    x, y, n_sites = load_curve(flux)
    poly_func, _, x_fit, y_fit, r2, pi_upper, pi_lower = fit.fit_polynomial(x, y)
    thr, thr_lo, thr_hi = fit.calc_threshold(x_fit, y_fit, pi_lower, pi_upper)
    name, colour = NAMES[flux], COLORS[flux]
    control = flux in NO_THRESHOLD
    panel = ax_reco if (control and ax_reco is not None) else ax

    panel.fill_between(x_fit, pi_lower, pi_upper, color=colour, alpha=0.10, lw=0, zorder=1)
    panel.plot(x_fit, y_fit, color=colour, lw=2.6, zorder=3, alpha=0.75 if control else 1.0,
               label=f"{name}, {SOURCE[flux]}  (n = {n_sites}, fit R² = {r2:.2f})")

    marked = not control and np.isfinite(thr)
    if marked:
        panel.scatter([thr], [0], s=64, color=colour, edgecolor='white', linewidth=1.2, zorder=5)
        marked_thresholds.append((flux, thr))
    elif np.isfinite(thr):
        # The central curve does cross, so the crossing is drawn, as an open marker, with
        # the interval the band gives it. One side of that interval is unbounded, which is
        # what "not resolved" means here, and the figure shows it rather than saying it.
        panel.scatter([thr], [0], s=64, facecolor='white', edgecolor=colour, linewidth=1.6, zorder=5)
        x_left = x_fit.min() if not np.isfinite(thr_lo) else thr_lo
        x_right = x_fit.max() if not np.isfinite(thr_hi) else thr_hi
        panel.plot([x_left, x_right], [0, 0], color=colour, lw=4, alpha=0.3, solid_capstyle='butt', zorder=4)
        control_threshold = (flux, thr, thr_lo, thr_hi)

    curves.append(pd.DataFrame({'flux': name, 'vpd_z': x_fit, 'vpd_kpa_network': sigma_to_kpa(x_fit),
                                'effect_z': y_fit, 'band_lower': pi_lower, 'band_upper': pi_upper}))
    thresholds.append({'flux': name, 'source': SOURCE[flux], 'n_sites': n_sites, 'fit_r2': round(r2, 3),
                       'threshold_z': thr, 'lower_z': thr_lo, 'upper_z': thr_hi,
                       'threshold_kpa_network': sigma_to_kpa(thr) if np.isfinite(thr) else np.nan,
                       'marked_in_figure': marked,
                       'note': '' if marked else 'fit too weak to resolve a crossing'})

# Label band above the curves, guides down to the crossings.
ymin, ymax = ax.get_ylim()
span = ymax - ymin
# The extra 0.3 of span above the curves holds the labels clear of the kPa axis on top.
ax.set_ylim(ymin, ymax + 0.30 * span)
for flux, thr in marked_thresholds:
    y_text = ymax + (0.14 - 0.07 * LABEL_ROW[flux]) * span
    ax.plot([thr, thr], [0, y_text], ls=(0, (1, 3)), color=COLORS[flux], lw=1.2, zorder=2)
    ax.text(thr, y_text, f"{NAMES[flux]} {thr:.2f}", color=COLORS[flux], ha='center',
            va='bottom', fontsize=AX_LABELS_FONTSIZE * 0.9, fontweight='bold', zorder=6,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5))

# Neutral zone wording, since a positive value is uptake for two fluxes and loss for two.
ax.text(0.015, 0.97, 'VPD raises the flux', transform=ax.transAxes, ha='left', va='top',
        color='#777777', fontsize=AX_LABELS_FONTSIZE * 0.85)
ax.text(0.015, 0.03, 'VPD lowers the flux', transform=ax.transAxes, ha='left', va='bottom',
        color='#777777', fontsize=AX_LABELS_FONTSIZE * 0.85)

style_axes(ax)
ax.set_ylabel(r'VPD effect on daytime flux ($\sigma$)', fontsize=AX_LABELS_FONTSIZE)
ax.legend(loc='lower left', bbox_to_anchor=(0.015, 0.09), frameon=False,
          fontsize=AX_LABELS_FONTSIZE * 0.85, handlelength=2.2)

if ax_reco is not None:
    style_axes(ax_reco)
    ax_reco.spines['left'].set_visible(False)
    ax_reco.tick_params(axis='y', left=False)
    ax_reco.legend(loc='lower left', bbox_to_anchor=(0.015, 0.09), frameon=False,
                   fontsize=AX_LABELS_FONTSIZE * 0.85, handlelength=2.2)
    flux, thr, thr_lo, thr_hi = control_threshold or ('RECO_ZSCORE', np.nan, np.nan, np.nan)
    lo_txt = f"{thr_lo:.2f}" if np.isfinite(thr_lo) else "open"
    hi_txt = f"{thr_hi:.2f}" if np.isfinite(thr_hi) else "open"
    y_text = ymax + 0.14 * span
    ax_reco.plot([thr, thr], [0, y_text], ls=(0, (1, 3)), color=COLORS[flux], lw=1.2, zorder=2)
    ax_reco.text(thr, y_text, f"{NAMES[flux]} {thr:.2f}", color=COLORS[flux], ha='center',
                 va='bottom', fontsize=AX_LABELS_FONTSIZE * 0.9, fontweight='bold', zorder=6,
                 bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
    ax_reco.text(0.97, 0.97, f'95 % interval {lo_txt} to {hi_txt}:\ncrossing not resolved',
                 transform=ax_reco.transAxes, ha='right', va='top', color='#777777',
                 fontsize=AX_LABELS_FONTSIZE * 0.8, linespacing=1.3)
    for panel, letter in ((ax, 'a'), (ax_reco, 'b')):
        panel.text(-0.02 if panel is ax_reco else -0.11, 1.12, letter, transform=panel.transAxes,
                   fontsize=AX_LABELS_FONTSIZE * 1.4, fontweight='bold', ha='left', va='top')

fig.savefig(f'{stem}.png', dpi=300, facecolor='white')
pd.concat(curves, ignore_index=True).to_csv(f'{stem}_CURVES.csv', index=False)
thresholds = pd.DataFrame(thresholds)
thresholds.to_csv(f'{stem}_THRESHOLDS.csv', index=False)

print()
print(thresholds.round(3).to_string(index=False))
print(f"\nkPa axis: {KPA_MEAN:.3f} + z * {KPA_SD:.3f}, network mean of the site statistics")
print(f"\nSaved {stem}.png, _CURVES.csv and _THRESHOLDS.csv")

if SHOW_PLOT:
    plt.show()
