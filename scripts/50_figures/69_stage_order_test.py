"""
Does the stage order manufacture the result?

Reviewer 2 objected that the published stage sequence is not a balanced design. It lets
temperature rise and soil water fall step by step while holding VPD below its extreme
cut-off, and only adds extreme VPD at the last stage. Any driver held back to the finale
will look dramatic when it finally arrives, so the sequence may favour VPD.

The mirrored sequence in `src/stages.py` swaps the two roles: VPD escalates through the
stages and extreme soil dryness is added last. Cut-offs and the temperature ladder are
unchanged, so the two sequences differ only in which driver comes last, and they end at
the identical final stage.

This script reads the two stage 44 outputs and compares them. Two numbers answer the
objection:

  1. what the final step contributes in each sequence, that is the drop from the second
     to last stage to the last one
  2. the second to last stages themselves, which isolate one extreme driver each: extreme
     VPD without extreme soil dryness against extreme soil dryness without extreme VPD

Run `44_agg_per_scenario.py` twice first, once with STAGE_SEQUENCE = "published" and once
with "mirrored". Nothing is recomputed here.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.paths import load_settings

SHOW_PLOT = False

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""
SITE_SUBSET = ""

STAGES = list(range(1, 9))
NET = 'NET_SHAPVALS_OVR_AVG'
DRIVERS = {'VPD': 'VPD_ZSCORE_SHAPVALS_OVR_AVG', 'TA': 'TA_ZSCORE_SHAPVALS_OVR_AVG',
           'SM': 'SWC_ZSCORE_SHAPVALS_OVR_AVG', 'SW': 'SWIN_ZSCORE_SHAPVALS_OVR_AVG'}
COLORS = {'published': '#D55E00', 'mirrored': '#0072B2'}
AX_LABELS_FONTSIZE = 12

shap_type = 'conditional' if CONDITIONAL else 'interventional'
settings = load_settings()
base = (Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT
        / SITE_SUBSET)
outdir = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT / SITE_SUBSET
outdir.mkdir(parents=True, exist_ok=True)

FILENAME = f'44_SHAPVALUES-{shap_type}_AggregatedAcrossScenarios_{FLUX}.parquet'
SOURCES = {'published': base / FILENAME, 'mirrored': base / 'mirrored-stages' / FILENAME}
for name, path in SOURCES.items():
    if not path.is_file():
        raise FileNotFoundError(
            f"No {name} stage output at {path}. Run 44_agg_per_scenario.py with "
            f"STAGE_SEQUENCE = {name!r} first.")


def summarise(path, label):
    """Cross-site mean of the net effect and of each driver, per stage."""
    df = pd.read_parquet(path)
    rows = []
    for stage in STAGES:
        sub = df.loc[df['SCENARIO'] == stage]
        row = {'sequence': label, 'stage': stage, 'sites': len(sub),
               'records': int(sub['N_VALUES'].sum()),
               'condition': sub['CONDITION'].iloc[0] if len(sub) else '',
               'net': sub[NET].mean()}
        for short, col in DRIVERS.items():
            row[short] = sub[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


summary = pd.concat([summarise(p, name) for name, p in SOURCES.items()],
                    ignore_index=True)

# The two numbers that answer the objection.
verdict = {}
for name in SOURCES:
    seq = summary.loc[summary['sequence'] == name].set_index('stage')
    verdict[name] = {
        'second_last_net': seq.loc[7, 'net'],
        'last_net': seq.loc[8, 'net'],
        'final_step': seq.loc[8, 'net'] - seq.loc[7, 'net'],
    }
ratio = verdict['published']['final_step'] / verdict['mirrored']['final_step']
isolated = verdict['mirrored']['second_last_net'] / verdict['published']['second_last_net']

print(summary.round(3).to_string(index=False))
print()
print(f"final step, published (extreme VPD added last):  "
      f"{verdict['published']['final_step']:+.3f}")
print(f"final step, mirrored (extreme soil dryness last): "
      f"{verdict['mirrored']['final_step']:+.3f}")
print(f"adding VPD last costs {ratio:.1f} times more than adding soil dryness last")
print()
print(f"extreme VPD alone (mirrored stage 7):        "
      f"{verdict['mirrored']['second_last_net']:+.3f}")
print(f"extreme soil dryness alone (published st. 7): "
      f"{verdict['published']['second_last_net']:+.3f}")
print(f"VPD alone is {isolated:.1f} times larger")

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150,
                              gridspec_kw={'width_ratios': [1.5, 1]},
                              constrained_layout=True)

# Left: the net effect along both sequences.
for name in SOURCES:
    seq = summary.loc[summary['sequence'] == name]
    label = ('published, extreme VPD added last' if name == 'published'
             else 'mirrored, extreme soil dryness added last')
    ax.plot(seq['stage'], seq['net'], 'o-', color=COLORS[name], lw=2, ms=6, label=label,
            zorder=3)
ax.axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.5, zorder=1)
ax.set_xticks(STAGES)
ax.set_xlabel('Stage', fontsize=AX_LABELS_FONTSIZE)
ax.set_ylabel(r'Net effect on daytime NEP ($\sigma$)', fontsize=AX_LABELS_FONTSIZE)
ax.set_title('a | Both sequences end at the same stage 8', fontsize=AX_LABELS_FONTSIZE,
             loc='left', fontweight='bold')
ax.legend(fontsize=AX_LABELS_FONTSIZE * 0.8, frameon=False, loc='lower left')
ax.tick_params(labelsize=AX_LABELS_FONTSIZE * 0.9)

# Right: what the last driver contributes, and what each extreme costs on its own.
# Short labels. The panel title carries the word extreme, so repeating it four times
# here only makes the ticks collide.
bars = {
    'VPD\nadded last': (verdict['published']['final_step'], COLORS['published']),
    'soil dryness\nadded last': (verdict['mirrored']['final_step'],
                                 COLORS['mirrored']),
    'VPD\nalone': (verdict['mirrored']['second_last_net'], COLORS['published']),
    'soil dryness\nalone': (verdict['published']['second_last_net'],
                             COLORS['mirrored']),
}
for x, (label, (value, color)) in enumerate(bars.items()):
    ax2.bar(x, value, color=color, width=0.62, zorder=3)
    # Inside the bar, just above its lower end. Below it the label would run into
    # the tick labels, since every bar is negative.
    ax2.text(x, value + 0.012, f'{value:+.3f}', ha='center', va='bottom',
             color='white', fontsize=AX_LABELS_FONTSIZE * 0.85,
             fontweight='bold', zorder=4)
ax2.axhline(0, color='black', lw=0.8, zorder=1)
ax2.set_xticks(range(len(bars)))
ax2.set_xticklabels(bars.keys(), fontsize=AX_LABELS_FONTSIZE * 0.8)
ax2.set_ylabel(r'Effect on daytime NEP ($\sigma$)', fontsize=AX_LABELS_FONTSIZE)
ax2.set_title('b | Effect of each extreme, whichever comes last',
              fontsize=AX_LABELS_FONTSIZE, loc='left', fontweight='bold')
ax2.tick_params(labelsize=AX_LABELS_FONTSIZE * 0.9)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
    ax2.spines[sp].set_visible(False)

fig.suptitle('PLANNED supplementary figure, not yet adopted   '
             '(stage order test, 208 sites)',
             fontsize=AX_LABELS_FONTSIZE * 0.85, color='#D55E00', fontweight='bold',
             x=0.01, ha='left')

outfile = outdir / f'69_PLANNED-SUPPFIG_StageOrderTest_{FLUX}.png'
fig.savefig(outfile, dpi=300, facecolor='white', bbox_inches='tight')
summary.to_csv(str(outfile).replace('.png', '_DATA.csv'), index=False)
pd.DataFrame(verdict).T.to_csv(str(outfile).replace('.png', '_VERDICT.csv'))
print(f"\nSaved to {outfile}")
if SHOW_PLOT:
    plt.show()
