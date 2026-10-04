"""
Test whether the stage order drives the Stage 8 result by swapping the roles of VPD and soil water.

The main stage order adds extreme VPD only at Stage 8; the mirrored order in
`src/stages.py` escalates VPD and adds extreme soil dryness last. Cutoffs, the temperature
ladder and the final stage are the same. Compares:

1. the final step, the change in net effect from Stage 7 to Stage 8, in each order
2. the two Stage 7 conditions: extreme VPD without extreme soil dryness (mirrored) against
   extreme soil dryness without extreme VPD (main)

Prerequisite: run `44_agg_per_scenario.py` with STAGE_SEQUENCE = `"published"` and again
with `"mirrored"`. Script 53 with VARIANT = "mirrored-stages" draws the mirrored order.

Reads:
- 40_aggregation/<FLUX>/conditional/44_SHAPVALUES-conditional_AggregatedAcrossScenarios_<FLUX>.parquet
- the same file under conditional/mirrored-stages/

Writes:
- 80_info/<FLUX>/conditional/88_INFO_StageOrderTest_<FLUX>.csv: per stage and order
- 80_info/<FLUX>/conditional/88_INFO_StageOrderTest_<FLUX>_VERDICT.csv: the two comparisons
"""
from pathlib import Path

import pandas as pd

from src.paths import load_settings

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""
SITE_SUBSET = ""

STAGES = list(range(1, 9))
NET = 'NET_SHAPVALS_OVR_AVG'
DRIVERS = {'VPD': 'VPD_ZSCORE_SHAPVALS_OVR_AVG', 'TA': 'TA_ZSCORE_SHAPVALS_OVR_AVG',
           'SM': 'SWC_ZSCORE_SHAPVALS_OVR_AVG', 'SW': 'SWIN_ZSCORE_SHAPVALS_OVR_AVG'}

shap_type = 'conditional' if CONDITIONAL else 'interventional'
settings = load_settings()
base = (Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT
        / SITE_SUBSET)
outdir = Path(settings['DIR_INFO_OUT']) / FLUX / shap_type
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

# The two comparisons of the docstring.
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

outfile = outdir / f'88_INFO_StageOrderTest_{FLUX}.csv'
summary.to_csv(outfile, index=False)
pd.DataFrame(verdict).T.to_csv(str(outfile).replace('.csv', '_VERDICT.csv'))
print(f"\nSaved to {outfile}")
