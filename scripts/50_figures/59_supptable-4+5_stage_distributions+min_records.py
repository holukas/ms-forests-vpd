"""
Two supplementary tables about the stress stages.

Table A shows what the drivers look like inside each stage, for all forests together and
for each forest type, with the number of sites and records. Shortwave radiation is the
one driver the stage definitions never constrain, so its distribution says whether the
stages also differ in light and not only in heat and dryness. Both the standardized
values the stages are defined on and the measured values are given, since a fixed sigma
cut-off lands at a different absolute value in every site. The numbers are read from the
three CSVs written by `scripts/40_aggregation/45_stage_distributions.py`; nothing is
recomputed here.

Table B asks whether the Stage 8 result depends on sites that contribute only a handful
of compound-extreme records. No minimum was imposed. The table raises a minimum from 1 to
100 records per site and reports, at each step, how many sites and how many Stage 8
records survive, how many of the surviving sites still show a negative VPD effect, the
three effects Figure 3 reports, and the ratio of the VPD effect to the soil water effect
with a bootstrap interval over sites. This part reads the per-site SHAP files, because the
count of Stage 8 records per site is not in any aggregated file.

The bootstrap uses a fixed seed, so the interval is reproducible.

Reads:
    40_aggregation/<FLUX>/conditional/45_StageDistributions_overall.csv
    40_aggregation/<FLUX>/conditional/45_StageDistributions_byIGBP.csv
    30_shap/<FLUX>/conditional/{site}_shap-conditional_<FLUX>.parquet

Writes, into the plot folder next to the other supplementary tables:
    59_SUPPTABLE-4_StageDistributions_<FLUX>.csv | .xlsx
    59_SUPPTABLE-5_Stage8MinRecords_<FLUX>.csv | .xlsx   one column per minimum
    59_SUPPTABLE-5_Stage8MinRecords_<FLUX>_DATA.csv      the same, one row per minimum, unrounded
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import src.stages as stg
from src.paths import load_settings

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""
SITE_SUBSET = ""

# Forest types, in the order used by the other stage figures and tables.
IGBP_ORDER = ['ENF', 'DBF', 'MF', 'EBF']

# Driver order, display names, units and the factor that takes the stored value into
# that unit. The files use the FLUXNET variable names, the manuscript calls soil water
# SM and incoming shortwave radiation SW. VPD is stored in hPa and the manuscript works
# in kPa throughout, so it is divided by ten here and nowhere else.
DRIVERS = [('TA', 'TA', 'degC', 1.0), ('VPD', 'VPD', 'kPa', 0.1),
           ('SWC', 'SM', '%', 1.0), ('SWIN', 'SW', 'W m-2', 1.0)]

# Stage cut-offs, repeated here only so the table can state them.
CUT_A, CUT_B, CUT_C = 0.32, 0.71, 1.28

STAGE_DEFINITIONS = {
    0: "All records, no restriction",
    1: "Normal: TA, VPD and SM each within +/-a",
    2: "Warm: TA in (a, b], SM within +/-a, VPD <= c",
    3: "Warm and dry soil: TA in (a, b], SM in [-b, -a), VPD <= c",
    4: "Hot and dry soil: TA in (b, c], SM in [-b, -a), VPD <= c",
    5: "Hot and very dry soil: TA in (b, c], SM in [-c, -b), VPD <= c",
    6: "Extremely hot, very dry soil: TA > c, SM in [-c, -b), VPD <= c",
    7: "Extremely hot, extremely dry soil: TA > c, SM < -c, VPD <= c",
    8: "Compound extreme: TA > c, SM < -c, VPD > c",
}

# Table B settings
MIN_RECORDS_STEPS = [1, 5, 10, 20, 30, 50, 100]
N_BOOTSTRAP = 10000
SEED = 42

# The two values the Results quote for shortwave radiation, checked below.
SW_CHECK = {1: 0.34, 8: 0.65}

settings = load_settings()
shap_type = 'conditional' if CONDITIONAL else 'interventional'
dir_agg = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT / SITE_SUBSET
dir_shap = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type / VARIANT
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT / SITE_SUBSET
dir_out.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Table A, driver distributions per stage
# ---------------------------------------------------------------------------

def stat_cell(row, driver: str, tag: str, decimals: int, factor: float = 1.0) -> str:
    """Median across sites of the site median, with the interquartile range."""
    value = row.get(f"{driver}_{tag}")
    if pd.isna(value):
        return "n/a"
    # Script 45 writes the range as the string "[q25, q75]". Rescaling a unit means
    # taking the two numbers back out of it.
    quartiles = [float(part) * factor
                 for part in str(row.get(f"{driver}_{tag}_iqr")).strip('[]').split(',')]
    return (f"{value * factor:.{decimals}f} "
            f"[{quartiles[0]:.{decimals}f}, {quartiles[1]:.{decimals}f}]")


def stage_block(frame: pd.DataFrame, group_name: str) -> list:
    rows = []
    for stage in sorted(frame['stage'].unique()):
        record = frame[frame['stage'] == stage].iloc[0]
        row = {'Group': group_name,
               'Stage': int(stage),
               # No definition column: the stages are defined in Supplementary Table 3 and
               # the caption points there. STAGE_DEFINITIONS stays for the console output.
               'Sites': int(record['sites']),
               'Records': int(record['records'])}
        for column, label, unit, factor in DRIVERS:
            # Two decimals for VPD in kPa, which is a number near one, otherwise one.
            decimals = 2 if factor != 1.0 else 1
            row[f"{label} (sigma)"] = stat_cell(record, column, 'z', 2)
            row[f"{label} ({unit})"] = stat_cell(record, column, 'abs', decimals, factor)
        rows.append(row)
    return rows


overall = pd.read_csv(dir_agg / "45_StageDistributions_overall.csv")
by_igbp = pd.read_csv(dir_agg / "45_StageDistributions_byIGBP.csv")

table_a_rows = stage_block(overall, 'Global forests')
for igbp in IGBP_ORDER:
    subset = by_igbp[by_igbp['IGBP'] == igbp]
    if subset.empty:
        print(f"  no rows for {igbp} in 45_StageDistributions_byIGBP.csv")
        continue
    table_a_rows += stage_block(subset, igbp)

table_a = pd.DataFrame(table_a_rows)
table_a.to_csv(dir_out / f"59_SUPPTABLE-4_StageDistributions_{FLUX}.csv",
               index=False, encoding='utf-8-sig')
table_a.to_excel(dir_out / f"59_SUPPTABLE-4_StageDistributions_{FLUX}.xlsx", index=False)

# The Results sentence quotes the shortwave radiation medians of two stages. If the
# stored numbers disagree, the sentence is what has to be checked.
print("\nShortwave radiation across all forests, the driver no stage constrains")
for stage, expected in SW_CHECK.items():
    got = overall.loc[overall['stage'] == stage, 'SWIN_z'].iloc[0]
    verdict = "matches the text" if abs(got - expected) < 0.005 else "DOES NOT MATCH THE TEXT"
    print(f"  stage {stage}: {got:+.3f} sigma, text says {expected:+.2f}, {verdict}")

# ---------------------------------------------------------------------------
# Table B, minimum number of Stage 8 records per site
# ---------------------------------------------------------------------------

COLS = ['TA_ZSCORE', 'VPD_ZSCORE', 'SWC_ZSCORE', 'SUM',
        'VPD_ZSCORE_SHAPVALS', 'SWC_ZSCORE_SHAPVALS']

print(f"\nReading Stage 8 from the per-site files in {dir_shap}")
site_rows = []
files = sorted(glob.glob(str(dir_shap / f'*_shap-{shap_type}_{FLUX}.parquet')))
for i, filepath in enumerate(files):
    site = Path(filepath).name.split('_')[0]
    df = pq.read_table(filepath, columns=COLS).to_pandas()
    stage8, *_ = stg.stage_8(df)
    if stage8.empty:
        continue
    site_rows.append({'site': site,
                      'n_records': len(stage8),
                      'net': stage8['SUM'].mean(),
                      'vpd': stage8['VPD_ZSCORE_SHAPVALS'].mean(),
                      'sm': stage8['SWC_ZSCORE_SHAPVALS'].mean()})
    if i % 50 == 0:
        print(f"  {i + 1}/{len(files)} sites")

sites = pd.DataFrame(site_rows)
records_total = int(sites['n_records'].sum())
print(f"  {len(sites)} of {len(files)} sites have at least one Stage 8 record, "
      f"{records_total} records in total")

rng = np.random.default_rng(SEED)


def ratio_interval(vpd: np.ndarray, sm: np.ndarray) -> tuple:
    """Ratio of the two mean effects, with a bootstrap interval over sites."""
    if len(vpd) < 2 or sm.mean() == 0:
        return np.nan, np.nan, np.nan
    point = vpd.mean() / sm.mean()
    draws = rng.integers(0, len(vpd), size=(N_BOOTSTRAP, len(vpd)))
    means_sm = sm[draws].mean(axis=1)
    ratios = vpd[draws].mean(axis=1) / means_sm
    ratios = ratios[np.isfinite(ratios)]
    lower, upper = np.percentile(ratios, [2.5, 97.5])
    return point, lower, upper


table_b_rows = []
for minimum in MIN_RECORDS_STEPS:
    kept = sites[sites['n_records'] >= minimum]
    if kept.empty:
        continue
    vpd = kept['vpd'].to_numpy(float)
    sm = kept['sm'].to_numpy(float)
    ratio, lower, upper = ratio_interval(vpd, sm)
    table_b_rows.append({
        'Minimum Stage 8 records per site': minimum,
        'Sites kept': len(kept),
        'Stage 8 records kept': int(kept['n_records'].sum()),
        'Share of Stage 8 records kept (%)': 100 * kept['n_records'].sum() / records_total,
        'Sites with a negative VPD effect': int((vpd < 0).sum()),
        'Share of kept sites with a negative VPD effect (%)': 100 * (vpd < 0).mean(),
        'Net effect (sigma)': kept['net'].mean(),
        'VPD effect (sigma)': vpd.mean(),
        'SM effect (sigma)': sm.mean(),
        'VPD to SM ratio': ratio,
        'VPD to SM ratio, 95% bootstrap': f"[{lower:.2f}, {upper:.2f}]",
    })

table_b = pd.DataFrame(table_b_rows)

# The display table is the transpose: one column per minimum, one row per quantity, with
# short row labels and rounded values, because the long headers did not fit an A4 page.
# The ratio and its interval share one cell. The long form stays beside it as _DATA.csv.
def _fmt(value, decimals):
    return f"{value:.{decimals}f}"


display_rows = [
    ('Sites kept', lambda r: f"{int(r['Sites kept'])}"),
    ('Stage 8 records kept', lambda r: f"{int(r['Stage 8 records kept']):,}"),
    ('Share of Stage 8 records kept (%)', lambda r: _fmt(r['Share of Stage 8 records kept (%)'], 1)),
    ('Sites with a negative VPD effect', lambda r: f"{int(r['Sites with a negative VPD effect'])}"),
    ('Share of kept sites with a negative VPD effect (%)',
     lambda r: _fmt(r['Share of kept sites with a negative VPD effect (%)'], 1)),
    ('Net effect (σ)', lambda r: _fmt(r['Net effect (sigma)'], 2)),
    ('VPD effect (σ)', lambda r: _fmt(r['VPD effect (sigma)'], 2)),
    ('SM effect (σ)', lambda r: _fmt(r['SM effect (sigma)'], 2)),
    ('VPD to SM ratio [95% bootstrap interval]',
     lambda r: f"{r['VPD to SM ratio']:.2f} {r['VPD to SM ratio, 95% bootstrap']}"),
]
table_b_display = pd.DataFrame(
    {str(int(r['Minimum Stage 8 records per site'])): [f(r) for _, f in display_rows]
     for _, r in table_b.iterrows()},
    index=[label for label, _ in display_rows])
table_b_display.index.name = 'Minimum Stage 8 records per site'
table_b_display.to_csv(dir_out / f"59_SUPPTABLE-5_Stage8MinRecords_{FLUX}.csv", encoding='utf-8-sig')
table_b_display.to_excel(dir_out / f"59_SUPPTABLE-5_Stage8MinRecords_{FLUX}.xlsx")
table_b.to_csv(dir_out / f"59_SUPPTABLE-5_Stage8MinRecords_{FLUX}_DATA.csv", index=False, encoding='utf-8-sig')

# ---------------------------------------------------------------------------
# The numbers the two Results sentences rest on
# ---------------------------------------------------------------------------

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 50)

print(f"\nTable A, {len(table_a)} rows, stages 0 to 8 for all forests and for each forest type")
print(table_a[['Group', 'Stage', 'Sites', 'Records',
               'TA (sigma)', 'VPD (sigma)', 'SM (sigma)', 'SW (sigma)']].to_string(index=False))

print("\nTable B, minimum number of Stage 8 records per site")
print(table_b.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

base = table_b.iloc[0]
ten = table_b[table_b['Minimum Stage 8 records per site'] == 10]
print(f"\nStage 8 covers {int(base['Sites kept'])} sites and {records_total} records, "
      f"{int(base['Sites with a negative VPD effect'])} sites "
      f"({base['Share of kept sites with a negative VPD effect (%)']:.1f}%) show a negative VPD effect.")
print(f"The VPD effect is {base['VPD to SM ratio']:.2f} times the soil water effect, "
      f"{base['VPD to SM ratio, 95% bootstrap']} over sites.")
if not ten.empty:
    ten = ten.iloc[0]
    dropped_sites = int(base['Sites kept']) - int(ten['Sites kept'])
    dropped_records = 100 - ten['Share of Stage 8 records kept (%)']
    print(f"Requiring at least 10 Stage 8 records removes {dropped_sites} sites and "
          f"{dropped_records:.1f}% of the Stage 8 records, leaving "
          f"{int(ten['Sites kept'])} sites, of which "
          f"{ten['Share of kept sites with a negative VPD effect (%)']:.1f}% stay negative, "
          f"and a ratio of {ten['VPD to SM ratio']:.2f}.")

print(f"\nStage cut-offs used by the definitions: a = {CUT_A}, b = {CUT_B}, c = {CUT_C} sigma.")
print(f"Saved four files to {dir_out}")
