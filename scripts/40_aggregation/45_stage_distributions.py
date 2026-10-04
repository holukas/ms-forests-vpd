"""
Driver distributions within each stress stage.

Takes the median of TA, VPD, SWC and SWIN per site and stage, then summarizes these
site medians (median, quartiles, site and record counts) across all sites and per forest type.
SWIN is the one driver no stage definition constrains, so its distribution shows
whether the stages also differ in light. Values are reported standardized, as the
stages are defined, and in measured units, since a fixed sigma cut-off falls at a
different absolute value at every site.

Reads the per-site SHAP files, which carry the drivers next to the attributions.
Writes 45_StageDistributions_perSite.csv, 45_StageDistributions_overall.csv and
45_StageDistributions_byIGBP.csv.
"""
from pathlib import Path

import pandas as pd

import diive as dv
import src.stages as stg
from src.paths import load_settings

VARIANT = ""
FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
DRIVERS = ['TA', 'VPD', 'SWC', 'SWIN']
STAGES = [stg.stage_0, stg.stage_1, stg.stage_2, stg.stage_3, stg.stage_4,
          stg.stage_5, stg.stage_6, stg.stage_7, stg.stage_8]


def describe(values: pd.Series) -> dict:
    return {'median': values.median(), 'q25': values.quantile(0.25),
            'q75': values.quantile(0.75), 'n': int(values.notna().sum())}


def main():
    settings = load_settings()
    shap_type = 'conditional' if CONDITIONAL else 'interventional'
    dir_in = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type / VARIANT
    subsets = pd.read_csv(Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
                          / "21_SUBSETS_parquet_vars_stats_subsets.csv")

    rows = []
    for ix, site in subsets.iterrows():
        filepath = dir_in / f"{site['SITE']}_shap-{shap_type}_{FLUX}.parquet"
        if not filepath.is_file():
            continue
        df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
        for i, stage in enumerate(STAGES):
            subset, *_ = stage(df)
            if subset.empty:
                continue
            row = {'SITE': site['SITE'], 'IGBP': site['IGBP'], 'stage': i, 'n_records': len(subset)}
            for driver in DRIVERS:
                for col, tag in ((driver, 'abs'), (f"{driver}_ZSCORE", 'z')):
                    if col in subset.columns:
                        row[f"{driver}_{tag}_median"] = subset[col].median()
            rows.append(row)
        if ix % 40 == 0:
            print(f"  {ix + 1}/{len(subsets)} sites")

    per_site = pd.DataFrame(rows)

    def summarise(frame, by):
        out = []
        for keys, g in frame.groupby(by, observed=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            row = dict(zip(by, keys))
            row['sites'] = g['SITE'].nunique()
            row['records'] = int(g['n_records'].sum())
            for driver in DRIVERS:
                for tag in ('z', 'abs'):
                    col = f"{driver}_{tag}_median"
                    if col in g:
                        stats = describe(g[col])
                        row[f"{driver}_{tag}"] = stats['median']
                        row[f"{driver}_{tag}_iqr"] = f"[{stats['q25']:.2f}, {stats['q75']:.2f}]"
            out.append(row)
        return pd.DataFrame(out)

    overall = summarise(per_site, ['stage'])
    by_biome = summarise(per_site, ['stage', 'IGBP'])

    print("\nmedian across sites of the site median, standardized units")
    cols = ['stage', 'sites', 'records'] + [f"{d}_z" for d in DRIVERS]
    print(overall[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\nsame in measured units, TA in degC, VPD and SWC and SWIN in their file units")
    cols = ['stage', 'sites', 'records'] + [f"{d}_abs" for d in DRIVERS]
    print(overall[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    swin = overall[['stage', 'SWIN_z', 'SWIN_abs']]
    print(f"\nShortwave radiation is never constrained by a stage definition. Its median"
          f" moves from {swin['SWIN_abs'].iloc[0]:.0f} at stage 0 to"
          f" {swin['SWIN_abs'].iloc[-1]:.0f} at stage 8, in the same units.")

    dir_out = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT
    dir_out.mkdir(parents=True, exist_ok=True)
    per_site.to_csv(dir_out / "45_StageDistributions_perSite.csv", index=False)
    overall.to_csv(dir_out / "45_StageDistributions_overall.csv", index=False)
    by_biome.to_csv(dir_out / "45_StageDistributions_byIGBP.csv", index=False)
    print(f"\nSaved three tables to {dir_out}")


if __name__ == '__main__':
    main()
