"""
Convert the VPD threshold from standardized units into kPa.

Reviewer 2 could not identify this step in the code, and rightly so: the numbers in
the manuscript were produced by hand. This script does it reproducibly.

Each site standardizes VPD against its own mean and standard deviation, so a
threshold expressed in sigma means a different absolute VPD at every site. The
absolute threshold for one site is

    VPD_abs = VPD_mean + z * VPD_sd

with both statistics in hPa, taken over the same records the models saw and stored
by stage 21. Dividing by 10 gives kPa. Sites are then averaged with equal weight,
matching the aggregation used everywhere else in the analysis, and the biome value
is the mean over the sites of that biome.

The published range of 1.22 to 1.32 kPa is the span of the four biome means. It is
not a confidence interval, and the biome intervals overlap heavily.

Reads the z-score thresholds from the Figure 4 coefficient table, so a rerun of 54
flows through automatically.
"""
from pathlib import Path

import pandas as pd

from src.paths import load_settings

VARIANT = ""
SITE_SUBSET = ""
FLUX = 'NEP_ZSCORE'

# CD-Ygb records VPD in Pa where every other site uses hPa, which is documented in
# 21_prepare_input_data.py. Its site mean of 1810 would otherwise pull the global
# threshold from 1.26 to 2.19 kPa. Excluded rather than rescaled, because the
# z-scores it contributes elsewhere are unaffected by the unit and only this
# absolute conversion breaks.
PA_UNIT_SITES = ['CD-Ygb']

COEFF_FILE = ("54_FIG-4_ResponseCurve_ShapMeans_NEP_ZSCORE_BIN_VPD_ZSCORE+"
              "VPD_ZSCORE_SHAPVALS+TA_ZSCORE_DATA_COEFFICIENTS.csv")


def to_kpa(sites: pd.DataFrame, z: float) -> float:
    """Site-wise absolute threshold in kPa, averaged with equal weight per site."""
    return ((sites['VPD_Z0'] + z * sites['VPD_SD']) / 10).mean()


def parse_threshold(text: str) -> tuple:
    """Split '0.20 [0.05, 0.34]' into its three numbers."""
    point, rest = text.split('[')
    lo, hi = rest.rstrip(']').split(',')
    return float(point), float(lo), float(hi)


def main():
    settings = load_settings()
    sites = pd.read_csv(Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
                        / "21_SUBSETS_parquet_vars_stats_subsets.csv")
    dropped = sites[sites['SITE'].isin(PA_UNIT_SITES)]
    sites = sites[~sites['SITE'].isin(PA_UNIT_SITES)]
    for _, row in dropped.iterrows():
        print(f"Excluded {row['SITE']}, VPD recorded in Pa (site mean {row['VPD_Z0']:.0f})")

    coeffs = pd.read_csv(Path(settings['DIR_PLOTS_OUT']) / FLUX / 'conditional'
                         / VARIANT / SITE_SUBSET / COEFF_FILE)

    rows = []
    for _, c in coeffs.iterrows():
        group = c['IGBP']
        z, z_lo, z_hi = parse_threshold(c['Threshold'])
        sub = sites if group == 'ALL SITES' else sites[sites['IGBP'] == group]
        rows.append({'group': group, 'n_sites': len(sub), 'z': z, 'z_lower': z_lo, 'z_upper': z_hi,
                     'kPa': to_kpa(sub, z), 'kPa_lower': to_kpa(sub, z_lo), 'kPa_upper': to_kpa(sub, z_hi)})

    out = pd.DataFrame(rows)
    print()
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    biomes = out[out['group'] != 'ALL SITES']
    print(f"\nBiome range: {biomes['kPa'].min():.2f} to {biomes['kPa'].max():.2f} kPa, "
          f"the span of {len(biomes)} biome means, not a confidence interval.")
    overlap = biomes['kPa_lower'].max() <= biomes['kPa_upper'].min()
    print(f"Every biome interval overlaps every other: {overlap}")

    dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / 'conditional' / VARIANT / SITE_SUBSET
    dir_out.mkdir(parents=True, exist_ok=True)
    outfile = dir_out / "58_Threshold_zscore_to_kPa.csv"
    out.to_csv(outfile, index=False)
    print(f"\nSaved {outfile}")


if __name__ == '__main__':
    main()
