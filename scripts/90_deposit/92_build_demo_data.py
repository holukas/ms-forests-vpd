"""
Build a small demo dataset: the model input of one site, to try the code without
downloading the full deposit.

The archive holds the site table of stage 21 reduced to DEMO_SITE, the model input of
that site, and the published output of script 31 for it as the expected result. Paths
inside the archive are relative to data/outputs/, like the stage archives of script 93.
Unpacked into the data/outputs/ folder of an empty data root, script 31 runs as
committed and fits the one site. README.txt in the archive gives the steps.

Without arguments it is a dry run. With ``--build`` it writes
<DATA_ROOT>/deposit_eth_research_collection/ms-forests-vpd_demo.zip (replacing an
existing one). Run it before 93_build_deposit_archives.py --build, which lists the
archive in MANIFEST.csv and ARCHIVES.csv.
"""
import io
import sys
import zipfile

import pandas as pd

from src.paths import DATA_ROOT, DEPOSIT_DIR

DEMO_SITE = "CH-Dav"

OUTPUTS = DATA_ROOT / "data" / "outputs"
ZIP_DEMO = DEPOSIT_DIR / "ms-forests-vpd_demo.zip"
TABLE = "20_subsets/21_SUBSETS_parquet_vars_stats_subsets.csv"
SUBSET = f"20_subsets/21_subsets_parquet/{DEMO_SITE}_subset_GPPhighest4_qc0_daytime.parquet"
EXPECTED = f"30_shap/NEP_ZSCORE/conditional/{DEMO_SITE}_shap-conditional_NEP_ZSCORE.parquet"
EXPECTED_IN_ZIP = f"expected_output/{DEMO_SITE}_shap-conditional_NEP_ZSCORE.parquet"

README = f"""\
DEMO DATA
An atmospheric dryness threshold limits daytime net CO2 uptake in forests

The model input of one site, {DEMO_SITE}, to try the code without the full deposit.

Contents, relative to data/outputs/:
- {TABLE}: the site table of stage 21, reduced to {DEMO_SITE}
- {SUBSET}: the peak-season daytime records of the site
- {EXPECTED_IN_ZIP}: the output of script 31 for the site as published (not read by the code)

Steps:
1. Install the code repository as its README describes (https://github.com/holukas/ms-forests-vpd).
2. Create an empty data root folder, set it as the environment variable MS_FORESTS_VPD_DATA,
   and unpack this archive into <data root>/data/outputs/.
3. Run: uv run python scripts/30_shap/31_shap.py
   It fits the XGBoost model of the site in five-fold cross-validation and computes the
   out-of-sample SHAP values.

Expected output: <data root>/data/outputs/30_shap/NEP_ZSCORE/conditional/ with
{DEMO_SITE}_shap-conditional_NEP_ZSCORE.parquet, the run log and the cross-validation results.
With xgboost 3.0.5 on 24 threads the parquet file equals the one in expected_output/; with
another thread count the values differ by up to a few percent (see the repository README).
The run takes a few seconds.
"""


def main(build: bool) -> None:
    missing = [p for p in (TABLE, SUBSET, EXPECTED) if not (OUTPUTS / p).is_file()]
    for p in (TABLE, SUBSET, EXPECTED):
        size = (OUTPUTS / p).stat().st_size / 1e6 if (OUTPUTS / p).is_file() else float("nan")
        print(f"{size:7.2f} MB  {p}")
    table = pd.read_csv(OUTPUTS / TABLE) if not missing else None
    if table is not None and (table["SITE"] == DEMO_SITE).sum() != 1:
        missing.append(f"{DEMO_SITE} as one row of {TABLE}")
    for p in missing:
        print("PROBLEM: missing", p)
    if not build:
        print("\nDry run. Add --build to write the archive.")
        return
    if missing:
        sys.exit("Not built: fix the problems above first.")

    buf = io.StringIO()
    table[table["SITE"] == DEMO_SITE].to_csv(buf, index=False)
    ZIP_DEMO.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_DEMO, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", README)
        z.writestr(TABLE, buf.getvalue())
        z.write(OUTPUTS / SUBSET, arcname=SUBSET)
        z.write(OUTPUTS / EXPECTED, arcname=EXPECTED_IN_ZIP)
    print(f"written {ZIP_DEMO}: {ZIP_DEMO.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(build="--build" in sys.argv[1:])
