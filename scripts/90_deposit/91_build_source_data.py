"""
Collect the Source Data of the paper: the data behind every figure, table and
Supplementary Data file of the main analysis, one plainly named file each.

The journal asks for the data behind the display items as labelled files in one zipped
folder named "Source Data". The files here are copies of pipeline outputs, renamed after
the display item they belong to; nothing is recomputed. README.txt in the archive lists
every file with the output it was copied from.

Without arguments it is a dry run that checks every source file. With ``--build`` it
writes the same archive twice (replacing existing ones):
    <DATA_ROOT>/deposit_eth_research_collection/ms-forests-vpd_SourceData.zip
    <DATA_ROOT>/submission_nature_communications/Source Data.zip, uploaded to the journal
Run it before 93_build_deposit_archives.py --build, which lists the deposit copy in
MANIFEST.csv and ARCHIVES.csv.
"""
import sys
import zipfile
from pathlib import Path

from src.paths import DATA_ROOT, DEPOSIT_DIR, SUBMISSION_DIR

OUTPUTS = DATA_ROOT / "data" / "outputs"
ZIP_DEPOSIT = DEPOSIT_DIR / "ms-forests-vpd_SourceData.zip"
ZIP_SUBMISSION = SUBMISSION_DIR / "Source Data.zip"

PLOTS = "50_plots/NEP_ZSCORE/conditional"
AGG = "40_aggregation/NEP_ZSCORE/conditional"
FIG4 = f"{PLOTS}/54_FIG-4_ResponseCurve_ShapMeans_NEP_ZSCORE_BIN_VPD_ZSCORE+VPD_ZSCORE_SHAPVALS+TA_ZSCORE"

# (name in the archive, display item, source file relative to data/outputs/).
# Main analysis only; the sensitivity variants are in the deposit.
FILES = [
    ("Fig1.csv", "Fig. 1", f"{PLOTS}/51_FIG-1_WorldMap_MAT_MAP_ERA5_DATA.csv"),
    ("Fig2.csv", "Fig. 2", f"{PLOTS}/52_FIG-2_FlamePlotsShapValues-VPD_ZSCORE_SHAPVALS_NEP_ZSCORE_DATA.csv"),
    ("Fig3.csv", "Fig. 3", f"{PLOTS}/53_FIG-3_SankeyPlotStages_NEP_ZSCORE_DATA.csv"),
    ("Fig4_all_sites.csv", "Fig. 4", f"{FIG4}_ALLSITES_DATA.csv"),
    ("Fig4_ENF.csv", "Fig. 4", f"{FIG4}_ENF_DATA.csv"),
    ("Fig4_DBF.csv", "Fig. 4", f"{FIG4}_DBF_DATA.csv"),
    ("Fig4_MF.csv", "Fig. 4", f"{FIG4}_MF_DATA.csv"),
    ("Fig4_EBF.csv", "Fig. 4", f"{FIG4}_EBF_DATA.csv"),
    ("Fig4_polynomial_coefficients.csv", "Fig. 4", f"{FIG4}_DATA_COEFFICIENTS.csv"),
    ("Table1.csv", "Table 1", f"{PLOTS}/55_TABLE-1_ThresholdRobustness_NEP_ZSCORE.csv"),
    ("SupplementaryFig1.csv", "Supplementary Fig. 1",
     f"{PLOTS}/52_SUPPFIG-1_FlamePlotsDriverEffects-TA+SM_NEP_ZSCORE_DATA.csv"),
    ("SupplementaryFig2.csv", "Supplementary Fig. 2",
     f"{AGG}/44_SHAPVALUES-conditional_AggregatedAcrossScenarios_NEP_ZSCORE.csv"),
    ("SupplementaryFig3.csv", "Supplementary Fig. 3", f"{PLOTS}/53_SUPPFIG-3_SankeyPlotStages_NEP_ZSCORE_DATA.csv"),
    ("SupplementaryFig4.csv", "Supplementary Fig. 4", f"{AGG}/47_THRESHOLD_Robustness_NEP_ZSCORE.csv"),
    ("SupplementaryFig5_curves.csv", "Supplementary Fig. 5",
     "50_plots/NEP_ZSCORE/ale/57_SUPPFIG-5_ALE_ResponseCurve_VPD_ZSCORE_NEP_ZSCORE_DATA.csv"),
    ("SupplementaryFig5_thresholds.csv", "Supplementary Fig. 5",
     "50_plots/NEP_ZSCORE/ale/57_SUPPFIG-5_ALE_ResponseCurve_VPD_ZSCORE_NEP_ZSCORE_DATA_Thresholds.csv"),
    ("SupplementaryFig6_curves.csv", "Supplementary Fig. 6",
     f"{PLOTS}/58_SUPPFIG-6_VpdResponseByFlux_NEP+GPP+RECO+ET_CURVES.csv"),
    ("SupplementaryFig6_thresholds.csv", "Supplementary Fig. 6",
     f"{PLOTS}/58_SUPPFIG-6_VpdResponseByFlux_NEP+GPP+RECO+ET_THRESHOLDS.csv"),
    ("SupplementaryFig7.csv", "Supplementary Fig. 7",
     f"{PLOTS}/52_SUPPFIG-7_FlamePlotsFluxes-NEP_ZSCORE_NEP_ZSCORE_DATA.csv"),
    ("SupplementaryFig8.csv", "Supplementary Fig. 8", f"{PLOTS}/60_SUPPFIG-8_ThresholdExceedance_NEP_ZSCORE_DATA.csv"),
    ("SupplementaryFig9_median.csv", "Supplementary Fig. 9", f"{PLOTS}/68_SUPPFIG-9_SiteCurvesKpa_NEP_ZSCORE_DATA.csv"),
    ("SupplementaryFig9_site_curves.csv", "Supplementary Fig. 9",
     f"{PLOTS}/68_SUPPFIG-9_SiteCurvesKpa_NEP_ZSCORE_DATA_SiteCurves.csv"),
    ("SupplementaryFig9_crossings.csv", "Supplementary Fig. 9",
     f"{PLOTS}/68_SUPPFIG-9_SiteCurvesKpa_Crossings_NEP_ZSCORE.csv"),
    ("SupplementaryFig10.csv", "Supplementary Fig. 10", f"{PLOTS}/66_SUPPFIG-10_DataFlow_Sankey_DATA.csv"),
    ("SupplementaryTables1-2_site_statistics.csv", "Supplementary Tables 1 and 2",
     "20_subsets/21_SUBSETS_parquet_vars_stats_subsets.csv"),
    ("SupplementaryTable1_ERA5_climate.csv", "Supplementary Table 1",
     "10_datasets/17_datasets_info_parquet_vars_stats_usedsites_era5.csv"),
    ("SupplementaryTable3_stage_definitions.csv", "Supplementary Table 3",
     f"{PLOTS}/61_SUPPTABLE-3_StageDefinitions_NEP_ZSCORE.csv"),
    ("SupplementaryTable3_cutoff_percentiles.csv", "Supplementary Table 3",
     f"{PLOTS}/61_SUPPTABLE-3_StageCutoffPercentiles_NEP_ZSCORE.csv"),
    ("SupplementaryTable4.csv", "Supplementary Table 4", f"{PLOTS}/59_SUPPTABLE-4_StageDistributions_NEP_ZSCORE.csv"),
    ("SupplementaryTable5.csv", "Supplementary Table 5", f"{PLOTS}/59_SUPPTABLE-5_Stage8MinRecords_NEP_ZSCORE.csv"),
    ("SupplementaryTable5_data.csv", "Supplementary Table 5",
     f"{PLOTS}/59_SUPPTABLE-5_Stage8MinRecords_NEP_ZSCORE_DATA.csv"),
    ("SupplementaryTable6.xlsx", "Supplementary Table 6", f"{PLOTS}/56_SUPPTABLE-6_Stages_SinaPlots_ShapMeans_NEP_ZSCORE.xlsx"),
    ("SupplementaryTable6_data.csv", "Supplementary Table 6",
     f"{PLOTS}/56_SUPPTABLE-6_Stages_SinaPlots_ShapMeans_NEP_ZSCORE_DATA-FeatureStatsFull.csv"),
    ("SupplementaryTable7.csv", "Supplementary Table 7", f"{PLOTS}/54_SUPPTABLE-7_ThresholdPolynomials_NEP_ZSCORE.csv"),
    ("SupplementaryTable8.csv", "Supplementary Table 8", f"{PLOTS}/63_SUPPTABLE-8_ThresholdVsSiteVPDRange_NEP_ZSCORE.csv"),
    ("SupplementaryTable9.csv", "Supplementary Table 9", f"{PLOTS}/62_SUPPTABLE-9_ThresholdDefinitions_NEP_ZSCORE.csv"),
    ("SupplementaryTable10.csv", "Supplementary Table 10", f"{PLOTS}/69_SUPPTABLE-10_FactorialSmVpd_NEP_ZSCORE.csv"),
    ("SupplementaryTable10_data.csv", "Supplementary Table 10",
     f"{PLOTS}/69_SUPPTABLE-10_FactorialSmVpd_NEP_ZSCORE_DATA.csv"),
    ("SupplementaryData1.csv", "Supplementary Data 1", f"{PLOTS}/67_SUPPDATA-1_DataFlowPerSite.csv"),
    ("SupplementaryData1.xlsx", "Supplementary Data 1", f"{PLOTS}/67_SUPPDATA-1_DataFlowPerSite.xlsx"),
    ("SupplementaryData2.csv", "Supplementary Data 2", f"{PLOTS}/64_SUPPDATA-2_SiteSkill_NEP_ZSCORE.csv"),
    ("SupplementaryData2.xlsx", "Supplementary Data 2", f"{PLOTS}/64_SUPPDATA-2_SiteSkill_NEP_ZSCORE.xlsx"),
]

README = """\
SOURCE DATA
An atmospheric dryness threshold limits daytime net CO2 uptake in forests

Hoertnagl L., Floriancic M. G., Gessler A., Vekuri H., Zweifel R., Etzold S., Gharun M.,
Kohonen K.-M., Feigenwinter I., Krebs L., Merbold L., Papale D., Scapucci L., Shekhar A.,
Meier P., Baur T., Buchmann N.

The data behind every figure, table and Supplementary Data file of the paper, one file per
item or panel group, named after the display item. Each file is a copy of an output of the
analysis pipeline (code: https://github.com/holukas/ms-forests-vpd), written by the
script whose number starts the source file name. The deposit in the ETH Research
Collection (https://doi.org/10.3929/ethz-c-000798579) holds the same outputs together with
every earlier step and the sensitivity runs. Column names follow the pipeline; the legend
of each display item gives the units. Licence: CC BY 4.0.

{table}
"""


def main(build: bool) -> None:
    missing = [src for _, _, src in FILES if not (OUTPUTS / src).is_file()]
    names = [name for name, _, _ in FILES]
    dupes = sorted({n for n in names if names.count(n) > 1})
    w = max(len(n) for n in names)
    for name, item, src in FILES:
        size = (OUTPUTS / src).stat().st_size / 1e6 if (OUTPUTS / src).is_file() else float("nan")
        print(f"{name:<{w}}  {size:7.2f} MB  {src}")
    total = sum((OUTPUTS / s).stat().st_size for _, _, s in FILES if (OUTPUTS / s).is_file())
    print(f"{len(FILES)} files, {total / 1e6:.1f} MB")
    for src in missing:
        print("PROBLEM: missing", src)
    for n in dupes:
        print("PROBLEM: name used twice", n)
    if not build:
        print("\nDry run. Add --build to write the archives.")
        return
    if missing or dupes:
        sys.exit("Not built: fix the problems above first.")

    table = "\n".join(f"{name:<{w}}  {item:<28}  data/outputs/{src}" for name, item, src in FILES)
    readme = README.format(table=f"{'File':<{w}}  {'Display item':<28}  Copied from\n{table}")
    for out in (ZIP_DEPOSIT, ZIP_SUBMISSION):
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("README.txt", readme)
            for name, _, src in FILES:
                z.write(OUTPUTS / src, arcname=name)
        print(f"written {out}: {len(FILES) + 1} files, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(build="--build" in sys.argv[1:])
