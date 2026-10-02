"""
Copy the figures of the paper into the documentation, scaled to web size.

The figure scripts write full-resolution PNG files to data/outputs/50_plots/. This script
copies the main and supplementary figures of the main analysis into docs/figures/,
scaled to WIDTH pixels wide and reduced to 256 colors, so that the Figures page of the
documentation shows them without the data folder and the repository stays small. The
full-resolution files are in the data deposit.

Without arguments it is a dry run that checks every source file. With ``--build`` it
writes the files to docs/figures/ (replacing existing ones).
"""
import sys

from PIL import Image

from src.paths import DATA_ROOT, REPO_ROOT

OUTPUTS = DATA_ROOT / "data" / "outputs"
DOCS_FIGURES = REPO_ROOT / "docs" / "figures"
WIDTH = 1600

PLOTS = "50_plots/NEP_ZSCORE/conditional"

# (name in docs/figures/, source file relative to data/outputs/)
FILES = [
    ("fig1.png", f"{PLOTS}/51_FIG-1_WorldMap_MAT_MAP_ERA5.png"),
    ("fig2.png", f"{PLOTS}/52_FIG-2_FlamePlotsShapValues-VPD_ZSCORE_SHAPVALS_NEP_ZSCORE.png"),
    ("fig3.png", f"{PLOTS}/53_FIG-3_SankeyPlotStages_NEP_ZSCORE.png"),
    ("fig4.png", f"{PLOTS}/54_FIG-4_ResponseCurve_ShapMeans_NEP_ZSCORE_BIN_VPD_ZSCORE+VPD_ZSCORE_SHAPVALS+TA_ZSCORE.png"),
    ("suppfig1.png", f"{PLOTS}/52_SUPPFIG-1_FlamePlotsDriverEffects-TA+SM_NEP_ZSCORE.png"),
    ("suppfig2.png", f"{PLOTS}/56_SUPPFIG-2_Stages_SinaPlots_ShapMeans_NEP_ZSCORE.png"),
    ("suppfig3.png", f"{PLOTS}/53_SUPPFIG-3_SankeyPlotStages_NEP_ZSCORE.png"),
    ("suppfig4.png", f"{PLOTS}/55_SUPPFIG-4_ThresholdRobustness_NEP_ZSCORE.png"),
    ("suppfig5.png", "50_plots/NEP_ZSCORE/ale/57_SUPPFIG-5_ALE_ResponseCurve_VPD_ZSCORE_NEP_ZSCORE.png"),
    ("suppfig6.png", f"{PLOTS}/58_SUPPFIG-6_VpdResponseByFlux_NEP+GPP+RECO+ET.png"),
    ("suppfig7.png", f"{PLOTS}/52_SUPPFIG-7_FlamePlotsFluxes-NEP_ZSCORE_NEP_ZSCORE.png"),
    ("suppfig8.png", f"{PLOTS}/60_SUPPFIG-8_ThresholdExceedance_NEP_ZSCORE.png"),
    ("suppfig9.png", f"{PLOTS}/68_SUPPFIG-9_SiteCurvesKpa_NEP_ZSCORE.png"),
    ("suppfig10.png", f"{PLOTS}/66_SUPPFIG-10_DataFlow_Sankey.png"),
]


def main(build: bool) -> None:
    missing = [src for _, src in FILES if not (OUTPUTS / src).is_file()]
    for name, src in FILES:
        size = (OUTPUTS / src).stat().st_size / 1e6 if (OUTPUTS / src).is_file() else float("nan")
        print(f"{size:6.2f} MB  {name:14s} {src}")
    for src in missing:
        print("PROBLEM: missing", src)
    if not build:
        print("\nDry run. Add --build to write the figures.")
        return
    if missing:
        sys.exit("Not written: fix the problems above first.")

    DOCS_FIGURES.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, src in FILES:
        with Image.open(OUTPUTS / src) as img:
            if img.width > WIDTH:
                img = img.resize((WIDTH, round(img.height * WIDTH / img.width)), Image.LANCZOS)
            img = img.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                               dither=Image.Dither.NONE)
            img.save(DOCS_FIGURES / name, optimize=True)
        total += (DOCS_FIGURES / name).stat().st_size
    print(f"written {len(FILES)} figures to {DOCS_FIGURES}: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main(build="--build" in sys.argv[1:])
