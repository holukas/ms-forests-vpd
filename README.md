# ms-forests-vpd

Code for the analysis of vapour pressure deficit (VPD) and soil water content (SWC)
effects on daytime net ecosystem production (NEP) at forest eddy covariance sites.

One XGBoost model is trained per site. The fitted models are interpreted with SHAP
values, calculated out-of-sample through 5-fold cross-validation, and with ALE curves.
Site results are aggregated with equal weight per site, overall and per IGBP forest
type.

- Target: `NEP_ZSCORE`, half-hourly, daytime, four months of highest GPP per site
- Features: `TA_ZSCORE`, `SWIN_ZSCORE`, `VPD_ZSCORE`, `SWC_ZSCORE`

**Data flow chart.** Every script of the pipeline with the files it reads and writes, from
the flux data to each figure and table, as one interactive page:
[`docs/data_flow.html`](docs/data_flow.html) (download and open it in a browser, or see
<https://holukas.github.io/ms-forests-vpd/data_flow.html> once the repository is
public). The same page is `DATA_FLOW.html` in the data deposit, where it also shows the
archive each file is deposited in.

Full documentation is a Quarto site under [`docs/`](docs/). Preview it locally with:

```powershell
uv sync --group dev
.\preview_docs.ps1
```

Outside Windows, `uv run quarto preview docs`.

Once the repository is public it is published at
<https://holukas.github.io/ms-forests-vpd/>.

## Requirements

- **Python 3.12**, pinned in `.python-version` and enforced by `pyproject.toml`
- [uv](https://docs.astral.sh/uv/) for dependency management

Direct dependencies are pinned in `pyproject.toml`. `uv.lock` fixes every package,
including the transitive ones, and `uv sync --locked` recreates that environment.
`pip install .` installs the direct pins only, not the lock file, so transitive versions
may differ. xgboost 3.0.5 is the version that produced the published results, and fitted
models differ between xgboost versions. shap 0.52.0 and scikit-learn 1.9.0 are newer than
the versions that produced the results (0.48.0 and 1.6.1) and reproduce them exactly.
diive is installed from PyPI at release 0.91.1.

| Package | Version |
|---------|---------|
| `diive` | 0.91.1 |
| `pandas` | 3.0.5 |
| `numpy` | 2.4.6 |
| `scipy` | 1.18.0 |
| `matplotlib` | 3.11.1 |
| `scikit-learn` | 1.9.0 |
| `xgboost` | 3.0.5, see the docs on reproducing the published numbers |
| `shap` | 0.52.0 |
| `pyale` | 1.2.0 |
| `statsmodels` | 0.14.6 |
| `geopandas` | 1.1.4 |
| `pyyaml` | 6.0.3 |
| `openpyxl` | 3.1.5 |

Optional extras: `era5` for the ERA5 download scripts, dev group for JupyterLab and
pytest.

## Install

```bash
git clone https://github.com/holukas/ms-forests-vpd.git
cd ms-forests-vpd
uv sync
```

uv downloads Python 3.12 itself if the system does not have it. To install uv:

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
curl -LsSf https://astral.sh/uv/install.sh | sh              # macOS / Linux
```

Tested on Windows 11 with Python 3.12. No special hardware is needed.
A fresh install with `uv sync --locked` and an empty uv cache takes about three minutes on a standard desktop computer.

## Demo

The data deposit (see Data) contains `ms-forests-vpd_demo.zip`, 10 MB: the model input
of one site, CH-Dav, and the published model output of that site.

1. Create an empty data folder, point `MS_FORESTS_VPD_DATA` at it (see Data), and unpack
   the archive into `<data folder>/data/outputs/`.
2. Run `uv run python scripts/30_shap/31_shap.py`.

The script fits the XGBoost model of the site in five-fold cross-validation and computes
the out-of-sample SHAP values; this took 7 seconds on a 24-thread desktop computer. It
writes `CH-Dav_shap-conditional_NEP_ZSCORE.parquet`, a run log and the cross-validation
results to `data/outputs/30_shap/NEP_ZSCORE/conditional/`. With xgboost 3.0.5 on 24
threads the parquet file equals `expected_output/CH-Dav_shap-conditional_NEP_ZSCORE.parquet`
from the archive; with other thread counts the values differ by up to a few percent
(`docs/installation.qmd`, Reproducibility).

## Data

Source files and analysis outputs are about 185 GB and are not part of this repository.
Point the code at a data folder with an environment variable:

```bash
$env:MS_FORESTS_VPD_DATA = "D:/ms-forests-vpd-data"   # Windows PowerShell
export MS_FORESTS_VPD_DATA=/data/ms-forests-vpd-data  # macOS / Linux
```

or by adding `DATA_ROOT` to `config/settings.local.yaml`. The folder is expected to
contain `data/00_raw/` and `data/outputs/`. See [`docs/data.qmd`](docs/data.qmd) for the
layout and for where the FLUXNET source datasets come from.

The per-site subsets and the aggregated outputs behind every figure and table are
deposited in the ETH Research Collection: <https://doi.org/10.3929/ethz-c-000798579>
[DEPOSIT PENDING: link goes live with the final upload]. Unpack them into
`data/outputs/` to run everything from `30_shap` onward without the raw source files.
The data behind each figure and table are also in the archive
`ms-forests-vpd_SourceData.zip` of the same deposit, one or more files per display item,
named after it (`Fig4_all_sites.csv`, `SupplementaryTable6.xlsx`).

## Running

Paths resolve from the repository root, so scripts can be started from anywhere:

```bash
# per-site subsets and z-scores
uv run python scripts/20_subsets/21_prepare_input_data.py

# XGBoost models and out-of-sample SHAP values (5-fold CV)
uv run python scripts/30_shap/31_shap.py

# ALE curves on the same models
uv run python scripts/30_shap/32_validate_shap_methods.py

# bin per site, aggregate across sites, then draw a figure
uv run python scripts/40_aggregation/41_binagg_per_site.py
uv run python scripts/40_aggregation/42_binagg_across_sites.py
uv run python scripts/50_figures/54_fig-4+supptable-7_vpd_response_curve_shap_agg.py
```

Scripts are numbered in run order within each folder. The two model scripts take hours
over the full site list; everything else is minutes. Display scripts carry the manuscript
item they produce in their name (`fig-4`, `supptable-7`, `suppdata-1`), and their outputs
carry the same token. Script 65 draws a figure that is not in the manuscript. Scripts
under `80_info/` write checks that are quoted in the text or used in supplementary tables
and Supplementary Data 2, and no figure reads them.

## Repository layout

| Path | Contents |
|------|----------|
| `scripts/10_datasets/` | Source dataset assembly, variable coverage, ERA5 climate normals |
| `scripts/20_subsets/` | Per-site analysis subsets and z-scores, records removed per filtering step |
| `scripts/30_shap/` | Model fitting, SHAP values, ALE curves |
| `scripts/40_aggregation/` | Aggregation per site, across sites, by IGBP, by stress stage |
| `scripts/50_figures/` | Manuscript figures, tables and data files, each script named after the items it produces |
| `scripts/80_info/` | Checks quoted in the text or used in supplementary tables, read by no figure |
| `scripts/90_deposit/` | Archives for the data deposit, the data flow chart, the Source Data and the demo data |
| `src/` | Shared code: models, aggregation, plotting, statistics, I/O, paths |
| `config/` | `settings.yaml` and FLUXNET site metadata |
| `data/worldmap/` | Natural Earth country outlines for the site map |
| `docs/` | Quarto documentation sources |

## Licence

GNU General Public License v3.0 or later, see [`LICENSE`](LICENSE).
