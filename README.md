# ms-forests-vpd

Code for the analysis of vapour pressure deficit (VPD) and soil water content (SWC)
effects on daytime net ecosystem productivity (NEP) at forest eddy covariance sites.

One XGBoost model is trained per site. The fitted models are interpreted with SHAP
values, calculated out-of-sample through 5-fold cross-validation, and with ALE curves.
Site results are aggregated with equal weight per site, overall and per IGBP forest
type.

- Target: `NEP_ZSCORE`, half-hourly, daytime, four months of highest GPP per site
- Features: `TA_ZSCORE`, `SWIN_ZSCORE`, `VPD_ZSCORE`, `SWC_ZSCORE`

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

Direct dependencies, pinned in `pyproject.toml` to the versions the published results
were produced with, so that `pip install .` and `uv sync` give the same environment.
Transitive versions are in `uv.lock`.

| Package | Version |
|---------|---------|
| `diive` | 0.91.0 |
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
carry the same token; scripts under `80_info/` write checks that are quoted in the text
and read by no figure.

## Repository layout

| Path | Contents |
|------|----------|
| `scripts/10_datasets/` | Source dataset assembly, variable coverage, ERA5 climate normals |
| `scripts/20_subsets/` | Per-site analysis subsets and z-scores |
| `scripts/30_shap/` | Model fitting, SHAP values, ALE curves |
| `scripts/40_aggregation/` | Aggregation per site, across sites, by IGBP, by stress stage |
| `scripts/50_figures/` | Manuscript figures, tables and data files, one script per display item |
| `scripts/80_info/` | Checks whose numbers are quoted in the text, read by no figure |
| `src/` | Shared code: models, aggregation, plotting, statistics, I/O, paths |
| `config/` | `settings.yaml` and FLUXNET site metadata |
| `data/worldmap/` | Natural Earth country outlines for the site map |
| `docs/` | Quarto documentation sources |

## Licence

GNU General Public License v3.0 or later, see [`LICENSE`](LICENSE).
