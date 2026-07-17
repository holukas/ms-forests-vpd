# MS Forests VPD Analysis

Machine learning analysis of CO₂ penalty impacts on net ecosystem productivity (NEP)
across forest flux sites, with a focus on the vapor pressure deficit (VPD) response.

XGBoost models are trained per FLUXNET site and interpreted with two complementary
methods — **SHAP** (out-of-sample, per-sample feature contributions) and **ALE**
(isolated feature effects) — then aggregated across sites and ecosystem types to
identify VPD response thresholds.

- **Target:** `NEP_ZSCORE` (also ET, GPP, RECO available)
- **Features:** `TA_ZSCORE`, `SWIN_ZSCORE`, `VPD_ZSCORE`, `SWC_ZSCORE`

## Repository layout

| Path | Contents |
|------|----------|
| `scripts/10_datasets/` | Dataset assembly and ERA5 climate (MAT/MAP) aggregation |
| `scripts/20_subsets/`  | Ecological-condition subsets |
| `scripts/30_shap/`     | SHAP calculation (5-fold CV) and ALE validation |
| `scripts/40_aggregation/` | Cross-site SHAP/ALE aggregation (overall, by IGBP, by scenario) |
| `scripts/50_figures/`  | Manuscript figures (response curves, thresholds) |
| `src/`                 | Shared library (models, plotting, stats, I/O, sites) |
| `config/`              | FLUXNET site lists and `settings.yaml` |
| `data/`                | Inputs and generated outputs |

See [`CLAUDE.md`](CLAUDE.md) for detailed script documentation, the ERA5 source-selection
logic, and XGBoost hyperparameters.

## Setup

Dependencies are managed with [Poetry](https://python-poetry.org/) (Python 3.11):

```bash
poetry install
```

## Usage

```bash
# SHAP analysis — out-of-sample via 5-fold CV
python scripts/30_shap/31_shap.py

# ALE validation — single 85/15 split, feature effect curves
python scripts/30_shap/32_validate_shap_methods.py

# ERA5 30-year (1991–2020) MAT/MAP aggregation
python scripts/10_datasets/17_add_era5_info.py

# VPD ALE response curves with consensus thresholds
python scripts/50_figures/55_fig_ale_response_curve_with_context.py
```

## License

GNU General Public License v3.0 — see [`LICENSE`](LICENSE).
