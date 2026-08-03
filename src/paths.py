"""
Locations of the repository and of the analysis data.

The FLUXNET source files and all analysis outputs (~185 GB) are kept outside
this repository. Everything that reads or writes data goes through `DATA_ROOT`,
which is resolved in this order:

1. environment variable ``MS_FORESTS_VPD_DATA``
2. key ``DATA_ROOT`` in ``config/settings.local.yaml`` (not under version control)
3. key ``DATA_ROOT`` in ``config/settings.yaml``

The data folder is expected to contain ``00_raw/`` and ``outputs/``; see
``docs/data.qmd`` for the full layout.

Scripts can be run from anywhere, e.g. from the repository root:

    python scripts/30_shap/31_shap.py
"""

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
SETTINGS_FILE = CONFIG_DIR / "settings.yaml"
SETTINGS_LOCAL_FILE = CONFIG_DIR / "settings.local.yaml"

ENV_VAR = "MS_FORESTS_VPD_DATA"

# Keys whose values are paths and are therefore resolved against DATA_ROOT.
_PATH_KEY_PREFIXES = ("DIR_", "INFOFILE_", "OUTFILE_")


def _read_yaml(filepath: Path) -> dict:
    if not filepath.is_file():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_data_root() -> Path:
    from_env = os.environ.get(ENV_VAR)
    if from_env:
        return Path(from_env).expanduser().resolve()

    for filepath in (SETTINGS_LOCAL_FILE, SETTINGS_FILE):
        data_root = _read_yaml(filepath).get("DATA_ROOT")
        if data_root:
            return Path(data_root).expanduser().resolve()

    raise FileNotFoundError(
        f"No data folder configured. Set the {ENV_VAR} environment variable, or add "
        f"DATA_ROOT to {SETTINGS_LOCAL_FILE}. See docs/data.qmd."
    )


DATA_ROOT = _resolve_data_root()


def data_path(*parts: str) -> Path:
    """Path inside the external data folder, e.g. ``data_path('data/outputs/30_shap')``."""
    return DATA_ROOT.joinpath(*parts)


def repo_path(*parts: str) -> Path:
    """Path inside this repository, for files small enough to ship with the code."""
    return REPO_ROOT.joinpath(*parts)


def load_settings() -> dict:
    """
    Read ``config/settings.yaml``, apply local overrides, resolve data paths.

    Path values are stored relative to the data folder, so a settings entry
    ``outputs/30_shap`` comes back as an absolute path under `DATA_ROOT`.
    Absolute values are left untouched.
    """
    settings = _read_yaml(SETTINGS_FILE)
    settings.update(_read_yaml(SETTINGS_LOCAL_FILE))
    settings["DATA_ROOT"] = str(DATA_ROOT)
    settings["REPO_ROOT"] = str(REPO_ROOT)

    for key, value in settings.items():
        if not key.startswith(_PATH_KEY_PREFIXES) or key == "DIR_DATA_ROOT":
            continue
        if isinstance(value, str) and not Path(value).is_absolute():
            settings[key] = str(DATA_ROOT / value)

    return settings
