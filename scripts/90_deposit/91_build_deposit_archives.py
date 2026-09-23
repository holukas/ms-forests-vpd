"""
Build the zip archives of the pipeline outputs for the data deposit.

One uncompressed zip per output stage, with paths relative to ``data/outputs/``.
Unpacked there, the archives let the pipeline run from stage 31 onward
without the flux products.

Contents per stage:
- 10_datasets: the top-level csv and log files of scripts 11-17 and the Google
  Earth Engine ERA5 folder
- 30_shap, 40_aggregation: everything except csv files with a parquet file of
  the same name in the same folder
- 20_subsets, 50_plots, 80_info: everything

It does not build if a file breaks the ETH Research Collection limits (ASCII
names without blanks, paths under 200 characters, files up to 10 GB, 50 GB in
total).

Without arguments it is a dry run. With ``--build`` it writes into
<DATA_ROOT>/deposit/ the archives ms-forests-vpd_outputs_<stage>.zip
(replacing existing ones), MANIFEST.csv (sha256 per file) and ARCHIVES.csv.
"""
import csv
import hashlib
import re
import sys
import zipfile
from pathlib import Path

from src.paths import DATA_ROOT

OUTPUTS = DATA_ROOT / "data" / "outputs"
DEPOSIT = DATA_ROOT / "deposit"
PREFIX = "ms-forests-vpd_outputs_"

STAGES = ["10_datasets", "20_subsets", "30_shap", "40_aggregation", "50_plots", "80_info"]

# 10_datasets: only these top-level entries (files or folders)
RAW_KEEP = re.compile(r"^(1[1-7][a-z]?_.*\.(csv|log)|16_ERA5_climate_1991-2020_GoogleEarthEngine)$")

MAX_FILE_BYTES = 10 * 1024**3
MAX_ENTRY_BYTES = 50 * 1024**3
MAX_PATH_LEN = 200
NAME_OK = re.compile(r"^[A-Za-z0-9!$&'()+,\-.;=@_/]+$")


def select(stage: str) -> list[Path]:
    """Return the files of one stage that go into the deposit."""
    root = OUTPUTS / stage
    if stage == "10_datasets":
        files = []
        for entry in sorted(root.iterdir()):
            if not RAW_KEEP.match(entry.name):
                continue
            files += [entry] if entry.is_file() else sorted(p for p in entry.rglob("*") if p.is_file())
        return files
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if stage in ("30_shap", "40_aggregation"):
        files = [p for p in files if not (p.suffix == ".csv" and p.with_suffix(".parquet").exists())]
    return files


def check(files: list[Path]) -> list[str]:
    problems = []
    for p in files:
        rel = p.relative_to(OUTPUTS).as_posix()
        if not NAME_OK.match(rel):
            problems.append(f"name breaks the character rule: {rel}")
        if len(rel) > MAX_PATH_LEN:
            problems.append(f"path longer than {MAX_PATH_LEN}: {rel}")
        if p.stat().st_size > MAX_FILE_BYTES:
            problems.append(f"file larger than 10 GB: {rel}")
    return problems


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main(build: bool) -> None:
    plan = {stage: select(stage) for stage in STAGES}
    problems = [msg for files in plan.values() for msg in check(files)]
    total = 0
    print(f"{'archive':<48} {'files':>6} {'GB':>8}")
    for stage, files in plan.items():
        size = sum(p.stat().st_size for p in files)
        total += size
        print(f"{PREFIX + stage + '.zip':<48} {len(files):>6} {size / 1024**3:>8.2f}")
    print(f"{'total':<48} {sum(len(f) for f in plan.values()):>6} {total / 1024**3:>8.2f}")
    if total > MAX_ENTRY_BYTES:
        problems.append(f"entry larger than 50 GB: {total / 1024**3:.1f} GB")
    for msg in problems:
        print("PROBLEM:", msg)
    if not build:
        print("\nDry run. Add --build to write the archives.")
        return
    if problems:
        sys.exit("Not built: fix the problems above first.")

    DEPOSIT.mkdir(parents=True, exist_ok=True)
    manifest_rows, archive_rows = [], []
    for stage, files in plan.items():
        out = DEPOSIT / f"{PREFIX}{stage}.zip"
        if out.exists():
            out.unlink()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for p in files:
                rel = p.relative_to(OUTPUTS).as_posix()
                z.write(p, arcname=rel)
                manifest_rows.append([out.name, rel, p.stat().st_size, sha256(p)])
        archive_rows.append([out.name, len(files), out.stat().st_size, sha256(out)])
        print(f"written {out.name}: {len(files)} files, {out.stat().st_size / 1024**3:.2f} GB")

    with open(DEPOSIT / "MANIFEST.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["archive", "path", "bytes", "sha256"])
        w.writerows(manifest_rows)
    with open(DEPOSIT / "ARCHIVES.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["archive", "files", "bytes", "sha256"])
        w.writerows(archive_rows)
    print(f"manifest and archive list written to {DEPOSIT}")


if __name__ == "__main__":
    main(build="--build" in sys.argv[1:])
