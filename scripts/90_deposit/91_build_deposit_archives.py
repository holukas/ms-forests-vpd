"""
Build the zip archives of the pipeline outputs for the data deposit.

One archive per output stage, with the paths inside each archive relative to
``data/outputs/``, so that unpacking all archives into ``data/outputs/`` restores the
layout every script expects and the pipeline runs from stage 31 onward without the
flux products.

What goes in, per stage:

    10_datasets     the site and variable tables the later stages read (the csv and log
                    files at the top level and the small Google Earth Engine folder);
                    not the merged flux products and not the ERA5 downloads
    20_subsets      everything (per-site subsets as parquet, the subsets tables, plots)
    30_shap         everything except csv files that have a parquet twin
    40_aggregation  everything except csv files that have a parquet twin
    50_plots        everything
    80_info         everything

A csv file is a twin when a file with the same name and the extension .parquet exists
in the same folder; the pipeline reads the parquet, and the csv copy is five to ten
times larger with the same numbers.

The archives follow the rules of the ETH Research Collection: zip with compression
level "store" (parquet is compressed already), no nested archives, ASCII names without
blanks, paths shorter than 200 characters, single files of at most about 10 GB and at
most 50 GB per entry. The script checks the names and the sizes and refuses to build
when a rule is broken.

Usage:
    python scripts/90_deposit/91_build_deposit_archives.py            dry run: list what would be archived
    python scripts/90_deposit/91_build_deposit_archives.py --build    write the archives

Writes, into <DATA_ROOT>/deposit/:
    ms-forests-vpd_outputs_<stage>.zip     one per stage
    MANIFEST.csv                           every archived file: archive, path, bytes, sha256
    ARCHIVES.csv                           every archive: name, files, bytes, sha256
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
    """The files of one stage that go into the deposit."""
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
