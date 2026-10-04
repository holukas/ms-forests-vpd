"""
Describe the columns of every output table and check the descriptions against the files.

The descriptions are kept by hand in 96_column_dictionary.yaml, next to this script: a
naming scheme shared by all files (variables, forms such as {VAR}_ZSCORE_SHAPVALS,
statistics) and, per output of 94_data_flow_graph.json, the columns the scheme does not
explain. The script reads the header of every csv and parquet file that the glob of an
output matches and explains each column by the entry of its output, then by the scheme.
Columns that nothing explains and descriptions that match no column are printed, so the
dictionary can be brought up to date.

csv twins of parquet files (same name, other suffix) are skipped in 30_shap and
40_aggregation, where the deposit holds the parquet only. xlsx files are tables laid out
for Word and are not read.

Reads:
    96_column_dictionary.yaml and 94_data_flow_graph.json, next to this script
    91_build_source_data.py, next to this script: the Source Data name of each file
    <DATA_ROOT>/data/outputs/: the header of every matched file

Writes:
    docs/columns.qmd, the Output columns page of the documentation
    with --build also <DATA_ROOT>/deposit_eth_research_collection/COLUMNS.csv, every column
    of every output with its description

Without arguments it writes only the docs page and prints the check.
"""
import ast
import csv
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml

from src.paths import DATA_ROOT, DEPOSIT_DIR, REPO_ROOT

HERE = Path(__file__).parent
DICTIONARY = HERE / "96_column_dictionary.yaml"
GRAPH = HERE / "94_data_flow_graph.json"
OUTPUTS = DATA_ROOT / "data" / "outputs"
DOCS_OUT = REPO_ROOT / "docs" / "columns.qmd"
CSV_OUT = DEPOSIT_DIR / "COLUMNS.csv"
PARQUET_ONLY = ("30_shap/", "40_aggregation/")

STAGE_TITLE = {
    "10_datasets": "10_datasets: site collection", "20_subsets": "20_subsets: model input",
    "30_shap": "30_shap: models and attribution", "40_aggregation": "40_aggregation: results across sites",
    "50_plots": "50_plots: figure and table data", "80_info": "80_info: checks quoted in the paper",
}

d = yaml.safe_load(DICTIONARY.read_text(encoding="utf8"))
graph = json.loads(GRAPH.read_text(encoding="utf8"))
VARIABLES = d["variables"]
STATISTICS = d["statistics"]
TOKENS = {
    "VAR": "|".join(sorted(map(re.escape, VARIABLES), key=len, reverse=True)),
    "STAT": "|".join(sorted(map(re.escape, STATISTICS), key=len, reverse=True)),
    "SITE": r"[A-Z]{2}-[A-Za-z0-9]{3}",
    "IGBP": "ENF|DBF|MF|EBF",
    "N": r"\d+",
    "GRID_X": r"-?\d+(?:\.\d+)?",
    "COL": ".+",
}


def pattern_regex(pattern):
    """Regex for a column pattern: {VAR}, {STAT}, {SITE}, {IGBP} and {N} stand for their sets."""
    parts = re.split(r"(\{[A-Z]+\})", pattern)
    out = ""
    for p in parts:
        token = p[1:-1] if p.startswith("{") and p.endswith("}") else None
        out += f"(?P<{token}>{TOKENS[token]})" if token in TOKENS and token not in out else (
            f"(?:{TOKENS[token]})" if token in TOKENS else re.escape(p))
    return re.compile(out + r"\Z")


def fill(text, m):
    """Put name and unit of the matched variable, and the matched statistic, into a description."""
    g = m.groupdict()
    if "VAR" in g:
        name, unit = VARIABLES[g["VAR"]]
        text = text.replace("{name}", name).replace("{unit}", unit).replace("{VAR}", g["VAR"])
    if "STAT" in g:
        text = text.replace("{stat}", STATISTICS[g["STAT"]])
    return text


def glob_regex(pattern):
    """Regex for a glob relative to data/outputs/, as in script 94."""
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        elif pattern[i] == "[":
            j = pattern.index("]", i)
            out, i = out + pattern[i:j + 1], j + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile(out + r"\Z")


def header(rel):
    f = OUTPUTS / rel
    if f.suffix == ".parquet":
        return list(pq.read_schema(f).names)
    return list(pd.read_csv(f, nrows=0).columns)


FORMS = [(p, pattern_regex(p), text) for p, text in d["forms"].items()]
USED = set()


def explain(col, own):
    """Description of a column, from the entry of its output first, then from the scheme."""
    if col.startswith("("):
        try:
            first, stat = ast.literal_eval(col)
        except (ValueError, SyntaxError):
            first, stat = None, None
        if first is not None and (stat == "" or stat in STATISTICS):
            hit = explain(first, own)
            if hit:
                pat, text = hit
                return f"({pat}, {stat or '-'})", text + (f", {STATISTICS[stat]}" if stat else "")
        return None
    for p, rx, text in own + FORMS:
        m = rx.match(col)
        # {COL} stands for a column the scheme explains
        if m and ("COL" not in m.groupdict() or explain(m["COL"], own)):
            if (p, rx, text) in own:
                USED.add(p)
            return p, fill(text, m)
    # {COL}_{STAT}: a statistic of a column the scheme explains
    for stat in sorted(STATISTICS, key=len, reverse=True):
        if col.endswith("_" + stat):
            hit = explain(col[:-len(stat) - 1], own)
            if hit:
                return hit[0] + "_" + stat, hit[1] + ", " + STATISTICS[stat]
    return None


# Source Data name of each output file
spec = importlib.util.spec_from_file_location("source_data", HERE / "91_build_source_data.py")
source_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_data)
SOURCE_NAME = defaultdict(list)
for name, _, src in source_data.FILES:
    SOURCE_NAME[src].append(name)

disk = sorted(f.relative_to(OUTPUTS).as_posix() for f in OUTPUTS.rglob("*")
              if f.is_file() and f.suffix in (".csv", ".parquet"))
disk_set = set(disk)

products = [p for p in graph["products"] if p.get("glob") and not p.get("group_end")]
entries = d["products"]
problems, rows, sections = [], [], defaultdict(list)
for p in products:
    pid = p["id"]
    e = entries.get(pid)
    globs = [p["glob"]] if isinstance(p["glob"], str) else p["glob"]
    rx = [glob_regex(g) for g in globs]
    files = [f for f in disk if any(x.match(f) for x in rx)]
    files = [f for f in files if not (f.startswith(PARQUET_ONLY) and f.endswith(".csv")
                                       and f[:-4] + ".parquet" in disk_set)]
    if not files:
        continue
    if e is None:
        problems.append(f"no entry: {pid}")
        continue
    own = [(c, pattern_regex(c), text) for c, text in (e.get("columns") or {}).items()]
    seen, unexplained = {}, defaultdict(list)
    USED.clear()
    if not e.get("skip"):
        for f in files:
            for c in header(f):
                hit = explain(c, own)
                if hit is None:
                    unexplained[c].append(f)
                    continue
                seen.setdefault(c, (hit[1], f))
    for c, fs in unexplained.items():
        problems.append(f"unexplained in {pid}: {c} ({len(fs)} files, e.g. {fs[0]})")
    for c, _, _ in own:
        if not e.get("skip") and c not in USED:
            problems.append(f"matches no column in {pid}: {c}")
    for c, (text, f) in seen.items():
        rows.append({"output": pid, "glob": "; ".join(globs), "column": c, "description": text,
                     "example_file": f})
    stage = pid.split("/")[1]
    names = sorted({n for f in files for n in SOURCE_NAME.get(f, [])})
    scheme = sorted({hit for c in seen if not any(r.match(c) for _, r, _ in own)
                     for hit in [explain(c, [])] if hit}, key=lambda h: h[0])
    sections[stage].append((p, e, globs, len(files), names, own, scheme))


def scheme_forms(patterns):
    """Scheme columns of an output, one entry per form with its statistics: `{VAR}_ZSCORE` (mean, sem)."""
    stats = defaultdict(set)
    for pat in patterns:
        if pat.startswith("("):
            base, stat = pat[1:-1].rsplit(", ", 1)
            stats[base].add("" if stat == "-" else stat)
            continue
        for st in sorted(STATISTICS, key=len, reverse=True):
            if pat.endswith("_" + st) and pat[:-len(st) - 1] in d["forms"]:
                stats[pat[:-len(st) - 1]].add(st)
                break
        else:
            stats[pat].add("")
    out = []
    for base in sorted(stats):
        listed = [st for st in STATISTICS if st in stats[base]]
        out.append(f"`{md_cell(base)}`" + (f" ({', '.join(listed)})" if listed else ""))
    return out


def md_cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


lines = [
    "---", 'title: "Output columns"', "---", "",
    "<!-- Written by scripts/90_deposit/96_build_column_reference.py from",
    "     96_column_dictionary.yaml. Edit the dictionary and rerun the script, not this page. -->", "",
    d["intro"].strip(), "",
    "## Naming scheme", "", d["scheme_intro"].strip(), "",
    "### Variables", "", "| Token | Variable | Unit |", "|---|---|---|",
]
lines += [f"| `{k}` | {md_cell(n)} | {md_cell(u)} |" for k, (n, u) in VARIABLES.items()]
lines += ["", "### Forms", "", "| Column | Meaning |", "|---|---|"]
lines += [f"| `{md_cell(k)}` | {md_cell(v.replace('{name}', 'VAR').replace('{unit}', 'unit of VAR'))} |"
          for k, v in d["forms"].items()]
lines += ["", "### Statistics", "", "| Suffix | Meaning |", "|---|---|"]
lines += [f"| `{k}` | {md_cell(v)} |" for k, v in STATISTICS.items()]
for stage in STAGE_TITLE:
    if stage not in sections:
        continue
    lines += ["", f"## {STAGE_TITLE[stage]}"]
    for p, e, globs, n, names, own, scheme in sections[stage]:
        lines += ["", f"### {p['label']}", ""]
        lines += [", ".join(f"`{g}`" for g in globs) + f" ({n} file{'s' if n > 1 else ''})", ""]
        if names:
            lines += ["Source Data: " + ", ".join(f"`{x}`" for x in names) + ".", ""]
        for key in ("row", "note", "skip"):
            if e.get(key):
                lines += [e[key].strip(), ""]
        if own:
            lines += ["| Column | Description |", "|---|---|"]
            lines += [f"| `{md_cell(c)}` | {md_cell(t)} |" for c, _, t in own]
            lines += [""]
        forms = scheme_forms(h[0] for h in scheme)
        if forms:
            lines += ["Columns of the naming scheme: " + ", ".join(forms) + ".", ""]

DOCS_OUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf8")
print(f"written {DOCS_OUT}: {sum(len(v) for v in sections.values())} outputs, {len(rows)} columns")
for t in problems:
    print("check:", t)
print(f"{len(problems)} problems")

if "--build" in sys.argv[1:]:
    with open(CSV_OUT, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["output", "glob", "column", "description", "example_file"])
        w.writeheader()
        w.writerows(rows)
    print(f"written {CSV_OUT}")
