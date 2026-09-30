"""
Draw the data flow of the pipeline as one zoomable HTML file for the data deposit.

Every script appears with the files it reads and the files it writes. Each file sits under
the script that writes it, and a line from a file leads to each script that reads it. The
page is self-contained (inline SVG and a little JavaScript): the mouse wheel zooms, dragging
moves the view, a click on a box highlights everything upstream and downstream of it.

Reads:
    92_data_flow_template.html, next to this script: the page with its styles and scripts.
    92_data_flow_graph.json, next to this script. The reads and writes of every script were
    taken from the code by hand and are kept in this file, with the run variants of every
    script and file and, for every output file, a glob relative to data/outputs/. When a
    script starts reading or writing a different file, update its entry here and rerun.
    <DATA_ROOT>/deposit/MANIFEST.csv, if it exists: archive, count and size of the deposited
    files of each output.
    <DATA_ROOT>/data/outputs/: every glob is checked against the files on disk.

Writes:
    <DATA_ROOT>/deposit/DATA_FLOW.html
    docs/data_flow.html, the same page for the documentation site. Written only together
    with the deposit copy, so a test run that sends OUT elsewhere leaves docs/ alone.

The chart shows only what exists: an output with no file on disk is left out. Deposited
files that no glob matches are printed, so the graph can be brought up to date; the page
itself shows no check results and links to nothing outside it.
"""
import csv
import datetime
import html
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.paths import DATA_ROOT, REPO_ROOT

GRAPH = Path(__file__).with_name("92_data_flow_graph.json")
OUT = DATA_ROOT / "deposit" / "DATA_FLOW.html"
DOCS_OUT = REPO_ROOT / "docs" / "data_flow.html"
OUTPUTS = DATA_ROOT / "data" / "outputs"
MANIFEST = DATA_ROOT / "deposit" / "MANIFEST.csv"

# Run variants: folder names of the sensitivity runs and the three other fluxes. The
# descriptions follow the "Run variants" table in docs/pipeline.qmd.
FLUX_TEXT = "attribution of the partitioned flux or of evapotranspiration instead of NEP"
VARIANTS = [
    ("deep-sm", "deep-sm", "deepest usable soil water layer per site"),
    ("blocked-cv", "blocked-cv", "leave-one-year-out instead of a shuffled split"),
    ("no_ta", "no_ta", "air temperature dropped from the predictors"),
    ("no_vpd", "no_vpd", "VPD dropped from the predictors"),
    ("deeper-only", "deeper-only", "SITE_SUBSET: the 128 sites with a soil water layer below layer 1"),
    ("mirrored-stages", "mirrored-stages",
     "stage 44 only, VPD escalates and extreme soil dryness is added last"),
    ("factorial-cells", "factorial-cells",
     "stage 44 only, four soil water classes by four VPD classes, temperature free"),
    ("GPP", "GPP", FLUX_TEXT), ("RECO", "RECO", FLUX_TEXT), ("ET", "ET", FLUX_TEXT),
]
# Folder names that mark a variant in a path under data/outputs/
VARIANT_FOLDER = {v[0]: v[0] for v in VARIANTS[:7]}
VARIANT_FOLDER.update({"GPP_ZSCORE": "GPP", "RECO_ZSCORE": "RECO", "ET_ZSCORE": "ET"})

graph = json.loads(GRAPH.read_text(encoding="utf8"))

# Okabe-Ito colors per stage: fill for files, border for scripts
STAGE_COLOR = {
    "00_raw": "#999999", "10_datasets": "#56B4E9", "20_subsets": "#009E73",
    "30_shap": "#E69F00", "40_aggregation": "#D55E00", "50_figures": "#CC79A7",
    "50_plots": "#CC79A7", "80_info": "#0072B2", "90_deposit": "#F0E442", "deposit": "#F0E442",
}
STAGE_LABEL = {
    "00_raw": "raw inputs", "10_datasets": "10 datasets", "20_subsets": "20 subsets",
    "30_shap": "30 models and SHAP", "40_aggregation": "40 aggregation",
    "50_figures": "50 figure scripts", "50_plots": "50 figures and tables", "80_info": "80 checks (INFO)",
    "90_deposit": "90 deposit", "deposit": "deposit files",
}

nodes = {}
for p in graph["products"]:
    nodes["P:" + p["id"]] = dict(kind="product", **p)
for s in graph["scripts"]:
    nodes["S:" + s["id"]] = dict(kind="script", **s)

edges = []
for s in graph["scripts"]:
    sid = "S:" + s["id"]
    for r in s.get("reads", []):
        if "P:" + r not in nodes:
            nodes["P:" + r] = dict(kind="product", id=r, label=r[-45:], stage="00_raw", note="(not listed)")
        edges.append(("P:" + r, sid))
    for w in s.get("writes", []):
        if "P:" + w not in nodes:
            nodes["P:" + w] = dict(kind="product", id=w, label=w[-45:], stage="40_aggregation", note="(not listed)")
        edges.append((sid, "P:" + w))
edges = sorted(set(edges))

preds, succs = defaultdict(list), defaultdict(list)
for a, b in edges:
    succs[a].append(b)
    preds[b].append(a)


def glob_regex(pattern):
    """Regex for a glob relative to data/outputs/: * and ? stay within a folder, ** spans folders."""
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


def globs_of(d):
    g = d.get("glob")
    return [] if not g else [g] if isinstance(g, str) else list(g)


# Deposit location and disk check. The manifest lists every deposited file, relative to
# data/outputs/; csv twins of parquet files are not deposited and so not in it.
warnings = []


def warn(text):
    """Printed for the maintainer only; the page shows no check results."""
    warnings.append(text)
    print("warning:", text)


manifest = []
if MANIFEST.exists():
    with open(MANIFEST, newline="", encoding="utf8") as f:
        manifest = list(csv.DictReader(f))
disk = None
if OUTPUTS.is_dir():
    disk = []
    for root, _, names in os.walk(OUTPUTS):
        rel = Path(root).relative_to(OUTPUTS).as_posix()
        disk += [n if rel == "." else f"{rel}/{n}" for n in names]
else:
    warn(f"no data folder at {OUTPUTS}, files not checked")

matched = set()
for n, d in nodes.items():
    d.update(archives=[], files=None, bytes=None, on_disk=None, disk_files=None)
    rx = [glob_regex(p) for p in globs_of(d)]
    if not rx:
        continue
    rows = [r for r in manifest if any(x.match(r["path"]) for x in rx)]
    if MANIFEST.exists():
        d.update(archives=sorted({r["archive"] for r in rows}), files=len(rows),
                 bytes=sum(int(r["bytes"]) for r in rows))
    if not d.get("group_end"):          # the stage folders of script 91 match everything
        matched.update(r["path"] for r in rows)
    if disk is None:
        continue
    hits = [p for p in disk if any(x.match(p) for x in rx)]
    d.update(on_disk=bool(hits), disk_files=len(hits))
    if not hits:
        print(f"left out, no file on disk: {d['id']}")
    found = {VARIANT_FOLDER[s] for p in hits for s in p.split("/")[:-1] if s in VARIANT_FOLDER}
    extra = sorted(found - set(d.get("variants", [])))
    if extra and not d.get("group_end"):
        warn(f"variants on disk but not in the graph: {d['id']}: {', '.join(extra)}")

# Run logs are no product of the graph, and Supplementary Table 1 is made by hand; neither is drift
HAND_MADE = ["ExtendedDataTable1_Site_DetailsStats_21_SUBSETS_parquet_vars_stats_subsets.xlsx"]
unmatched = sorted(r["path"] for r in manifest if r["path"] not in matched)
logs = [p for p in unmatched if p.endswith(".log") or p.rsplit("/", 1)[-1].startswith("1_") and p.endswith(".txt")]
hand = [p for p in unmatched if p.rsplit("/", 1)[-1] in HAND_MADE]
unmatched = [p for p in unmatched if p not in logs and p not in hand]
if logs:
    print(f"note: {len(logs)} run logs in the deposit belong to no product")
for p in hand:
    print(f"note: made by hand (Supplementary Table 1): {p}")
for stage, k in sorted(Counter(p.split("/")[0] for p in unmatched).items()):
    warn(f"{k} deposited files in {stage} match no glob")
for p in unmatched[:20]:
    warn(f"no glob matches: {p}")

# Only what exists goes into the chart: drop outputs with no file on disk and their edges
absent = {n for n, d in nodes.items() if d.get("on_disk") is False}
for n in absent:
    del nodes[n]
edges = [(a, b) for a, b in edges if a not in absent and b not in absent]
preds, succs = defaultdict(list), defaultdict(list)
for a, b in edges:
    succs[a].append(b)
    preds[b].append(a)

# Block layout: a file sits in the column of the script that writes it, under that script.
# A script sits one column right of the latest writer of its inputs. Raw inputs are column 0.
writer = {}
for sid in sorted(n for n in nodes if nodes[n]["kind"] == "script"):
    for w in succs[sid]:
        writer.setdefault(w, sid)          # first writer owns the file (13 and 13b, 16a and 16b)
rank = {}


def script_rank(sid, stack=()):
    if sid in rank:
        return rank[sid]
    if sid in stack:
        return 1
    r = 1
    for p in preds[sid]:
        w = writer.get(p)
        if w and w != sid:
            r = max(r, script_rank(w, stack + (sid,)) + 1)
    rank[sid] = r
    return r


for n in nodes:
    if nodes[n]["kind"] == "script":
        script_rank(n)
for n in nodes:
    if nodes[n]["kind"] == "product":
        rank[n] = rank[writer[n]] if n in writer else 0

# The deposit group (stage folders, scripts 91 and 92, deposit files) goes after everything else
end_nodes = [n for n in nodes if nodes[n].get("group_end")]
if end_nodes:
    last = max(r for n, r in rank.items() if n not in end_nodes)
    for n in end_nodes:
        d = nodes[n]
        if d["kind"] == "script":
            rank[n] = last + 2
        else:                               # a written file goes under its script
            rank[n] = last + 2 if n in writer else last + 1

# Blocks: a script with the files it owns; raw files and folder bundles are blocks of their own
blocks = {}
for n in nodes:
    if nodes[n]["kind"] == "script":
        blocks[n] = [n] + sorted(m for m in succs[n] if writer.get(m) == n)
for n in nodes:
    if nodes[n]["kind"] == "product" and n not in writer:
        blocks[n] = [n]
bcols = defaultdict(list)
for k in blocks:
    bcols[rank[k]].append(k)
for r in bcols:
    bcols[r].sort(key=lambda k: (nodes[k].get("stage", ""), nodes[k]["id"]))
maxr = max(bcols)


def layout_positions():
    pos = {}
    for r, ks in bcols.items():
        i = 0
        for k in ks:
            for n in blocks[k]:
                pos[n] = i
                i += 1
    return pos


for sweep in range(10):
    pos = layout_positions()
    order = range(1, maxr + 1) if sweep % 2 == 0 else range(maxr - 1, -1, -1)
    for r in order:
        def bary(k):
            members = blocks[k]
            nb = [m for n in members for m in (preds[n] if sweep % 2 == 0 else succs[n])
                  if m not in members and m in pos]
            return sum(pos[m] for m in nb) / len(nb) if nb else pos[members[0]]
        bcols[r].sort(key=bary)
        pos = layout_positions()

cols = {r: [n for k in ks for n in blocks[k]] for r, ks in bcols.items()}

# Geometry
CHAR_W, FONT = 6.6, 11
PAD_X, NODE_H, GAP_Y, BLOCK_GAP, COL_GAP = 8, 26, 6, 18, 60


def label_of(n):
    d = nodes[n]
    text = d.get("label") or d["id"]
    if d["kind"] == "script" and text.split(" ", 1)[0] == d["id"] and " " in text:
        text = text.split(" ", 1)[1]
    return text


def display_item(n):
    """(icon, badge) for a figure or table file: icon fig, table or data; badge Main or Suppl."""
    d = nodes[n]
    if d["kind"] != "product" or "50_plots/" not in d["id"]:
        return None
    name = d["id"].split("50_plots/")[1]
    if "_" not in name:
        return None
    token = name.split("_", 1)[1].split("-")[0]
    kinds = {"FIG": ("fig", "Main"), "TABLE": ("table", "Main"), "SUPPFIG": ("fig", "Suppl."),
             "SUPPTABLE": ("table", "Suppl."), "SUPPDATA": ("data", "Suppl.")}
    icon, badge = kinds.get(token, (None, None))
    if icon is None:
        return None
    if name.split("_", 1)[1].split("-")[1:2] == ["X"]:
        badge = "unused"
    return icon, badge


def icon_svg(kind, x, y, color):
    """A 14 by 11 icon with its top left corner at x, y, drawn in `color`."""
    st = f'fill="none" style="stroke:{color}" stroke-width="1.2" stroke-linejoin="round"'
    if kind == "fig":            # figure: a line chart in a frame
        return (f'<rect x="{x:.1f}" y="{y:.1f}" width="14" height="11" rx="1.5" {st}/>'
                f'<polyline points="{x + 2:.1f},{y + 8.5:.1f} {x + 5:.1f},{y + 5:.1f} {x + 8:.1f},{y + 7:.1f} '
                f'{x + 12:.1f},{y + 2.5:.1f}" {st}/>')
    if kind == "table":          # table: a grid
        return (f'<rect x="{x:.1f}" y="{y:.1f}" width="14" height="11" rx="1.5" {st}/>'
                f'<path d="M{x:.1f},{y + 3.7:.1f} H{x + 14:.1f} M{x:.1f},{y + 7.3:.1f} H{x + 14:.1f} '
                f'M{x + 5:.1f},{y:.1f} V{y + 11:.1f}" {st}/>')
    if kind == "data":           # supplementary data file: a sheet with lines
        return (f'<rect x="{x:.1f}" y="{y:.1f}" width="14" height="11" rx="1.5" {st}/>'
                f'<path d="M{x + 3:.1f},{y + 3.5:.1f} H{x + 11:.1f} M{x + 3:.1f},{y + 5.5:.1f} H{x + 11:.1f} '
                f'M{x + 3:.1f},{y + 7.5:.1f} H{x + 9:.1f}" {st}/>')
    if kind == "input":          # input data: a database cylinder
        cx = x + 7
        return (f'<ellipse cx="{cx:.1f}" cy="{y + 2:.1f}" rx="5.5" ry="2" {st}/>'
                f'<path d="M{x + 1.5:.1f},{y + 2:.1f} V{y + 9:.1f} A5.5,2 0 0 0 {x + 12.5:.1f},{y + 9:.1f} '
                f'V{y + 2:.1f} M{x + 1.5:.1f},{y + 5.5:.1f} A5.5,2 0 0 0 {x + 12.5:.1f},{y + 5.5:.1f}" {st}/>')
    if kind == "generated":      # generated data file: a page with a folded corner
        return (f'<path d="M{x + 3:.1f},{y:.1f} H{x + 9:.1f} L{x + 12:.1f},{y + 3:.1f} V{y + 11:.1f} '
                f'H{x + 3:.1f} Z M{x + 9:.1f},{y:.1f} V{y + 3:.1f} H{x + 12:.1f}" {st}/>')
    if kind == "analysis":       # analysis script: a gear
        cx, cy = x + 7, y + 5.5
        teeth = " ".join(f"M{cx + 3.6 * c:.1f},{cy + 3.6 * s_:.1f} L{cx + 5.4 * c:.1f},{cy + 5.4 * s_:.1f}"
                         for c, s_ in [(1, 0), (0.707, 0.707), (0, 1), (-0.707, 0.707), (-1, 0),
                                       (-0.707, -0.707), (0, -1), (0.707, -0.707)])
        return (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.6" {st}/><circle cx="{cx:.1f}" cy="{cy:.1f}" r="1.3" {st}/>'
                f'<path d="{teeth}" {st} stroke-width="1.8"/>')
    if kind == "draw":           # figure or table script: a pencil
        return (f'<path d="M{x + 2:.1f},{y + 10:.1f} L{x + 3:.1f},{y + 7:.1f} L{x + 10:.1f},{y:.1f} '
                f'L{x + 13:.1f},{y + 3:.1f} L{x + 6:.1f},{y + 10:.1f} Z M{x + 8.5:.1f},{y + 1.5:.1f} '
                f'L{x + 11.5:.1f},{y + 4.5:.1f}" {st}/>')
    if kind == "package":        # deposit script: a box
        return (f'<path d="M{x + 1:.1f},{y + 3:.1f} L{x + 7:.1f},{y:.1f} L{x + 13:.1f},{y + 3:.1f} V{y + 9:.1f} '
                f'L{x + 7:.1f},{y + 11:.1f} L{x + 1:.1f},{y + 9:.1f} Z M{x + 1:.1f},{y + 3:.1f} L{x + 7:.1f},{y + 5.5:.1f} '
                f'L{x + 13:.1f},{y + 3:.1f} M{x + 7:.1f},{y + 5.5:.1f} V{y + 11:.1f}" {st}/>')
    return ""


def node_icon(n):
    """Icon kind of any node: display item, input or generated data, or the kind of script."""
    d = nodes[n]
    if d["kind"] == "script":
        return {"50_figures": "draw", "90_deposit": "package"}.get(d.get("stage"), "analysis")
    item = display_item(n)
    if item:
        return item[0]
    if d.get("stage") == "00_raw" or d["id"].startswith("scripts/"):
        return "input"
    return "generated"


BADGE_W = {"Main": 34, "Suppl.": 40, "unused": 44}
CIRCLE_R, CIRCLE_GAP = 14, 5
LEFT = 2 * CIRCLE_R + CIRCLE_GAP          # room for the number circle left of the boxes


def width(n):
    text = label_of(n)
    extra = 22 + (BADGE_W[display_item(n)[1]] + 6 if display_item(n) else 0)
    return max(80, min(250 + extra, int(len(text) * CHAR_W) + 2 * PAD_X + extra))


col_w = {r: max(width(n) for n in ns) for r, ns in cols.items()}
x_of_col, x = {}, 20
for r in range(maxr + 1):
    x_of_col[r] = x
    x += col_w.get(r, 100) + LEFT + COL_GAP
total_w = x + 20


def col_height(r):
    return sum(len(blocks[k]) * (NODE_H + GAP_Y) + BLOCK_GAP for k in bcols[r])


total_h = max(col_height(r) for r in bcols) + 80
geom = {}
for r, ks in bcols.items():
    y = 60 + (total_h - 80 - col_height(r)) / 2
    for k in ks:
        for j, n in enumerate(blocks[k]):
            w = width(n)
            indent = 14 if j > 0 else 0          # files indented under their script
            geom[n] = (x_of_col[r] + LEFT + indent, y, min(w, col_w[r] - indent), NODE_H)
            y += NODE_H + GAP_Y
        y += BLOCK_GAP

# Stage lanes: a column takes the most common stage of its scripts, column 0 holds the raw
# inputs and the deposit group has a lane of its own. Neighboring columns of the same stage
# share one band.
end_cols = {rank[n] for n in end_nodes}
col_stage = {}
for r in range(maxr + 1):
    if r == 0:
        col_stage[r] = "00_raw"
    elif r in end_cols:
        col_stage[r] = "90_deposit"
    else:
        count = Counter(nodes[n].get("stage", "") for n in cols.get(r, []) if nodes[n]["kind"] == "script")
        # on a tie the earlier stage wins
        col_stage[r] = min(count, key=lambda s: (-count[s], s)) if count else col_stage[r - 1]
lanes = []
for r in range(maxr + 1):
    x0 = round(max(0, x_of_col[r] - COL_GAP / 2), 1)
    x1 = round(min(total_w, x_of_col[r] + col_w.get(r, 100) + LEFT + COL_GAP / 2), 1)
    if lanes and lanes[-1]["stage"] == col_stage[r]:
        lanes[-1]["x1"] = x1
    else:
        lanes.append({"stage": col_stage[r], "label": STAGE_LABEL.get(col_stage[r], col_stage[r]),
                      "x0": x0, "x1": x1})


def text_color(hex_color):
    """Black or white label text, whichever contrasts more with the fill."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if lum > 0.40 else "#ffffff"


def esc(s):
    return html.escape(str(s), quote=True)


LANE_TEXT = {"#F0E442": "#9c8a00"}           # the yellow is too light for text on a light band
svg = ['<g id="lanes">']
for ln in lanes:
    c = STAGE_COLOR.get(ln["stage"], "#777777")
    svg.append(f'<rect class="lane" x="{ln["x0"]}" y="0" width="{ln["x1"] - ln["x0"]:.1f}" height="{total_h:.0f}" '
               f'style="fill:{c};opacity:.07"/>'
               f'<text class="lanet" x="{ln["x0"] + 10:.1f}" y="30" '
               f'style="fill:{LANE_TEXT.get(c, c)};font-weight:700;font-size:14px;text-anchor:start">'
               f'{esc(ln["label"])}</text>')
svg.append('</g>')
for i, (a, b) in enumerate(edges):
    ax, ay, aw, ah = geom[a]
    bx, by, bw, bh = geom[b]
    if rank[a] == rank[b]:          # script to a file it owns, in the same column
        x1, y1 = ax + 6, ay + ah
        x2, y2 = bx, by + bh / 2
        path = f"M{x1:.1f},{y1:.1f} L{x1:.1f},{y2:.1f} L{x2:.1f},{y2:.1f}"
    else:
        x1, y1 = ax + aw, ay + ah / 2
        x2, y2 = bx - (LEFT if nodes[b]["kind"] == "script" else 0), by + bh / 2
        dx = max(30, (x2 - x1) / 2)
        path = f"M{x1:.1f},{y1:.1f} C{x1 + dx:.1f},{y1:.1f} {x2 - dx:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"
    svg.append(f'<path class="e" data-a="{esc(a)}" data-b="{esc(b)}" d="{path}"/>')
for n, (x0, y0, w, h) in geom.items():
    d = nodes[n]
    color = STAGE_COLOR.get(d.get("stage", ""), "#777777")
    label = label_of(n)
    tip = d.get("path") or d["id"]
    if d.get("note"):
        tip += " | " + d["note"]
    if d["kind"] == "script":
        cx, cy = x0 - CIRCLE_GAP - CIRCLE_R, y0 + h / 2
        shape = (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{CIRCLE_R}" class="num" style="fill:{color}"/>'
                 f'<text x="{cx:.1f}" y="{cy + 4:.1f}" class="numt" style="fill:{text_color(color)}">{esc(d["id"])}</text>'
                 f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w}" height="{h}" rx="3" '
                 f'class="script" style="stroke:{color}"/>')
    else:
        shape = (f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w}" height="{h}" rx="14" '
                 f'class="product" style="fill:{color}"/>')
    fill_text = f' style="fill:{text_color(color)}"' if d["kind"] == "product" else ""
    item = display_item(n)
    ink = text_color(color) if d["kind"] == "product" else "var(--fg)"
    shape += icon_svg(node_icon(n), x0 + 8, y0 + (h - 11) / 2, ink)
    text_x = x0 + 22 + (w - 22) / 2
    if item:
        badge = item[1]
        bw = BADGE_W[badge]
        shape += (f'<rect x="{x0 + w - bw - 6:.1f}" y="{y0 + 6:.1f}" width="{bw}" height="{h - 12}" rx="6" '
                  f'class="badge badge-{badge.rstrip('.').lower()}"/>'
                  f'<text x="{x0 + w - bw / 2 - 6:.1f}" y="{y0 + h / 2 + 3.5:.1f}" class="badget">{badge}</text>')
        text_x = x0 + 22 + (w - 22 - bw - 6) / 2
    svg.append(f'<g class="n {d["kind"]}" data-id="{esc(n)}"><title>{esc(tip)}</title>{shape}'
               f'<text x="{text_x:.1f}" y="{y0 + h / 2 + 4:.1f}"{fill_text}>{esc(label)}</text></g>')

legend_items = [(k, v) for k, v in STAGE_LABEL.items()
                if any(nodes[n].get("stage") == k for n in nodes)]
# Node table for the page script: everything the interactive features need per node
node_data = {}
for n, (x0, y0, w, h) in geom.items():
    d = nodes[n]
    item = display_item(n)
    node_data[n] = {
        "id": d["id"], "kind": d["kind"], "label": label_of(n), "stage": d.get("stage", ""),
        "path": d.get("path", ""), "note": d.get("note", ""), "icon": node_icon(n),
        "badge": item[1] if item else "", "rank": rank[n],
        "x": round(x0, 1), "y": round(y0, 1), "w": w, "h": h,
        "glob": globs_of(d), "archives": d["archives"], "files": d["files"], "bytes": d["bytes"],
        "disk_files": d["disk_files"], "variants": d.get("variants", []),
    }
data = {
    "succs": succs, "preds": preds, "nodes": node_data, "lanes": lanes,
    "variants": [{"id": v, "label": label, "description": text} for v, label, text in VARIANTS],
    "meta": {"built": datetime.date.today().isoformat()},
}

ICON_LEGEND = [("input", "input data"), ("generated", "generated data"), ("fig", "figure"),
               ("table", "table"), ("data", "data file"), ("analysis", "analysis script"),
               ("draw", "figure or table script"), ("package", "deposit script")]
legend = ['<span class="legend">']
legend += [f'<span><i style="background:{STAGE_COLOR[k]}"></i>{esc(v)}</span>' for k, v in legend_items]
legend += ['<span><i style="background:transparent;border:2px solid var(--fg);border-radius:2px"></i>script</span>',
           '<span><i style="background:#aaa"></i>file</span>']
legend += [f'<span><svg width="16" height="13">{icon_svg(k, 1, 1, "currentColor")}</svg>{v}</span>'
           for k, v in ICON_LEGEND]
legend += ['<span><b style="background:#111;color:#fff;border-radius:6px;padding:0 5px;font-size:10px">Main</b>'
           ' main text</span>',
           '<span><b style="background:#fff;color:#111;border:1px solid #999;border-radius:6px;padding:0 5px;'
           'font-size:10px">Suppl.</b> supplement</span>', '</span>']
TEMPLATE = Path(__file__).with_name("92_data_flow_template.html")
page = (TEMPLATE.read_text(encoding="utf8")
        .replace("<!--__LEGEND__-->", "".join(legend))
        .replace("<!--__SVG__-->", "\n".join(svg))
        .replace("/*__DATA__*/null", json.dumps(data))
        .replace("__FONT__", str(FONT))
        .replace("__W__", str(total_w)).replace("__H__", f"{total_h:.0f}"))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(page, encoding="utf8")
print(f"wrote {OUT}: {len(nodes)} nodes, {len(edges)} edges, {maxr + 1} columns, {total_w}x{total_h:.0f}")
if OUT.parent == DATA_ROOT / "deposit":
    DOCS_OUT.write_text(page, encoding="utf8")
    print(f"wrote {DOCS_OUT}")
print(f"{len(warnings)} warnings for the maintainer (not shown on the page)")
