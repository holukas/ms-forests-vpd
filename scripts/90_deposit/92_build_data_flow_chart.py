"""
Draw the data flow of the pipeline as one zoomable HTML file for the data deposit.

Every script appears with the files it reads and the files it writes. Each file sits under
the script that writes it, and a line from a file leads to each script that reads it. The
page is self-contained (inline SVG and a little JavaScript): the mouse wheel zooms, dragging
moves the view, a click on a box highlights everything upstream and downstream of it.

Reads:
    92_data_flow_graph.json, next to this script. The reads and writes of every script were
    taken from the code by hand and are kept in this file. When a script starts reading or
    writing a different file, update its entry here and rerun.

Writes:
    <DATA_ROOT>/deposit/DATA_FLOW.html
"""
import html
import json
from collections import defaultdict
from pathlib import Path

from src.paths import DATA_ROOT

GRAPH = Path(__file__).with_name("92_data_flow_graph.json")
OUT = DATA_ROOT / "deposit" / "DATA_FLOW.html"

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


def text_color(hex_color):
    """Black or white label text, whichever contrasts more with the fill."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if lum > 0.40 else "#ffffff"


def esc(s):
    return html.escape(str(s), quote=True)


svg = []
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
data = {"succs": succs, "preds": preds}

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ms-forests-vpd data flow</title>
<style>
:root {{ --bg:#ffffff; --fg:#1a1a1a; --muted:#666; --edge:#b9b9b9; --script:#ffffff; --hl:#111; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16181c; --fg:#e8e8e8; --muted:#9aa; --edge:#4a4f57; --script:#23262c; --hl:#fff; }} }}
html,body {{ margin:0; height:100%; background:var(--bg); color:var(--fg); font:13px system-ui,-apple-system,Segoe UI,sans-serif; }}
header {{ position:fixed; top:0; left:0; right:0; padding:8px 12px; background:var(--bg); border-bottom:1px solid var(--edge); z-index:2; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }}
header h1 {{ font-size:14px; margin:0 12px 0 0; }}
header button {{ font:inherit; padding:2px 9px; cursor:pointer; }}
header input {{ font:inherit; padding:2px 6px; width:170px; }}
.legend span {{ display:inline-flex; align-items:center; gap:4px; margin-right:10px; color:var(--muted); }}
.legend i {{ width:12px; height:12px; border-radius:6px; display:inline-block; }}
.hint {{ color:var(--muted); font-size:12px; }}
svg {{ position:fixed; top:0; left:0; width:100%; height:100%; cursor:grab; }}
svg.drag {{ cursor:grabbing; }}
.e {{ fill:none; stroke:var(--edge); stroke-width:1.2; }}
.n text {{ font-size:{FONT}px; text-anchor:middle; fill:var(--fg); pointer-events:none; }}
rect.script {{ fill:var(--script); stroke-width:2.2; }}
.numt {{ font-size:11px; font-weight:700; text-anchor:middle; pointer-events:none; }}
.badge {{ fill:#ffffff; opacity:.92; }} .badge-main {{ fill:#111111; }} .badge-unused {{ fill:#ffffff; opacity:.6; }}
.n text.badget {{ font-size:9px; font-weight:700; fill:#111111; pointer-events:none; }}
.badge-main + text.badget {{ fill:#ffffff; }}
.legend svg {{ position:static; width:auto; height:auto; cursor:default; vertical-align:middle; }}
rect.product {{ stroke:none; opacity:.85; }}
.dim .e {{ opacity:.12; }} .dim .n {{ opacity:.18; }}
.dim .e.on {{ opacity:1; stroke:var(--hl); stroke-width:2; }} .dim .n.on {{ opacity:1; }}
.n {{ cursor:pointer; }}
</style></head><body>
<header>
<h1>Data flow: scripts and the files they read and write</h1>
<button id="zin">+</button><button id="zout">&minus;</button><button id="fit">Fit</button>
<input id="q" placeholder="Search a script or file">
<span class="legend">{''.join(f'<span><i style="background:{STAGE_COLOR[k]}"></i>{esc(v)}</span>' for k, v in legend_items)}
<span><i style="background:transparent;border:2px solid var(--fg);border-radius:2px"></i>script</span>
<span><i style="background:#aaa"></i>file</span><span><svg width="16" height="13">{icon_svg("input", 1, 1, "currentColor")}</svg>input data</span><span><svg width="16" height="13">{icon_svg("generated", 1, 1, "currentColor")}</svg>generated data</span><span><svg width="16" height="13">{icon_svg("fig", 1, 1, "currentColor")}</svg>figure</span><span><svg width="16" height="13">{icon_svg("table", 1, 1, "currentColor")}</svg>table</span><span><svg width="16" height="13">{icon_svg("data", 1, 1, "currentColor")}</svg>data file</span><span><svg width="16" height="13">{icon_svg("analysis", 1, 1, "currentColor")}</svg>analysis script</span><span><svg width="16" height="13">{icon_svg("draw", 1, 1, "currentColor")}</svg>figure or table script</span><span><svg width="16" height="13">{icon_svg("package", 1, 1, "currentColor")}</svg>deposit script</span><span><b style="background:#111;color:#fff;border-radius:6px;padding:0 5px;font-size:10px">Main</b> main text</span><span><b style="background:#fff;color:#111;border:1px solid #999;border-radius:6px;padding:0 5px;font-size:10px">Suppl.</b> supplement</span></span>
<span class="hint">Each file sits under the script that writes it; a line from a file leads to a script that reads it. Wheel: zoom. Drag: pan. Hover: direct links. Click: full lineage. Esc: clear. Hover a box for the file path.</span>
</header>
<svg id="c" viewBox="0 0 {total_w} {total_h}" xmlns="http://www.w3.org/2000/svg"><g id="vp">
{chr(10).join(svg)}
</g></svg>
<script>
const G = {json.dumps(data)};
const svg = document.getElementById('c'), vp = document.getElementById('vp');
const W = {total_w}, H = {total_h};
let vb = {{x:0, y:0, w:W, h:H}};
function apply() {{ svg.setAttribute('viewBox', `${{vb.x}} ${{vb.y}} ${{vb.w}} ${{vb.h}}`); }}
function fit() {{
  const r = svg.getBoundingClientRect(), top = 90 * W / r.width;
  const s = Math.max(W / r.width, (H + top) / r.height);
  vb = {{w: r.width * s, h: r.height * s, x: 0, y: -top}}; vb.x = (W - vb.w) / 2; apply();
}}
function zoom(f, cx, cy) {{
  const r = svg.getBoundingClientRect();
  const px = vb.x + (cx - r.left) / r.width * vb.w, py = vb.y + (cy - r.top) / r.height * vb.h;
  vb.w *= f; vb.h *= f; vb.x = px - (cx - r.left) / r.width * vb.w; vb.y = py - (cy - r.top) / r.height * vb.h; apply();
}}
svg.addEventListener('wheel', e => {{ e.preventDefault(); zoom(e.deltaY > 0 ? 1.15 : 1/1.15, e.clientX, e.clientY); }}, {{passive:false}});
let drag = null;
svg.addEventListener('pointerdown', e => {{ drag = {{x:e.clientX, y:e.clientY, vx:vb.x, vy:vb.y, moved:false}}; svg.classList.add('drag'); }});
window.addEventListener('pointermove', e => {{
  if (!drag) return; const r = svg.getBoundingClientRect();
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
  vb.x = drag.vx - dx / r.width * vb.w; vb.y = drag.vy - dy / r.height * vb.h; apply();
}});
window.addEventListener('pointerup', () => {{ setTimeout(() => drag = null, 0); svg.classList.remove('drag'); }});
document.getElementById('zin').onclick = () => {{ const r = svg.getBoundingClientRect(); zoom(1/1.4, r.left + r.width/2, r.top + r.height/2); }};
document.getElementById('zout').onclick = () => {{ const r = svg.getBoundingClientRect(); zoom(1.4, r.left + r.width/2, r.top + r.height/2); }};
document.getElementById('fit').onclick = fit;
const nodesEl = [...document.querySelectorAll('.n')], edgesEl = [...document.querySelectorAll('.e')];
let pinned = null;
function walk(start, map) {{ const seen = new Set([start]), st = [start];
  while (st.length) {{ const n = st.pop(); for (const m of (map[n] || [])) if (!seen.has(m)) {{ seen.add(m); st.push(m); }} }} return seen; }}
function show(set) {{
  vp.classList.toggle('dim', !!set);
  nodesEl.forEach(el => el.classList.toggle('on', !!set && set.has(el.dataset.id)));
  edgesEl.forEach(el => el.classList.toggle('on', !!set && set.has(el.dataset.a) && set.has(el.dataset.b)));
}}
function direct(id) {{ return new Set([id, ...(G.preds[id] || []), ...(G.succs[id] || [])]); }}
function lineage(id) {{ return new Set([...walk(id, G.preds), ...walk(id, G.succs)]); }}
nodesEl.forEach(el => {{
  el.addEventListener('mouseenter', () => {{ if (!pinned) show(direct(el.dataset.id)); }});
  el.addEventListener('mouseleave', () => {{ if (!pinned) show(null); }});
  el.addEventListener('click', () => {{ if (drag && drag.moved) return; pinned = el.dataset.id; show(lineage(pinned)); }});
}});
window.addEventListener('keydown', e => {{ if (e.key === 'Escape') {{ pinned = null; show(null); }} }});
document.getElementById('q').addEventListener('input', e => {{
  const q = e.target.value.trim().toLowerCase(); if (!q) {{ show(pinned ? lineage(pinned) : null); return; }}
  show(new Set(nodesEl.filter(el => (el.dataset.id + ' ' + el.textContent).toLowerCase().includes(q)).map(el => el.dataset.id)));
}});
window.addEventListener('resize', fit); fit();
</script></body></html>
"""
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(page, encoding="utf8")
print(f"wrote {OUT}: {len(nodes)} nodes, {len(edges)} edges, {maxr + 1} columns, {total_w}x{total_h:.0f}")
