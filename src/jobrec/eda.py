import html
import math
from collections import Counter
from pathlib import Path


def bar_svg(title, counts, width=960):
    items = sorted(counts.items(), key=lambda pair: (-pair[1], str(pair[0])))[:24]
    height = 70 + len(items) * 28
    maximum = max((value for _, value in items), default=1) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#f8fafc"/>',
             f'<text x="20" y="30" font-family="sans-serif" font-size="18">{html.escape(title)}</text>']
    for index, (label, value) in enumerate(items):
        vertical = 55 + index * 28
        length = (width - 420) * value / maximum
        parts.extend([f'<text x="20" y="{vertical + 15}" font-family="sans-serif" font-size="12">{html.escape(str(label)[:42])}</text>',
                      f'<rect x="330" y="{vertical}" width="{length:.1f}" height="19" fill="#2563eb"/>',
                      f'<text x="{340 + length:.1f}" y="{vertical + 15}" font-family="sans-serif" font-size="12">{value}</text>'])
    return "\n".join(parts + ["</svg>"])


def write_eda(bundle, audit, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    charts = {}
    for name in ("candidates", "jobs"):
        charts[f"{name}_missing"] = (f"{name}: missing counts", audit[name]["missing"])
        skills = Counter(token.strip().lower() for row in bundle[name] for token in row.get("skills", "").split(";")
                         if token.strip() and token.strip() != "_rare_skill_")
        charts[f"{name}_skills"] = (f"{name}: manual skills", skills)
    charts["jobs_per_user"] = ("Unique jobs per candidate", audit["interactions"]["jobs_per_user"])
    charts["stages"] = ("Final recorded stage (EDA only, not input features)", audit["interactions"]["last_stage_counts"])
    charts["timeline"] = ("First-pair events by month (top 24)", Counter(row["time"][:7] for row in bundle["pairs"]))
    for filename, (title, counts) in charts.items():
        (output / f"{filename}.svg").write_text(bar_svg(title, counts), encoding="utf-8")
    body = "\n".join(f'<section><h2>{html.escape(title)}</h2><img src="{filename}.svg" alt="{html.escape(title)}"/></section>'
                     for filename, (title, _) in charts.items())
    (output / "index.html").write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>JTH EDA</title><body><h1>Local JTH audit</h1>{body}</body></html>', encoding="utf-8")
    return len(charts)


def write_graph_svg(graph, path):
    positions = {"candidate": (150, 180), "job": (150, 480)}
    groups = [kind for kind in graph["nodes"] if kind not in positions]
    positions.update({kind: (700, 90 + index * 100) for index, kind in enumerate(groups)})
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="760" viewBox="0 0 960 760">',
             '<rect width="100%" height="100%" fill="#f8fafc"/>',
             '<text x="24" y="30" font-family="sans-serif" font-size="22">JobRec graph schema</text>',
             '<text x="24" y="710" font-family="sans-serif" font-size="13">Lines show forward relations. Every relation also has a reverse edge.</text>',
             '<text x="24" y="735" font-family="sans-serif" font-size="13">Node labels show type and count; edge counts are in graph_summary.json.</text>']
    for source, relation, destination in graph["edge_types"]:
        if relation.startswith("rev_"):
            continue
        left, top = positions[source]
        right, bottom = positions[destination]
        color = "#2563eb" if source == "candidate" else "#059669"
        parts.append(f'<line x1="{left}" y1="{top}" x2="{right}" y2="{bottom}" stroke="{color}" stroke-width="2" opacity="0.5"/>')
    for kind, (horizontal, vertical) in positions.items():
        parts.extend([f'<rect x="{horizontal - 100}" y="{vertical - 25}" width="200" height="50" rx="10" fill="white" stroke="#475569"/>',
                      f'<text x="{horizontal}" y="{vertical - 3}" text-anchor="middle" font-family="sans-serif" font-size="15">{html.escape(kind)}</text>',
                      f'<text x="{horizontal}" y="{vertical + 16}" text-anchor="middle" font-family="sans-serif" font-size="13">{len(graph["nodes"][kind])} nodes</text>'])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts + ["</svg>"]), encoding="utf-8")


def write_learning_curves(history, output_dir):
    output = Path(output_dir)
    for key in ("loss", "validation_ndcg20"):
        values = [row[key] for row in history]
        if not values or not all(math.isfinite(value) for value in values):
            continue
        lower, upper = min(values), max(values)
        span = max(upper - lower, 0.01)
        points = " ".join(f"{70 + index * 780 / max(1, len(values) - 1):.2f},{340 - (value - lower) * 250 / span:.2f}" for index, value in enumerate(values))
        title = html.escape(f"{key}: epochs 1-{len(values)}, min={lower:.5f}, max={upper:.5f}")
        content = (f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="420">'
                   f'<rect width="100%" height="100%" fill="#f8fafc"/><text x="25" y="35" font-family="sans-serif" font-size="18">{title}</text>'
                   '<path d="M70 70 V360 H880" fill="none" stroke="#64748b"/>'
                   f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="3"/>'
                   '<text x="70" y="395" font-family="sans-serif">Epoch</text></svg>')
        (output / f"{key}.svg").write_text(content, encoding="utf-8")
