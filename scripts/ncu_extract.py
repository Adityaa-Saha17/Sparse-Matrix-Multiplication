#!/usr/bin/env python3
"""Extract ncu GPU Speed-Of-Light summaries from colabRunner.ipynb cell outputs.

Parses every code cell whose output contains an ncu "GPU Speed Of Light
Throughput" section, pulls the throughput percentages and kernel duration,
derives the *measured* DRAM bytes (DRAM% x 320 GB/s x duration) and, where the
harness line is present in the same output, the measured arithmetic intensity
(FLOPs / measured DRAM bytes) versus the compulsory-model AI.

Usage:
    python3 scripts/ncu_extract.py            # writes reports/data/ncu_sol.csv

Reproducible: reads only the committed notebook, no network or GPU needed.
"""

import csv
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(ROOT, "colabRunner.ipynb")
OUT = os.path.join(ROOT, "reports", "data", "ncu_sol.csv")

DRAM_BW_GBs = 320.0  # T4 GDDR6 peak


def cell_text(cell):
    s = ""
    for o in cell.get("outputs", []):
        if "text" in o:
            s += "".join(o["text"])
        elif "data" in o and "text/plain" in o["data"]:
            s += "".join(o["data"]["text/plain"])
    return s


def pct(t, name):
    m = re.search(re.escape(name) + r"\s+%\s+([\d.]+)", t)
    return float(m.group(1)) if m else None


def duration_ms(t):
    m = re.search(r"Duration\s+(us|ms)\s+([\d.,]+)", t)
    if not m:
        return None
    v = float(m.group(2).replace(",", ""))
    return v / 1000.0 if m.group(1) == "us" else v


def harness_line(t):
    m = re.search(
        r"kernel=(\w+) m=(\d+) k=(\d+) n=(\d+) nnz=(\d+) density=([\d.]+)"
        r" time_ms=([\d.]+) gflops=([\d.]+)",
        t,
)
    return m


def compulsory_bytes(m, k, n, nnz):
    # CSR compulsory traffic: vals+cols once, row_ptr, B once, C once.
    return 8 * nnz + 4 * (m + 1) + 4 * k * n + 4 * m * n


def main:
    nb = json.load(open(NOTEBOOK))
    rows = []
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        t = cell_text(cell)
        # cell 61 bundles several profiles; split on the bundle headers too
        chunks = re.split(r"={10}\s+(\S+\.txt)\s+={10}", t)
        # re.split keeps captured names: [pre, name1, body1, name2, body2, ...]
        parts = []
        if len(chunks) > 1:
            for j in range(1, len(chunks), 2):
                parts.append((chunks[j], chunks[j + 1]))
        else:
            parts.append((None, t))
        label = "".join(cell["source"]).strip.splitlines
        label = label[0].lstrip("# ").strip if label else ""
        for name, body in parts:
            if "GPU Speed Of Light" not in body:
                continue
            dram = pct(body, "DRAM Throughput")
            dur = duration_ms(body)
            h = harness_line(body)
            row = {
                "cell": idx,
                "source": name or label[:70],
                "mem_pct": pct(body, "Memory Throughput"),
                "dram_pct": dram,
                "l1tex_pct": pct(body, "L1/TEX Cache Throughput"),
                "l2_pct": pct(body, "L2 Cache Throughput"),
                "sm_pct": pct(body, "Compute (SM) Throughput"),
                "duration_ms": dur,
            }
            if dram is not None and dur is not None:
                row["measured_dram_MB"] = round(
                    dram / 100.0 * DRAM_BW_GBs * 1e9 * dur * 1e-3 / 1e6, 1
)
            if h:
                kern, m_, k_, n_, nnz = h.group(1), *map(int, h.groups[1:5])
                row.update(kernel=kern, m=m_, k=k_, n=n_, nnz=nnz)
                flops = 2.0 * nnz * n_
                cb = compulsory_bytes(m_, k_, n_, nnz)
                row["ai_compulsory"] = round(flops / cb, 2)
                if row.get("measured_dram_MB"):
                    mb = row["measured_dram_MB"] * 1e6
                    row["ai_measured"] = round(flops / mb, 2)
                    row["traffic_ratio"] = round(mb / cb, 1)
            rows.append(row)

    # bundle/print cells repeat earlier profiles verbatim — keep first sighting
    seen, unique = set, []
    for r in rows:
        key = (r["dram_pct"], r["l1tex_pct"], r["l2_pct"], r["duration_ms"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    rows = unique

    cols = [
        "cell", "source", "kernel", "m", "k", "n", "nnz",
        "mem_pct", "dram_pct", "l1tex_pct", "l2_pct", "sm_pct",
        "duration_ms", "measured_dram_MB",
        "ai_compulsory", "ai_measured", "traffic_ratio",
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} profiles)")
    for r in rows:
        print(
            f"  cell {r['cell']:>3} {str(r.get('kernel','?')):>10} "
            f"DRAM {r['dram_pct']}% L1 {r['l1tex_pct']}% L2 {r['l2_pct']}% "
            f"SM {r['sm_pct']}% dur {r['duration_ms']}ms "
            f"-> {r.get('measured_dram_MB','-')} MB"
)


if __name__ == "__main__":
    main
