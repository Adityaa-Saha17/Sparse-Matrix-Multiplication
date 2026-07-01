#!/usr/bin/env python3
"""plot_speedup.py — speedup graph for the SpMM deliverable.

Reads measured benchmark data and plots each kernel's speedup over the CSR
baseline, per matrix size, faceted by sparsity pattern. Speedup is
    gflops(kernel) / gflops(baseline)
within the same cell (same pattern, size, nnz) — identical to
time(baseline)/time(kernel) since the FLOP count is fixed.

Input can be either:
  * a pattern-sweep CSV  (pattern,kernel,m,k,n,nnz,density,time_ms,gflops)
  * a plain sweep CSV    (kernel,m,k,n,nnz,time_ms,gflops)  -> pattern='uniform'
  * a harness log        (lines 'kernel=... m=... gflops=...')

When a cell has several densities for the same (pattern, size) the per-kernel
speedups are aggregated with the median.

Usage:
    python3 scripts/plot_speedup.py --sweep reports/data/pattern_sweep.csv \
        --out reports/figures/speedup_by_pattern.png
    python3 scripts/plot_speedup.py --sweep reports/data/sweep_raw.log \
        --out reports/figures/speedup_by_size.png
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELD = re.compile(r"(\w+)=([\-\d.eE+]+|\w+\[?\w*\]?)")

# Kernels shown by default (kept small so the bars stay readable).
DEFAULT_KERNELS = ["baseline", "memopt", "tiled", "tiled_v3", "tiled_v4", "wmma"]


def load_rows(path):
    """Return list of dicts with pattern,kernel,m,nnz,gflops from CSV or log."""
    rows = []
    text = open(path).read()
    is_csv = path.endswith(".csv") and "kernel=" not in text[:200]
    if is_csv:
        for d in csv.DictReader(text.splitlines()):
            try:
                rows.append({
                    "pattern": d.get("pattern", "uniform") or "uniform",
                    "kernel": d["kernel"],
                    "m": int(d["m"]), "nnz": int(d["nnz"]),
                    "gflops": float(d["gflops"]),
                })
            except (KeyError, ValueError):
                continue
    else:
        for line in text.splitlines():
            if not line.strip().startswith("kernel="):
                continue
            d = dict(FIELD.findall(line))
            try:
                rows.append({
                    "pattern": "uniform",
                    "kernel": d["kernel"],
                    "m": int(d["m"]), "nnz": int(d["nnz"]),
                    "gflops": float(d["gflops"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def speedups(rows, kernels):
    """Nested dict: pattern -> kernel -> {size: median_speedup}."""
    # group by (pattern, m, nnz) -> {kernel: gflops}
    cells = {}
    for r in rows:
        cells.setdefault((r["pattern"], r["m"], r["nnz"]), {})[r["kernel"]] = r["gflops"]

    # per (pattern, m): list of speedups across nnz variants
    acc = {}  # (pattern, kernel, m) -> [speedups]
    for (pattern, m, _nnz), kg in cells.items():
        base = kg.get("baseline")
        if not base:
            continue
        for kname, g in kg.items():
            key = kname
            # normalise hybrid[...] label
            if kname.startswith("hybrid"):
                key = "hybrid"
            if key not in kernels:
                continue
            acc.setdefault((pattern, key, m), []).append(g / base)

    out = {}
    for (pattern, kname, m), vals in acc.items():
        out.setdefault(pattern, {}).setdefault(kname, {})[m] = statistics.median(vals)
    return out


def plot(data, kernels, out_path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available; skipping plot", file=sys.stderr)
        return False

    patterns = [p for p in ["uniform", "banded", "block-diagonal", "power-law"]
                if p in data] or sorted(data)
    sizes = sorted({m for p in data.values() for k in p.values() for m in k})
    kernels_present = [k for k in kernels
                       if any(k in data[p] for p in patterns)]

    ncols = len(patterns)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    x = np.arange(len(sizes))
    width = 0.8 / max(1, len(kernels_present))
    cmap = plt.get_cmap("tab10")

    for ax, pattern in zip(axes, patterns):
        for j, kname in enumerate(kernels_present):
            heights = [data[pattern].get(kname, {}).get(m, 0.0) for m in sizes]
            ax.bar(x + j * width - 0.4 + width / 2, heights, width,
                   label=kname, color=cmap(j % 10))
        ax.axhline(1.0, color="black", lw=1, ls="--", alpha=0.7)
        ax.set_title(pattern)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m//1024}k" if m >= 1024 else str(m) for m in sizes])
        ax.set_xlabel("matrix size (m=k)")
        ax.grid(True, axis="y", ls=":", alpha=0.4)
    axes[0].set_ylabel("speedup over CSR baseline (x)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(kernels_present),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(title, y=1.12, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Wrote {out_path}")
    return True


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default="reports/data/pattern_sweep.csv",
                   help="pattern-sweep CSV, plain sweep CSV, or harness log")
    p.add_argument("--out", default="reports/figures/speedup_by_pattern.png")
    p.add_argument("--kernels", nargs="+", default=DEFAULT_KERNELS)
    p.add_argument("--title", default="SpMM speedup over CSR baseline")
    args = p.parse_args()

    sweep = args.sweep if os.path.isabs(args.sweep) else os.path.join(ROOT, args.sweep)
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    if not os.path.exists(sweep):
        print(f"[error] input not found: {sweep}", file=sys.stderr)
        return 1

    rows = load_rows(sweep)
    if not rows:
        print(f"[error] no rows parsed from {sweep}", file=sys.stderr)
        return 1
    data = speedups(rows, args.kernels)
    if not data:
        print("[error] no baseline rows found; cannot compute speedup",
              file=sys.stderr)
        return 1
    plot(data, args.kernels, out, args.title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
