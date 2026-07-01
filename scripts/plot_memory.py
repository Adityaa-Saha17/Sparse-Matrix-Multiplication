#!/usr/bin/env python3
"""plot_memory.py — memory-utilization graph for the SpMM deliverable.

Two panels:

  (left)  Achieved memory-bandwidth utilization — for the best CSR kernel in
          each (size, density) cell, the effective DRAM bandwidth as a percent
          of the T4's 320 GB/s peak. Computed from reports/data/roofline.csv,
          which roofline.py regenerates from live benchmark data, so this panel
          always reflects the most recent run.

  (right) Nsight Compute Speed-of-Light — measured DRAM%, L1/TEX%, and SM%
          utilization for the profiled kernels, from reports/data/ncu_sol.csv.
          Rendered only if that CSV has profiled rows (ncu is optional and not
          available in every notebook environment).

Usage:
    python3 scripts/plot_memory.py \
        --roofline reports/data/roofline.csv \
        --ncu reports/data/ncu_sol.csv \
        --out reports/figures/memory_utilization.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_roofline(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for d in csv.DictReader(open(path).read().splitlines()):
        try:
            rows.append({
                "kernel": d["kernel"], "m": int(d["m"]), "nnz": int(d["nnz"]),
                "gflops": float(d["gflops"]),
                "pct_dram_bw": float(d["pct_dram_bw"]),
                "pct_roofline": float(d["pct_roofline"]),
            })
        except (KeyError, ValueError):
            continue
    return rows


def best_per_cell(rows):
    """Best non-wmma kernel per (m, nnz), sorted by size then density."""
    cells = {}
    for r in rows:
        if r["kernel"] == "wmma":
            continue
        key = (r["m"], r["nnz"])
        if key not in cells or r["gflops"] > cells[key]["gflops"]:
            cells[key] = r
    return sorted(cells.values(), key=lambda r: (r["m"], -r["nnz"]))


def load_ncu(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for d in csv.DictReader(open(path).read().splitlines()):
        if not d.get("kernel") or not d.get("m"):
            continue  # skip rows without a resolved kernel/size
        try:
            rows.append({
                "kernel": d["kernel"], "m": int(d["m"]), "nnz": int(d["nnz"]),
                "dram_pct": float(d["dram_pct"]),
                "l1tex_pct": float(d["l1tex_pct"]),
                "sm_pct": float(d["sm_pct"]),
            })
        except (KeyError, ValueError):
            continue
    # de-duplicate on (kernel, m, nnz), keep first
    seen, uniq = set(), []
    for r in rows:
        key = (r["kernel"], r["m"], r["nnz"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return uniq


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roofline", default="reports/data/roofline.csv")
    p.add_argument("--ncu", default="reports/data/ncu_sol.csv")
    p.add_argument("--out", default="reports/figures/memory_utilization.png")
    args = p.parse_args()

    def abspath(x):
        return x if os.path.isabs(x) else os.path.join(ROOT, x)

    roof = best_per_cell(load_roofline(abspath(args.roofline)))
    ncu = load_ncu(abspath(args.ncu))
    if not roof and not ncu:
        print("[error] no roofline or ncu data available", file=sys.stderr)
        return 1

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available; skipping plot", file=sys.stderr)
        return 1

    npanel = (1 if roof else 0) + (1 if ncu else 0)
    fig, axes = plt.subplots(1, npanel, figsize=(6.4 * npanel, 4.8),
                             squeeze=False)
    axcol = list(axes[0])

    if roof:
        ax = axcol.pop(0)
        labels = [f"m={r['m']//1024}k\nd={r['nnz']/(r['m']*r['m']):.0e}" for r in roof]
        vals = [r["pct_dram_bw"] for r in roof]
        x = np.arange(len(roof))
        ax.bar(x, vals, color="#3b7dd8")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("% of T4 peak DRAM bandwidth (320 GB/s)")
        ax.set_title("Achieved memory-bandwidth utilization\n(best CSR kernel per cell)")
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=7)

    if ncu:
        ax = axcol.pop(0)
        ncu = sorted(ncu, key=lambda r: (r["m"], r["kernel"]))
        cats = [f"{r['kernel']}\nm={r['m']//1024}k" for r in ncu]
        x = np.arange(len(ncu))
        width = 0.26
        ax.bar(x - width, [r["dram_pct"] for r in ncu], width,
               label="DRAM %", color="#3b7dd8")
        ax.bar(x, [r["l1tex_pct"] for r in ncu], width,
               label="L1/TEX %", color="#e0902b")
        ax.bar(x + width, [r["sm_pct"] for r in ncu], width,
               label="SM %", color="#4aa564")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=7, rotation=0)
        ax.set_ylabel("% utilization (Nsight Compute Speed-of-Light)")
        ax.set_title("Measured memory / compute utilization")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", ls=":", alpha=0.4)

    fig.suptitle("SpMM memory utilization on T4", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
