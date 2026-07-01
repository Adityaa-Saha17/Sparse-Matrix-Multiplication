#!/usr/bin/env python3
"""run_pattern_sweep.py — performance evaluation across sparsity patterns and sizes.

Generates a sparse matrix for each (pattern, size) combination, runs the whole
kernel set through the benchmark harness (spmm_bench --kernel all --bin ...),
parses the harness output, and writes one tidy CSV with a `pattern` column:

    reports/data/pattern_sweep.csv
    pattern,kernel,m,k,n,nnz,density,time_ms,gflops

This is the data behind the cross-pattern speedup graph. It needs a built
`spmm_bench` and a GPU, so it is meant to be run inside the Kaggle/Colab
notebook (colabRunner.ipynb, the pattern-sweep section).

Patterns (from scripts/gen_matrices.py):
    uniform        — Bernoulli(density); the random-sparsity reference.
    banded         — nonzeros within +/- bandwidth of the diagonal.
    block-diagonal — dense 16x16 blocks (fill-in ~1; the Tensor-Core best case).
    power-law      — graph-like row-degree distribution.

Per-pattern generator parameters are fixed across sizes so the pattern (not the
density) is the variable; the actual nnz/density is recorded per row so speedups
stay exact.

Usage (from the repo root, after `make`):
    python3 scripts/run_pattern_sweep.py
    python3 scripts/run_pattern_sweep.py --sizes 4096 8192 --patterns banded power-law
    python3 scripts/run_pattern_sweep.py --bench ./build/spmm_bench --n 256 --iters 20
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIELD = re.compile(r"(\w+)=([\-\d.eE+]+|\w+\[?\w*\]?)")

# Fixed generator parameters per pattern (size-independent so the *pattern* is
# the variable being studied). Uniform uses a density; the rest use structural
# parameters. Tweak here if a different operating point is wanted.
PATTERN_PARAMS = {
    "uniform":        ["--density", "0.01"],
    "banded":         ["--bandwidth", "32"],
    "block-diagonal": ["--block", "16"],
    "power-law":      ["--mean-deg", "32", "--alpha", "2.5"],
}


def parse_harness_lines(stdout: str):
    """Yield dicts for every 'kernel=...' line the harness prints."""
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("kernel="):
            continue
        d = dict(FIELD.findall(line))
        try:
            yield {
                "kernel": d["kernel"],
                "m": int(d["m"]), "k": int(d["k"]), "n": int(d["n"]),
                "nnz": int(d["nnz"]),
                "density": float(d.get("density", "nan")),
                "time_ms": float(d["time_ms"]),
                "gflops": float(d["gflops"]),
            }
        except (KeyError, ValueError):
            continue


def sh(cmd, cwd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", default="./build/spmm_bench",
                   help="path to the benchmark binary (default ./build/spmm_bench)")
    p.add_argument("--gen", default="scripts/gen_matrices.py",
                   help="path to the matrix generator")
    p.add_argument("--out", default="reports/data/pattern_sweep.csv")
    p.add_argument("--datadir", default="data",
                   help="scratch dir for generated .bin matrices")
    p.add_argument("--patterns", nargs="+",
                   default=["uniform", "banded", "block-diagonal", "power-law"])
    p.add_argument("--sizes", nargs="+", type=int,
                   default=[1024, 4096, 8192, 16384])
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cwd", default=ROOT,
                   help="repo root to run from (default: script's project root)")
    args = p.parse_args()

    cwd = args.cwd
    os.makedirs(os.path.join(cwd, args.datadir), exist_ok=True)
    os.makedirs(os.path.join(cwd, os.path.dirname(args.out)), exist_ok=True)

    if not os.path.exists(os.path.join(cwd, args.bench.lstrip("./"))) \
       and not os.path.exists(os.path.join(cwd, args.bench)):
        print(f"[warn] benchmark binary '{args.bench}' not found under {cwd}. "
              f"Build it first with `make`.", file=sys.stderr)

    rows = []
    for pattern in args.patterns:
        gen_params = PATTERN_PARAMS.get(pattern)
        if gen_params is None:
            print(f"[skip] unknown pattern '{pattern}'", file=sys.stderr)
            continue
        for m in args.sizes:
            binf = os.path.join(args.datadir, f"{pattern.replace('-', '')}_{m}.bin")
            gen_cmd = (f"python3 {args.gen} --pattern {pattern} "
                       f"--rows {m} --cols {m} --seed {args.seed} "
                       f"{' '.join(gen_params)} --out {binf}")
            g = sh(gen_cmd, cwd)
            print(g.stdout.strip() or f"generated {binf}")
            if g.returncode:
                print(f"[gen failed] {pattern} m={m}\n{g.stderr[-800:]}",
                      file=sys.stderr)
                continue

            bench_cmd = (f"{args.bench} --kernel all --bin {binf} --n {args.n} "
                         f"--warmup {args.warmup} --iters {args.iters}")
            b = sh(bench_cmd, cwd)
            if b.returncode:
                print(f"[bench failed] {pattern} m={m}\n{b.stderr[-800:]}",
                      file=sys.stderr)
                continue

            got = 0
            for rec in parse_harness_lines(b.stdout):
                rec["pattern"] = pattern
                rows.append(rec)
                got += 1
            print(f"  {pattern} m={m}: {got} kernel rows")

    if not rows:
        print("[error] no rows collected — nothing written.", file=sys.stderr)
        return 1

    out_path = os.path.join(cwd, args.out)
    cols = ["pattern", "kernel", "m", "k", "n", "nnz", "density", "time_ms", "gflops"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}  ({len(rows)} rows, "
          f"{len(args.patterns)} patterns x {len(args.sizes)} sizes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
