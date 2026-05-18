#!/usr/bin/env python3
"""Generate synthetic sparse matrices and save them in the project's binary
CSR format (matches csr_read_binary_host in src/csr.cu).

Layout (all little-endian, native int sizes assumed = 4 bytes):
    int32  num_rows
    int32  num_cols
    int32  nnz
    int32  row_ptr[num_rows + 1]
    int32  col_idx[nnz]
    float  values[nnz]

Patterns:
    uniform        — Bernoulli(density) per entry.
    banded         — nonzeros within +/- bandwidth of the main diagonal.
    block-diagonal — square dense blocks of size `block` along the diagonal.
    power-law      — row-degrees follow a power-law (graph-like sparsity).
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np


def gen_uniform(rows: int, cols: int, density: float, rng: np.random.Generator):
    mask = rng.random((rows, cols)) < density
    rs, cs = np.nonzero(mask)
    vs = rng.uniform(-1.0, 1.0, size=rs.size).astype(np.float32)
    return rs.astype(np.int32), cs.astype(np.int32), vs


def gen_banded(rows: int, cols: int, bandwidth: int, rng: np.random.Generator):
    rs, cs = [], []
    for r in range(rows):
        lo = max(0, r - bandwidth)
        hi = min(cols, r + bandwidth + 1)
        cols_r = np.arange(lo, hi, dtype=np.int32)
        rs.append(np.full(cols_r.size, r, dtype=np.int32))
        cs.append(cols_r)
    rs = np.concatenate(rs)
    cs = np.concatenate(cs)
    vs = rng.uniform(-1.0, 1.0, size=rs.size).astype(np.float32)
    return rs, cs, vs


def gen_block_diagonal(rows: int, cols: int, block: int, rng: np.random.Generator):
    n_blocks = min(rows, cols) // block
    rs, cs = [], []
    for b in range(n_blocks):
        rr = np.arange(b * block, (b + 1) * block, dtype=np.int32)
        cc = np.arange(b * block, (b + 1) * block, dtype=np.int32)
        R, C = np.meshgrid(rr, cc, indexing="ij")
        rs.append(R.ravel())
        cs.append(C.ravel())
    rs = np.concatenate(rs)
    cs = np.concatenate(cs)
    vs = rng.uniform(-1.0, 1.0, size=rs.size).astype(np.float32)
    return rs, cs, vs


def gen_power_law(rows: int, cols: int, mean_deg: float,
                  alpha: float, rng: np.random.Generator):
    # Degrees follow a Pareto-like distribution truncated to [1, cols].
    raw = rng.pareto(alpha, size=rows) + 1.0  # Pareto >= 1
    raw = raw * (mean_deg / raw.mean())
    degrees = np.clip(np.round(raw).astype(np.int64), 1, cols)

    rs_list, cs_list = [], []
    for r, d in enumerate(degrees):
        chosen = rng.choice(cols, size=int(d), replace=False)
        rs_list.append(np.full(d, r, dtype=np.int32))
        cs_list.append(chosen.astype(np.int32))
    rs = np.concatenate(rs_list)
    cs = np.concatenate(cs_list)
    vs = rng.uniform(-1.0, 1.0, size=rs.size).astype(np.float32)
    return rs, cs, vs


def coo_to_csr(rows: int, cols: int,
               r: np.ndarray, c: np.ndarray, v: np.ndarray):
    order = np.lexsort((c, r))
    r, c, v = r[order], c[order], v[order]
    row_ptr = np.zeros(rows + 1, dtype=np.int32)
    np.add.at(row_ptr, r + 1, 1)
    np.cumsum(row_ptr, out=row_ptr)
    return row_ptr, c.astype(np.int32), v.astype(np.float32)


def write_binary(path: Path, rows: int, cols: int,
                 row_ptr: np.ndarray, col_idx: np.ndarray, values: np.ndarray):
    nnz = int(values.size)
    with open(path, "wb") as f:
        f.write(struct.pack("<iii", rows, cols, nnz))
        f.write(row_ptr.astype("<i4").tobytes())
        f.write(col_idx.astype("<i4").tobytes())
        f.write(values.astype("<f4").tobytes())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", required=True,
                   choices=["uniform", "banded", "block-diagonal", "power-law"])
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--cols", type=int, required=True)
    p.add_argument("--density", type=float, default=0.01, help="for uniform")
    p.add_argument("--bandwidth", type=int, default=16, help="for banded")
    p.add_argument("--block", type=int, default=32, help="for block-diagonal")
    p.add_argument("--mean-deg", type=float, default=16.0, help="for power-law")
    p.add_argument("--alpha", type=float, default=2.5, help="for power-law")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.pattern == "uniform":
        r, c, v = gen_uniform(args.rows, args.cols, args.density, rng)
    elif args.pattern == "banded":
        r, c, v = gen_banded(args.rows, args.cols, args.bandwidth, rng)
    elif args.pattern == "block-diagonal":
        r, c, v = gen_block_diagonal(args.rows, args.cols, args.block, rng)
    else:
        r, c, v = gen_power_law(args.rows, args.cols, args.mean_deg, args.alpha, rng)

    row_ptr, col_idx, values = coo_to_csr(args.rows, args.cols, r, c, v)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_binary(args.out, args.rows, args.cols, row_ptr, col_idx, values)

    nnz = int(values.size)
    density = nnz / float(args.rows * args.cols)
    print(f"wrote {args.out}  rows={args.rows} cols={args.cols} "
          f"nnz={nnz} density={density:.6f}")


if __name__ == "__main__":
    sys.exit(main())
