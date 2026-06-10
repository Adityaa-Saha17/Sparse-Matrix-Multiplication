#!/usr/bin/env python3
"""
roofline.py — Week 4 roofline / "% of theoretical" analysis for SpMM on T4.

Takes the harness benchmark lines (kernel=... m=... gflops=...) and computes,
for every kernel x problem cell:

  * achieved GFLOP/s            (from the harness, = 2*nnz*N / time)
  * arithmetic intensity (AI)   (FLOPs / compulsory DRAM bytes)
  * effective DRAM bandwidth    (compulsory bytes / time)
  * roofline ceiling            (min(peak_fp32, AI * peak_BW))
  * % of FP32 peak              (achieved / 8140 GFLOP/s)
  * % of the roofline ceiling   (achieved / ceiling) -- "how close to theoretical"
  * % of peak DRAM bandwidth    (effective BW / 320 GB/s)

Theoretical ceilings are the NVIDIA T4 (Turing TU104) datasheet values:
  FP32 peak            = 8140  GFLOP/s
  FP16 Tensor Core peak= 65130 GFLOP/s
  DRAM bandwidth       = 320   GB/s
  L2 cache             = 4     MB   (B fits in L2 when k*N*4 <= 4 MB)

The DRAM-byte model is the *compulsory* (lower-bound) traffic: every operand is
counted as read/written exactly once, assuming perfect reuse. This is the
optimistic model -> highest AI -> a roofline ceiling that is a genuine upper
bound on what the hardware can deliver for this access pattern. A kernel that
sits at, say, 70% of this ceiling has ~30% of headroom that better reuse /
scheduling could in principle recover.

Usage:
    python3 scripts/roofline.py                 # uses embedded captured sweep
    python3 scripts/roofline.py results.csv     # parse a harness CSV/log instead
    python3 scripts/roofline.py --plot          # also write the roofline PNG

Outputs (under reports/):
    reports/data/sweep.csv          raw measured points
    reports/data/roofline.csv       computed roofline table
    reports/figures/roofline_t4.png roofline plot (with --plot)
"""
import csv
import os
import re
import sys

# ---------------------------------------------------------------- T4 ceilings
FP32_PEAK_GFLOPS = 8140.0     # NVIDIA T4 datasheet, single precision
FP16_TC_GFLOPS   = 65130.0    # NVIDIA T4 datasheet, FP16 Tensor Core
DRAM_BW_GBs      = 320.0      # NVIDIA T4 datasheet, GDDR6
L2_BYTES         = 4 * 1024 * 1024  # 4 MB Turing L2
RIDGE_FP32       = FP32_PEAK_GFLOPS / DRAM_BW_GBs  # ~25.4 FLOP/byte

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------- embedded dataset
# Captured from colabRunner.ipynb cell 26 (full Phase-2 sweep, Colab T4,
# n=256, warmup 5, iters 20, seed 42). This is the canonical measured data so
# the analysis is reproducible without a GPU. Re-running the notebook and
# feeding the new log to this script reproduces (and refreshes) the table.
EMBEDDED = """
kernel=baseline m=1024 k=1024 n=256 nnz=52260 time_ms=0.1392 gflops=192.18
kernel=memopt m=1024 k=1024 n=256 nnz=52260 time_ms=0.1741 gflops=153.71
kernel=memopt_v2 m=1024 k=1024 n=256 nnz=52260 time_ms=0.1731 gflops=154.62
kernel=tiled m=1024 k=1024 n=256 nnz=52260 time_ms=0.1047 gflops=255.63
kernel=tiled_v2 m=1024 k=1024 n=256 nnz=52260 time_ms=0.1024 gflops=261.30
kernel=tiled_v3 m=1024 k=1024 n=256 nnz=52260 time_ms=0.1006 gflops=266.04
kernel=tiled_v4 m=1024 k=1024 n=256 nnz=52260 time_ms=0.1016 gflops=263.44
kernel=wmma m=1024 k=1024 n=256 nnz=52260 time_ms=0.2099 gflops=127.50
kernel=baseline m=1024 k=1024 n=256 nnz=10388 time_ms=0.0261 gflops=203.94
kernel=memopt m=1024 k=1024 n=256 nnz=10388 time_ms=0.0307 gflops=173.13
kernel=memopt_v2 m=1024 k=1024 n=256 nnz=10388 time_ms=0.0306 gflops=173.68
kernel=tiled m=1024 k=1024 n=256 nnz=10388 time_ms=0.0224 gflops=237.10
kernel=tiled_v2 m=1024 k=1024 n=256 nnz=10388 time_ms=0.0200 gflops=266.36
kernel=tiled_v3 m=1024 k=1024 n=256 nnz=10388 time_ms=0.0214 gflops=248.07
kernel=tiled_v4 m=1024 k=1024 n=256 nnz=10388 time_ms=0.0246 gflops=216.42
kernel=wmma m=1024 k=1024 n=256 nnz=10388 time_ms=0.1497 gflops=35.54
kernel=baseline m=1024 k=1024 n=256 nnz=1044 time_ms=0.0157 gflops=34.09
kernel=memopt m=1024 k=1024 n=256 nnz=1044 time_ms=0.0143 gflops=37.29
kernel=memopt_v2 m=1024 k=1024 n=256 nnz=1044 time_ms=0.0146 gflops=36.71
kernel=tiled m=1024 k=1024 n=256 nnz=1044 time_ms=0.0114 gflops=47.05
kernel=tiled_v2 m=1024 k=1024 n=256 nnz=1044 time_ms=0.0107 gflops=49.86
kernel=tiled_v3 m=1024 k=1024 n=256 nnz=1044 time_ms=0.0116 gflops=46.14
kernel=tiled_v4 m=1024 k=1024 n=256 nnz=1044 time_ms=0.0121 gflops=44.19
kernel=wmma m=1024 k=1024 n=256 nnz=1044 time_ms=0.0452 gflops=11.84
kernel=baseline m=4096 k=4096 n=256 nnz=839236 time_ms=1.2759 gflops=336.76
kernel=memopt m=4096 k=4096 n=256 nnz=839236 time_ms=1.3057 gflops=329.10
kernel=memopt_v2 m=4096 k=4096 n=256 nnz=839236 time_ms=1.3058 gflops=329.06
kernel=tiled m=4096 k=4096 n=256 nnz=839236 time_ms=0.9889 gflops=434.53
kernel=tiled_v2 m=4096 k=4096 n=256 nnz=839236 time_ms=0.9395 gflops=457.38
kernel=tiled_v3 m=4096 k=4096 n=256 nnz=839236 time_ms=0.9646 gflops=445.45
kernel=tiled_v4 m=4096 k=4096 n=256 nnz=839236 time_ms=0.8491 gflops=506.04
kernel=wmma m=4096 k=4096 n=256 nnz=839236 time_ms=2.0869 gflops=205.90
kernel=baseline m=4096 k=4096 n=256 nnz=167686 time_ms=0.1904 gflops=450.84
kernel=memopt m=4096 k=4096 n=256 nnz=167686 time_ms=0.1905 gflops=450.62
kernel=memopt_v2 m=4096 k=4096 n=256 nnz=167686 time_ms=0.1903 gflops=451.15
kernel=tiled m=4096 k=4096 n=256 nnz=167686 time_ms=0.1494 gflops=574.76
kernel=tiled_v2 m=4096 k=4096 n=256 nnz=167686 time_ms=0.1372 gflops=625.69
kernel=tiled_v3 m=4096 k=4096 n=256 nnz=167686 time_ms=0.1456 gflops=589.67
kernel=tiled_v4 m=4096 k=4096 n=256 nnz=167686 time_ms=0.1670 gflops=514.18
kernel=wmma m=4096 k=4096 n=256 nnz=167686 time_ms=1.9534 gflops=43.95
kernel=baseline m=4096 k=4096 n=256 nnz=16935 time_ms=0.0881 gflops=98.46
kernel=memopt m=4096 k=4096 n=256 nnz=16935 time_ms=0.0744 gflops=116.49
kernel=memopt_v2 m=4096 k=4096 n=256 nnz=16935 time_ms=0.0740 gflops=117.25
kernel=tiled m=4096 k=4096 n=256 nnz=16935 time_ms=0.0618 gflops=140.39
kernel=tiled_v2 m=4096 k=4096 n=256 nnz=16935 time_ms=0.0614 gflops=141.12
kernel=tiled_v3 m=4096 k=4096 n=256 nnz=16935 time_ms=0.0620 gflops=139.89
kernel=tiled_v4 m=4096 k=4096 n=256 nnz=16935 time_ms=0.0596 gflops=145.52
kernel=wmma m=4096 k=4096 n=256 nnz=16935 time_ms=0.5323 gflops=16.29
kernel=baseline m=8192 k=8192 n=256 nnz=3354625 time_ms=7.9662 gflops=215.61
kernel=memopt m=8192 k=8192 n=256 nnz=3354625 time_ms=7.4911 gflops=229.28
kernel=memopt_v2 m=8192 k=8192 n=256 nnz=3354625 time_ms=7.6778 gflops=223.71
kernel=tiled m=8192 k=8192 n=256 nnz=3354625 time_ms=4.0758 gflops=421.41
kernel=tiled_v2 m=8192 k=8192 n=256 nnz=3354625 time_ms=4.1165 gflops=417.24
kernel=tiled_v3 m=8192 k=8192 n=256 nnz=3354625 time_ms=3.6639 gflops=468.78
kernel=tiled_v4 m=8192 k=8192 n=256 nnz=3354625 time_ms=3.5058 gflops=489.92
kernel=wmma m=8192 k=8192 n=256 nnz=3354625 time_ms=8.7354 gflops=196.62
kernel=baseline m=8192 k=8192 n=256 nnz=671114 time_ms=1.7326 gflops=198.32
kernel=memopt m=8192 k=8192 n=256 nnz=671114 time_ms=1.8980 gflops=181.04
kernel=memopt_v2 m=8192 k=8192 n=256 nnz=671114 time_ms=1.8924 gflops=181.58
kernel=tiled m=8192 k=8192 n=256 nnz=671114 time_ms=0.8935 gflops=384.57
kernel=tiled_v2 m=8192 k=8192 n=256 nnz=671114 time_ms=1.2456 gflops=275.86
kernel=tiled_v3 m=8192 k=8192 n=256 nnz=671114 time_ms=0.8462 gflops=406.04
kernel=tiled_v4 m=8192 k=8192 n=256 nnz=671114 time_ms=0.8540 gflops=402.35
kernel=wmma m=8192 k=8192 n=256 nnz=671114 time_ms=8.0694 gflops=42.58
kernel=baseline m=8192 k=8192 n=256 nnz=67137 time_ms=0.2499 gflops=137.58
kernel=memopt m=8192 k=8192 n=256 nnz=67137 time_ms=0.2662 gflops=129.11
kernel=memopt_v2 m=8192 k=8192 n=256 nnz=67137 time_ms=0.2659 gflops=129.26
kernel=tiled m=8192 k=8192 n=256 nnz=67137 time_ms=0.2273 gflops=151.21
kernel=tiled_v2 m=8192 k=8192 n=256 nnz=67137 time_ms=0.2063 gflops=166.62
kernel=tiled_v3 m=8192 k=8192 n=256 nnz=67137 time_ms=0.2090 gflops=164.50
kernel=tiled_v4 m=8192 k=8192 n=256 nnz=67137 time_ms=0.2231 gflops=154.09
kernel=wmma m=8192 k=8192 n=256 nnz=67137 time_ms=2.1053 gflops=16.33
kernel=baseline m=16384 k=16384 n=256 nnz=13421456 time_ms=41.0064 gflops=167.58
kernel=memopt m=16384 k=16384 n=256 nnz=13421456 time_ms=48.0727 gflops=142.95
kernel=memopt_v2 m=16384 k=16384 n=256 nnz=13421456 time_ms=48.5683 gflops=141.49
kernel=tiled m=16384 k=16384 n=256 nnz=13421456 time_ms=30.7573 gflops=223.42
kernel=tiled_v2 m=16384 k=16384 n=256 nnz=13421456 time_ms=30.8335 gflops=222.87
kernel=tiled_v3 m=16384 k=16384 n=256 nnz=13421456 time_ms=28.2071 gflops=243.62
kernel=tiled_v4 m=16384 k=16384 n=256 nnz=13421456 time_ms=26.2352 gflops=261.93
kernel=wmma m=16384 k=16384 n=256 nnz=13421456 time_ms=37.1836 gflops=184.81
kernel=baseline m=16384 k=16384 n=256 nnz=2684748 time_ms=10.0332 gflops=137.00
kernel=memopt m=16384 k=16384 n=256 nnz=2684748 time_ms=10.9344 gflops=125.71
kernel=memopt_v2 m=16384 k=16384 n=256 nnz=2684748 time_ms=10.9286 gflops=125.78
kernel=tiled m=16384 k=16384 n=256 nnz=2684748 time_ms=7.3849 gflops=186.14
kernel=tiled_v2 m=16384 k=16384 n=256 nnz=2684748 time_ms=7.7537 gflops=177.28
kernel=tiled_v3 m=16384 k=16384 n=256 nnz=2684748 time_ms=6.6985 gflops=205.21
kernel=tiled_v4 m=16384 k=16384 n=256 nnz=2684748 time_ms=6.5679 gflops=209.29
kernel=wmma m=16384 k=16384 n=256 nnz=2684748 time_ms=34.2917 gflops=40.09
kernel=baseline m=16384 k=16384 n=256 nnz=268160 time_ms=1.0875 gflops=126.25
kernel=memopt m=16384 k=16384 n=256 nnz=268160 time_ms=1.2268 gflops=111.92
kernel=memopt_v2 m=16384 k=16384 n=256 nnz=268160 time_ms=1.2226 gflops=112.30
kernel=tiled m=16384 k=16384 n=256 nnz=268160 time_ms=1.0287 gflops=133.46
kernel=tiled_v2 m=16384 k=16384 n=256 nnz=268160 time_ms=0.9219 gflops=148.93
kernel=tiled_v3 m=16384 k=16384 n=256 nnz=268160 time_ms=0.9295 gflops=147.71
kernel=tiled_v4 m=16384 k=16384 n=256 nnz=268160 time_ms=0.9864 gflops=139.19
kernel=wmma m=16384 k=16384 n=256 nnz=268160 time_ms=8.8579 gflops=15.50
"""

FIELD = re.compile(r"(\w+)=([\-\d.eE+]+|\w+)")


def parse(text):
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line.startswith("kernel="):
            continue
        d = dict(FIELD.findall(line))
        rows.append({
            "kernel": d["kernel"],
            "m": int(d["m"]), "k": int(d["k"]), "n": int(d["n"]),
            "nnz": int(d["nnz"]),
            "time_ms": float(d["time_ms"]),
            "gflops": float(d["gflops"]),
        })
    return rows


def csr_dram_bytes(m, k, n, nnz):
    """Compulsory DRAM traffic for one CSR SpMM C[m,n] = A[m,k] * B[k,n], fp32.
    Each operand counted once (perfect-reuse lower bound)."""
    a_vals = nnz * 4          # float32 values
    a_cols = nnz * 4          # int32 column indices
    a_rptr = (m + 1) * 4      # int32 row pointers
    b      = k * n * 4        # dense B, read once
    c      = m * n * 4        # dense C, written once
    return a_vals + a_cols + a_rptr + b + c


def analyse(rows):
    out = []
    for r in rows:
        m, k, n, nnz = r["m"], r["k"], r["n"], r["nnz"]
        flops = 2.0 * nnz * n
        bytes_ = csr_dram_bytes(m, k, n, nnz)
        ai = flops / bytes_
        ceiling = min(FP32_PEAK_GFLOPS, ai * DRAM_BW_GBs)
        ach = r["gflops"]
        eff_bw = bytes_ / (r["time_ms"] * 1.0e6)  # GB/s
        b_l2 = (k * n * 4) <= L2_BYTES
        out.append({
            **r,
            "flops": flops,
            "dram_bytes": bytes_,
            "ai_flop_per_byte": ai,
            "ceiling_gflops": ceiling,
            "pct_fp32_peak": 100.0 * ach / FP32_PEAK_GFLOPS,
            "pct_roofline": 100.0 * ach / ceiling,
            "eff_dram_bw_gbs": eff_bw,
            "pct_dram_bw": 100.0 * eff_bw / DRAM_BW_GBs,
            "bound": "compute/L2" if ai > RIDGE_FP32 else "memory",
            "B_fits_L2": b_l2,
        })
    return out


def write_csv(path, rows, cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def density_label(m, nnz):
    return nnz / (m * m)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_plot = "--plot" in sys.argv

    if args:
        text = open(args[0]).read()
    else:
        text = EMBEDDED
    rows = parse(text)
    res = analyse(rows)

    # raw + computed CSVs
    write_csv(os.path.join(ROOT, "reports", "data", "sweep.csv"), rows,
              ["kernel", "m", "k", "n", "nnz", "time_ms", "gflops"])
    write_csv(os.path.join(ROOT, "reports", "data", "roofline.csv"), res,
              ["kernel", "m", "density", "nnz", "gflops", "ai_flop_per_byte",
               "ceiling_gflops", "pct_fp32_peak", "pct_roofline",
               "eff_dram_bw_gbs", "pct_dram_bw", "bound", "B_fits_L2"])

    # ---- console: roofline ridge + per-cell best CSR kernel
    print(f"T4 ceilings: FP32 {FP32_PEAK_GFLOPS:.0f} GFLOP/s | "
          f"DRAM {DRAM_BW_GBs:.0f} GB/s | ridge {RIDGE_FP32:.1f} FLOP/byte | "
          f"L2 {L2_BYTES//(1024*1024)} MB\n")

    cells = sorted({(r["m"], r["nnz"]) for r in res})
    hdr = (f"{'cell (m, dens)':>18} {'AI':>6} {'bound':>11} {'roof':>8} "
           f"{'best CSR':>10} {'GFLOP/s':>8} {'%peak':>6} {'%roof':>6} {'%BW':>6}")
    print(hdr)
    print("-" * len(hdr))
    for (m, nnz) in cells:
        csr = [r for r in res if r["m"] == m and r["nnz"] == nnz
               and r["kernel"] != "wmma"]
        best = max(csr, key=lambda r: r["gflops"])
        dens = density_label(m, nnz)
        print(f"{('%d, %.3f' % (m, dens)):>18} "
              f"{best['ai_flop_per_byte']:>6.1f} {best['bound']:>11} "
              f"{best['ceiling_gflops']:>7.0f}G {best['kernel']:>10} "
              f"{best['gflops']:>8.1f} {best['pct_fp32_peak']:>5.1f}% "
              f"{best['pct_roofline']:>5.1f}% {best['pct_dram_bw']:>5.1f}%")

    if do_plot:
        plot(res)
    print("\nWrote reports/data/sweep.csv and reports/data/roofline.csv")


def plot(res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    # roofline lines
    import numpy as np
    ai = np.logspace(-1, 2.5, 200)
    roof = np.minimum(FP32_PEAK_GFLOPS, ai * DRAM_BW_GBs)
    ax.plot(ai, roof, "k-", lw=2, label="FP32 roofline")
    ax.axhline(FP32_PEAK_GFLOPS, ls="--", c="gray", lw=1)
    ax.axvline(RIDGE_FP32, ls=":", c="gray", lw=1)
    ax.text(RIDGE_FP32 * 1.05, 200, f"ridge {RIDGE_FP32:.1f}", fontsize=8)
    # best CSR kernel per cell
    cells = sorted({(r["m"], r["nnz"]) for r in res})
    for (m, nnz) in cells:
        csr = [r for r in res if r["m"] == m and r["nnz"] == nnz
               and r["kernel"] != "wmma"]
        best = max(csr, key=lambda r: r["gflops"])
        ax.plot(best["ai_flop_per_byte"], best["gflops"], "o", ms=7)
        ax.annotate(f"m={m}\nd={density_label(m,nnz):.0e}",
                    (best["ai_flop_per_byte"], best["gflops"]),
                    fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte, compulsory DRAM)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("T4 FP32 roofline — best CSR SpMM kernel per cell")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    out = os.path.join(ROOT, "reports", "figures", "roofline_t4.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
