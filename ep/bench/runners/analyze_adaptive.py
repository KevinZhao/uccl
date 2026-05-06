"""Adaptive-num_sms validation analyzer for Phase 10 workload logs.

usage:
  python analyze_adaptive.py \
      --static workload-static-r0.log workload-static-r1.log \
      --adaptive workload-adaptive-r0.log workload-adaptive-r1.log \
      [--iter-warmup 2] [--csv adaptive-grid.csv]

Parses BENCH rows emitted by bench_overlap.py --mode=workload, aggregates
p50 / p99 / p99.9 across ranks (and across iterations beyond --iter-warmup)
for each (mode, num_tokens, num_sms) cell, and evaluates four gates:

  G1 — adaptive p99(ntok) <= 1.03 * min static p99 at that ntok
  G2 — decode win preserved:  p99_adapt(128) <= 0.75 * p99_static(128, 96)
  G3 — prefill preserved:     p99_adapt(512) <= p99_static(512, 22)
  G4 — 384 boundary sanity:   p99_adapt(384) within ±5% of min(p99_static(384,22), p99_static(384,48))

Any gate missing the required data prints MISSING rather than crashing.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict


# BENCH rank=0 iter=0 mode=combine-overlap-22 num_tokens=128 num_sms=22
#   resolved_sms=22 avg=... p50=... p99=... p999=... min=... max=...
_BENCH_RE = re.compile(r"^BENCH\s+(.*)$")
_KV_RE = re.compile(r"(\w+)=(\S+)")

_NUMERIC = {"rank", "iter", "num_tokens", "num_sms", "resolved_sms",
            "avg", "p50", "p99", "p999", "min", "max"}


def parse_bench_lines(paths, iter_warmup: int) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                m = _BENCH_RE.search(line)
                if not m:
                    continue
                body = m.group(1)
                row = {k: v for k, v in _KV_RE.findall(body)}
                for k in _NUMERIC:
                    if k in row:
                        try:
                            row[k] = float(row[k]) if "." in row[k] else int(row[k])
                        except ValueError:
                            pass
                if "iter" in row and isinstance(row["iter"], int) and row["iter"] < iter_warmup:
                    continue
                rows.append(row)
    return rows


def aggregate(rows, mode_prefix: str):
    """Group combine-overlap rows by (ntok, resolved_sms|num_sms) and
    return {(ntok, key_sms): {mean_p99, median_p99, mean_p50, n_samples}}.

    key_sms uses `resolved_sms` when present (Grid B adaptive), else
    `num_sms` (Grid A static / older logs without the resolved field).
    """
    buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
    p50s: dict[tuple[int, int], list[float]] = defaultdict(list)
    p999s: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in rows:
        mode = r.get("mode", "")
        if not mode.startswith(mode_prefix):
            continue
        if not isinstance(r.get("num_tokens"), int):
            continue
        ntok = r["num_tokens"]
        key_sms = r.get("resolved_sms")
        if not isinstance(key_sms, int) or key_sms <= 0:
            key_sms = r.get("num_sms", 0)
        p99 = r.get("p99")
        if isinstance(p99, float):
            buckets[(ntok, key_sms)].append(p99)
        p50 = r.get("p50")
        if isinstance(p50, float):
            p50s[(ntok, key_sms)].append(p50)
        p999 = r.get("p999")
        if isinstance(p999, float):
            p999s[(ntok, key_sms)].append(p999)
    out = {}
    for key, vals in buckets.items():
        out[key] = {
            "median_p99": statistics.median(vals),
            "mean_p99": statistics.mean(vals),
            "mean_p50": statistics.mean(p50s.get(key, [0.0])),
            "mean_p999": statistics.mean(p999s.get(key, [0.0])),
            "n": len(vals),
        }
    return out


def fmt(v, precision=1):
    if v is None:
        return "MISSING"
    return f"{v:.{precision}f}"


def evaluate_gates(static, adaptive, out_stream=sys.stdout):
    """Evaluate G1-G4 and print a pass/fail report."""
    # Determine ntok set from adaptive (authoritative — what we actually ran)
    ntoks_adapt = sorted({ntok for (ntok, _) in adaptive.keys()})
    report = []

    print("\n=== Grid A (static num_sms) median p99 (µs) ===", file=out_stream)
    static_ntoks = sorted({ntok for (ntok, _) in static.keys()})
    static_sms = sorted({sms for (_, sms) in static.keys()})
    print(f"{'ntok':>6} " + "".join(f"{s:>10}" for s in static_sms) + f"{'best':>10}", file=out_stream)
    for ntok in static_ntoks:
        row = [f"{ntok:>6}"]
        best = None
        for s in static_sms:
            v = static.get((ntok, s), {}).get("median_p99")
            if v is not None and (best is None or v < best):
                best = v
            row.append(f"{fmt(v, 0):>10}")
        row.append(f"{fmt(best, 0):>10}")
        print(" ".join(row), file=out_stream)

    print("\n=== Grid B (adaptive num_sms) median p99 (µs) ===", file=out_stream)
    print(f"{'ntok':>6} {'resolved':>10} {'p50':>10} {'p99':>10} {'p99.9':>10} {'n':>6}", file=out_stream)
    for (ntok, sms), stats in sorted(adaptive.items()):
        print(f"{ntok:>6} {sms:>10} "
              f"{fmt(stats['mean_p50'], 0):>10} "
              f"{fmt(stats['median_p99'], 0):>10} "
              f"{fmt(stats['mean_p999'], 0):>10} "
              f"{stats['n']:>6}", file=out_stream)

    print("\n=== Gate evaluation ===", file=out_stream)
    # G1: adaptive <= 1.03 * best static for every ntok
    g1_fail = []
    for ntok in ntoks_adapt:
        adapt_p99 = min(
            (s["median_p99"] for (n, _), s in adaptive.items() if n == ntok),
            default=None,
        )
        static_best = min(
            (s["median_p99"] for (n, _), s in static.items() if n == ntok),
            default=None,
        )
        if adapt_p99 is None or static_best is None:
            print(f"G1 ntok={ntok}: MISSING data", file=out_stream)
            continue
        ratio = adapt_p99 / static_best
        ok = ratio <= 1.03
        tag = "PASS" if ok else "FAIL"
        print(f"G1 ntok={ntok}: adaptive={adapt_p99:.0f} best_static={static_best:.0f} ratio={ratio:.3f} → {tag}", file=out_stream)
        if not ok:
            g1_fail.append((ntok, ratio))

    # G2: decode preserved — p99_adapt(128) <= 0.75 * p99_static(128,96)
    adapt_128 = min((s["median_p99"] for (n, _), s in adaptive.items() if n == 128), default=None)
    static_128_96 = static.get((128, 96), {}).get("median_p99")
    if adapt_128 is None or static_128_96 is None:
        print(f"G2 ntok=128: MISSING (need adaptive@128 and static@128,96)", file=out_stream)
    else:
        ratio = adapt_128 / static_128_96
        tag = "PASS" if ratio <= 0.75 else "FAIL"
        print(f"G2 ntok=128: adaptive={adapt_128:.0f} static_96={static_128_96:.0f} ratio={ratio:.3f} (target ≤0.75) → {tag}", file=out_stream)

    # G3: prefill preserved — p99_adapt(512) <= p99_static(512,22)
    adapt_512 = min((s["median_p99"] for (n, _), s in adaptive.items() if n == 512), default=None)
    static_512_22 = static.get((512, 22), {}).get("median_p99")
    if adapt_512 is None or static_512_22 is None:
        print(f"G3 ntok=512: MISSING (need adaptive@512 and static@512,22)", file=out_stream)
    else:
        ratio = adapt_512 / static_512_22
        tag = "PASS" if ratio <= 1.00 else "FAIL"
        print(f"G3 ntok=512: adaptive={adapt_512:.0f} static_22={static_512_22:.0f} ratio={ratio:.3f} (target ≤1.00) → {tag}", file=out_stream)

    # G4: 384 boundary — within ±5% of min(static@384,22 or @384,48)
    adapt_384 = min((s["median_p99"] for (n, _), s in adaptive.items() if n == 384), default=None)
    s384_22 = static.get((384, 22), {}).get("median_p99")
    s384_48 = static.get((384, 48), {}).get("median_p99")
    candidates = [v for v in (s384_22, s384_48) if v is not None]
    if adapt_384 is None or not candidates:
        print(f"G4 ntok=384: MISSING (need adaptive@384 and static@384,22 or @384,48)", file=out_stream)
    else:
        best = min(candidates)
        ratio = adapt_384 / best
        tag = "PASS" if 0.95 <= ratio <= 1.05 else "FAIL"
        print(f"G4 ntok=384: adaptive={adapt_384:.0f} best(22,48)={best:.0f} ratio={ratio:.3f} (target 0.95-1.05) → {tag}", file=out_stream)


def write_csv(path, static, adaptive):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["grid", "ntok", "num_sms", "median_p99", "mean_p99", "mean_p50", "mean_p999", "n"])
        for (ntok, sms), stats in sorted(static.items()):
            w.writerow(["static", ntok, sms,
                        f"{stats['median_p99']:.2f}",
                        f"{stats['mean_p99']:.2f}",
                        f"{stats['mean_p50']:.2f}",
                        f"{stats['mean_p999']:.2f}",
                        stats["n"]])
        for (ntok, sms), stats in sorted(adaptive.items()):
            w.writerow(["adaptive", ntok, sms,
                        f"{stats['median_p99']:.2f}",
                        f"{stats['mean_p99']:.2f}",
                        f"{stats['mean_p50']:.2f}",
                        f"{stats['mean_p999']:.2f}",
                        stats["n"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", nargs="+", required=True, help="workload-static-r*.log files")
    ap.add_argument("--adaptive", nargs="+", required=True, help="workload-adaptive-r*.log files")
    ap.add_argument("--iter-warmup", type=int, default=2,
                    help="drop first N iterations as warmup (default 2)")
    ap.add_argument("--csv", type=str, default=None, help="optional CSV output path")
    args = ap.parse_args()

    static_rows = parse_bench_lines(args.static, args.iter_warmup)
    adaptive_rows = parse_bench_lines(args.adaptive, args.iter_warmup)

    static_agg = aggregate(static_rows, "combine-overlap-")
    adaptive_agg = aggregate(adaptive_rows, "combine-overlap-")

    if not static_agg:
        print("[warn] no static BENCH rows parsed (check --static paths)", file=sys.stderr)
    if not adaptive_agg:
        print("[warn] no adaptive BENCH rows parsed (check --adaptive paths)", file=sys.stderr)

    evaluate_gates(static_agg, adaptive_agg)

    if args.csv:
        write_csv(args.csv, static_agg, adaptive_agg)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
