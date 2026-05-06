"""Probe v2 gate evaluator (P1-P4 from gpu_session_runbook.md Phase 11).

usage: python gate_probev2.py probev2-r0.log probev2-r1.log

Parses PROBE_SCHEMA and PROBE rows, derives init/body/sync shares per cell
via analyze_probe.derive_indicators, then evaluates:

  P1 — every log contains PROBE_SCHEMA version=2
  P2 — every cell has body_share > 0.20
  P3 — init_share(128, 22) > init_share(512, 22)   (decode init-heavy)
  P4 — max sync_share across cells >= 0.10         (sync attackable?)

Exits 0 if all available gates PASS, non-zero if any FAIL. MISSING data
does not auto-fail but is reported.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# Import the existing analyzer as a library so we reuse its parsers
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_probe  # noqa: E402


_SCHEMA_RE = re.compile(r"PROBE_SCHEMA\s.*?version=(\d+)")


def check_schema(paths):
    per_file = {}
    for p in paths:
        with open(p) as f:
            text = f.read()
        m = _SCHEMA_RE.search(text)
        per_file[p] = int(m.group(1)) if m else None
    return per_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    args = ap.parse_args()

    fails = []

    # P1
    print("=== P1: PROBE_SCHEMA version=2 in every log ===")
    schemas = check_schema(args.logs)
    p1_ok = True
    for p, v in schemas.items():
        tag = "PASS" if v == 2 else "FAIL"
        print(f"  {os.path.basename(p)}: version={v} → {tag}")
        if v != 2:
            p1_ok = False
    if not p1_ok:
        fails.append("P1")

    # Parse probe rows via analyze_probe's machinery
    config, rows, warnings = analyze_probe.parse_logs(args.logs)
    for w in warnings:
        print(f"[warn] {w}", file=sys.stderr)
    cells = analyze_probe.aggregate(rows)

    # Build per-cell indicator dict
    print("\n=== Per-cell init/body/sync shares ===")
    print(f"{'ntok':>6} {'nsms':>6} {'init':>7} {'body':>7} {'sync':>7} {'T_slot(clk)':>12}")
    cell_indicators = {}
    for (ntok, nsms), mechs in sorted(cells.items()):
        ind = analyze_probe.derive_indicators(mechs)
        cell_indicators[(ntok, nsms)] = ind
        slot_mean = mechs.get("T_slot", {}).get("mean")
        def pct(v):
            return f"{v*100:>6.1f}%" if v is not None else "   N/A"
        slot_str = f"{slot_mean:.1f}" if isinstance(slot_mean, (int, float)) else "N/A"
        print(f"{ntok:>6} {nsms:>6} "
              f"{pct(ind.get('init_share')):>7} "
              f"{pct(ind.get('body_share')):>7} "
              f"{pct(ind.get('sync_share')):>7} "
              f"{slot_str:>12}")

    # P2: body_share > 0.20 every cell
    print("\n=== P2: body_share > 0.20 every cell ===")
    p2_ok = True
    for cell, ind in cell_indicators.items():
        body = ind.get("body_share")
        if body is None:
            print(f"  {cell}: body_share MISSING")
            p2_ok = False
            continue
        tag = "PASS" if body > 0.20 else "FAIL"
        print(f"  ntok={cell[0]} nsms={cell[1]}: body_share={body:.3f} → {tag}")
        if body <= 0.20:
            p2_ok = False
    if not p2_ok:
        fails.append("P2")

    # P3: init_share(128,22) > init_share(512,22)
    print("\n=== P3: init_share(128,22) > init_share(512,22) ===")
    i128 = cell_indicators.get((128, 22), {}).get("init_share")
    i512 = cell_indicators.get((512, 22), {}).get("init_share")
    if i128 is None or i512 is None:
        print(f"  MISSING (128,22)={i128} (512,22)={i512}")
    else:
        tag = "PASS" if i128 > i512 else "FAIL"
        print(f"  init@(128,22)={i128:.3f} init@(512,22)={i512:.3f} → {tag}")
        if i128 <= i512:
            fails.append("P3")

    # P4: at least one cell sync_share >= 0.10
    print("\n=== P4: max sync_share >= 0.10 ===")
    sync_vals = [(c, ind.get("sync_share")) for c, ind in cell_indicators.items()
                 if ind.get("sync_share") is not None]
    if not sync_vals:
        print("  no cells reported sync_share")
        fails.append("P4")
    else:
        best_cell, best = max(sync_vals, key=lambda kv: kv[1])
        tag = "PASS" if best >= 0.10 else "FAIL"
        print(f"  best cell: ntok={best_cell[0]} nsms={best_cell[1]} sync_share={best:.3f} → {tag}")
        if best < 0.10:
            fails.append("P4")

    print("\n=== Summary ===")
    if fails:
        print(f"FAIL: {', '.join(fails)}")
        sys.exit(1)
    else:
        print("All gates PASS")


if __name__ == "__main__":
    main()
