"""Probe log analyzer for bench_overlap.py --mode=probe.

usage: python analyze_probe.py probe-r0.log [probe-r1.log ...]

Parses PROBE_CONFIG / PROBE stderr lines emitted by bench_overlap.py when
run with --mode=probe, aggregates per-mechanism statistics across ranks,
and prints a decision-tree recommendation for which K-1 kernel variant to
implement next (K-1a token pipe, K-1b slot prefetch, or K-1c work-steal).

Mechanism keys (all microseconds):
  T_slot - whole slot wall-time
  T_put  - IBGDA put NIC window
  T_sm   - per-SM total

Decision rules come from mechanism_probe_plan.md; see the DECISION_RULES
table below. Rules fire independently per (ntok, nsms) cell; the summary
ranks recommendations using prefill configs (ntok in {256, 512}).
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# PROBE_CONFIG sm_clock_khz=1980000 hidden=7168 num_topk=8 num_experts=288 ...
_CONFIG_RE = re.compile(r"^PROBE_CONFIG\s+(.*)$")

# PROBE rank=0 ntok=128 nsms=22 mech=T_slot n=1650 mean=1.85 p50=1.70 p99=4.12 p999=5.23 max=5.31 stdev=0.45
_PROBE_RE = re.compile(r"^PROBE\s+(.*)$")

# Matches "key=value" tokens; value is non-space, possibly bracketed list.
_KV_RE = re.compile(r"(\w+)=(\[[^\]]*\]|\S+)")

_NUMERIC_FIELDS = {"n", "mean", "p50", "p99", "p999", "max", "stdev",
                   "rank", "ntok", "nsms"}


def _parse_kv(body: str) -> dict:
    """Parse a "k=v k=v ..." string into a dict, coercing numerics."""
    out: dict = {}
    for k, v in _KV_RE.findall(body):
        if v.startswith("["):
            # list literal like [128, 256, 512]
            inner = v[1:-1].strip()
            if not inner:
                out[k] = []
            else:
                parts = [p.strip() for p in inner.split(",")]
                coerced = []
                for p in parts:
                    try:
                        coerced.append(int(p))
                    except ValueError:
                        try:
                            coerced.append(float(p))
                        except ValueError:
                            coerced.append(p)
                out[k] = coerced
        elif k in _NUMERIC_FIELDS:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
        else:
            # best-effort numeric coercion for unknown keys
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def parse_logs(paths):
    """Parse one or more probe log files.

    Returns:
      config: dict from PROBE_CONFIG (from first file that has one)
      rows:   list of dicts, one per PROBE row
      config_warnings: list[str]
    """
    config: dict = {}
    rows: list[dict] = []
    warnings: list[str] = []
    seen_configs: list[tuple[str, dict]] = []

    for path in paths:
        try:
            with open(path, "r") as fh:
                lines = fh.readlines()
        except OSError as exc:
            warnings.append(f"could not read {path}: {exc}")
            continue

        for line in lines:
            line = line.strip()
            if not line:
                continue
            m_cfg = _CONFIG_RE.match(line)
            if m_cfg:
                cfg = _parse_kv(m_cfg.group(1))
                seen_configs.append((path, cfg))
                if not config:
                    config = cfg
                continue
            m_probe = _PROBE_RE.match(line)
            if m_probe:
                row = _parse_kv(m_probe.group(1))
                row["_source"] = path
                rows.append(row)

    # Sanity-check that PROBE_CONFIGs agree on the key scalar fields.
    scalar_fields = ("hidden", "num_topk", "num_experts", "world_size",
                     "probe_iters")
    for path, cfg in seen_configs[1:]:
        for f in scalar_fields:
            if f in config and f in cfg and config[f] != cfg[f]:
                warnings.append(
                    f"PROBE_CONFIG mismatch in {path}: "
                    f"{f}={cfg[f]} (first file had {config[f]})"
                )

    return config, rows, warnings


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

# Per (ntok, nsms, mech): list of per-rank stat-dicts.
# We aggregate by taking the MEAN of per-rank means etc., to avoid mixing raw
# samples across GPUs with potential clock skew.

_STAT_KEYS = ("mean", "p50", "p99", "p999", "max", "stdev")


def aggregate(rows):
    """Aggregate PROBE rows by (ntok, nsms, mech) across ranks.

    Returns a nested dict: cells[(ntok, nsms)][mech] = {
        "mean": mean of per-rank means,
        "p50":  mean of per-rank p50s,
        ...,
        "n":    sum of per-rank n (total samples seen),
        "n_per_rank": mean of per-rank n (used for n_slots inference),
        "ranks": set of rank ids that contributed,
    }
    """
    # buckets[(ntok, nsms, mech)] = list of row-dicts (one per rank)
    buckets: dict = defaultdict(list)
    for r in rows:
        try:
            key = (int(r["ntok"]), int(r["nsms"]), str(r["mech"]))
        except (KeyError, TypeError, ValueError):
            continue
        buckets[key].append(r)

    cells: dict = defaultdict(dict)
    for (ntok, nsms, mech), rank_rows in buckets.items():
        agg: dict = {}
        for sk in _STAT_KEYS:
            vals = [r[sk] for r in rank_rows if sk in r and isinstance(r[sk], (int, float))]
            agg[sk] = statistics.fmean(vals) if vals else float("nan")
        ns = [r["n"] for r in rank_rows if isinstance(r.get("n"), (int, float))]
        agg["n"] = sum(ns) if ns else 0
        agg["n_per_rank"] = statistics.fmean(ns) if ns else 0.0
        agg["ranks"] = sorted({r.get("rank") for r in rank_rows if "rank" in r})
        cells[(ntok, nsms)][mech] = agg
    return cells


def derive_indicators(cell_mechs):
    """Compute put_ratio, sm_overhead_ratio, sm_spread for one (ntok, nsms) cell.

    When v2 probe fields (T_init, T_body, T_sync) are present, also compute
    the T_slot decomposition shares (init_share, body_share, sync_share).
    The v2 shares expose where slot-inter overhead actually lives: K-1b
    attacks T_init; K-1a attacks T_body; the Sprint B probe overestimated
    K-1b's ceiling because it lumped T_sync into the hoistable budget.

    Returns a dict with indicators and any missing-data notes.
    """
    notes: list[str] = []
    put = cell_mechs.get("T_put")
    slot = cell_mechs.get("T_slot")
    sm = cell_mechs.get("T_sm")
    # v2 fields; may be absent on older probe logs.
    init = cell_mechs.get("T_init")
    body = cell_mechs.get("T_body")
    sync = cell_mechs.get("T_sync")

    # put_ratio
    put_ratio = None
    if put and slot and slot.get("mean") and slot["mean"] > 0 \
            and put.get("mean") is not None:
        put_ratio = put["mean"] / slot["mean"]
    else:
        notes.append("put_ratio: missing T_put or T_slot mean")

    # n_slots_avg: T_slot counts one sample per slot per SM, T_sm counts one
    # per SM. So per-rank n_slots_avg ~= T_slot.n_per_rank / T_sm.n_per_rank.
    n_slots_avg = None
    if slot and sm and sm.get("n_per_rank"):
        denom = sm["n_per_rank"]
        if denom and denom > 0:
            n_slots_avg = slot.get("n_per_rank", 0) / denom

    sm_overhead_ratio = None
    if sm and slot and n_slots_avg and n_slots_avg > 0 \
            and slot.get("mean") and slot["mean"] > 0 \
            and sm.get("mean") is not None:
        sm_overhead_ratio = sm["mean"] / (n_slots_avg * slot["mean"])
    else:
        notes.append("sm_overhead_ratio: missing T_sm/T_slot or zero n_slots_avg")

    sm_spread = None
    if sm and sm.get("p50") and sm["p50"] > 0 and sm.get("max") is not None:
        sm_spread = (sm["max"] - sm["p50"]) / sm["p50"]
    else:
        notes.append("sm_spread: missing T_sm p50 or max")

    # v2 T_slot decomposition shares (fractions of slot mean time).
    # init_share = T_init / T_slot  (mbarrier_init burst; hoistable by K-1b)
    # body_share = T_body / T_slot  (token pipeline; attackable by K-1a)
    # sync_share = T_sync / T_slot  (slot-end __syncthreads + finish-flag IBGDA;
    #                                NOT hoistable by K-1b — this is the
    #                                residual that Sprint B probe v1 lumped in)
    init_share = None
    body_share = None
    sync_share = None
    slot_mean = slot.get("mean") if slot else None
    if slot_mean and slot_mean > 0:
        if init and init.get("mean") is not None:
            init_share = init["mean"] / slot_mean
        if body and body.get("mean") is not None:
            body_share = body["mean"] / slot_mean
        if sync and sync.get("mean") is not None:
            sync_share = sync["mean"] / slot_mean
    if init is None and body is None and sync is None:
        notes.append("v2 decomposition: no T_init/T_body/T_sync rows (v1 probe log)")

    return {
        "put_ratio": put_ratio,
        "sm_overhead_ratio": sm_overhead_ratio,
        "sm_spread": sm_spread,
        "n_slots_avg": n_slots_avg,
        "init_share": init_share,
        "body_share": body_share,
        "sync_share": sync_share,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Decision rules (copied literally from mechanism_probe_plan.md)
# ---------------------------------------------------------------------------

# Each rule: (variant, condition_desc, trigger_fn(indicators) -> (fired, evidence_str))
# Thresholds from the plan:
#   put_ratio > 0.40            -> K-1a
#   sm_overhead_ratio > 1.30    -> K-1b
#   sm_spread "dominates"       -> K-1c (interpreted as sm_spread > 0.30,
#                                  i.e. max exceeds p50 by 30% or more;
#                                  this captures "tail dominates wall-time")

PUT_RATIO_THRESHOLD = 0.40
SM_OVERHEAD_THRESHOLD = 1.30
SM_SPREAD_THRESHOLD = 0.30


def _rule_k1a(ind):
    pr = ind["put_ratio"]
    if pr is None:
        return (False, "put_ratio unavailable")
    fired = pr > PUT_RATIO_THRESHOLD
    return (fired, f"put_ratio={pr:.3f} {'>' if fired else '<='} {PUT_RATIO_THRESHOLD:.2f}")


def _rule_k1b(ind):
    """K-1b hoists mbarrier_init (T_init) out of the slot loop. Prefer the
    v2 indicator `init_share` (direct measurement); fall back to v1's
    `sm_overhead_ratio` for older probe logs. Sprint B K-1b A/B (2026-05-05)
    showed sm_overhead_ratio > 1.30 was too loose — v1's threshold fired
    on cells where T_init was only ~20% of T_slot and K-1b regressed decode
    by 8%. Tighten: K-1b only recommended when init_share > 0.20 (v2) or
    sm_overhead_ratio > 1.30 AND ntok > 256 (v1 fallback)."""
    init_share = ind.get("init_share")
    if init_share is not None:
        fired = init_share > 0.20
        return (fired, f"init_share={init_share:.3f} {'>' if fired else '<='} 0.20 (v2 direct)")
    # v1 fallback (older probe logs without T_init)
    sor = ind["sm_overhead_ratio"]
    if sor is None:
        return (False, "both init_share and sm_overhead_ratio unavailable")
    fired = sor > SM_OVERHEAD_THRESHOLD
    return (fired, f"sm_overhead_ratio={sor:.3f} {'>' if fired else '<='} {SM_OVERHEAD_THRESHOLD:.2f} (v1 fallback)")


def _rule_k1c(ind):
    ss = ind["sm_spread"]
    if ss is None:
        return (False, "sm_spread unavailable")
    fired = ss > SM_SPREAD_THRESHOLD
    return (fired, f"sm_spread={ss:.3f} {'>' if fired else '<='} {SM_SPREAD_THRESHOLD:.2f}")


DECISION_RULES = [
    ("K-1a", "T_put / T_slot > 40% (token-intra-slot serialization)", _rule_k1a),
    ("K-1b", "T_sm / (n_slots × T_slot) > 1.3 (slot-inter-sync overhead)", _rule_k1b),
    ("K-1c", "T_SM.max - T_SM.p50 dominates (tail SM straggler)", _rule_k1c),
]


def apply_rules(ind):
    """Apply all three rules; return list of (variant, fired, evidence, desc)."""
    results = []
    for variant, desc, fn in DECISION_RULES:
        fired, evidence = fn(ind)
        results.append({
            "variant": variant,
            "desc": desc,
            "fired": fired,
            "evidence": evidence,
        })
    return results


# ---------------------------------------------------------------------------
# Signal strength for final ranking
# ---------------------------------------------------------------------------

def _signal_strength(variant, ind):
    """Return a normalized "how strongly did this fire" score, or None."""
    if variant == "K-1a":
        v = ind["put_ratio"]
        thr = PUT_RATIO_THRESHOLD
    elif variant == "K-1b":
        v = ind["sm_overhead_ratio"]
        thr = SM_OVERHEAD_THRESHOLD
    elif variant == "K-1c":
        v = ind["sm_spread"]
        thr = SM_SPREAD_THRESHOLD
    else:
        return None
    if v is None:
        return None
    # Relative excess over threshold; negative if not fired.
    return (v - thr) / thr


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(v, spec=".3f"):
    if v is None:
        return "n/a"
    if isinstance(v, float) and (v != v):  # NaN
        return "nan"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def print_config(config):
    print("=== PROBE CONFIG ===")
    if not config:
        print("  (no PROBE_CONFIG line found in inputs)")
        return
    order = ["sm_clock_khz", "hidden", "num_topk", "num_experts", "world_size",
             "probe_iters", "ntok_list", "sms_list", "probe_enabled"]
    for k in order:
        if k in config:
            print(f"  {k:16s} = {config[k]}")
    for k, v in config.items():
        if k not in order:
            print(f"  {k:16s} = {v}")


def print_aggregation_table(cells):
    print()
    print("=== AGGREGATION (per (ntok, nsms), mean of per-rank stats) ===")
    hdr = (
        f"{'ntok':>5} {'nsms':>5} "
        f"{'T_put.mean':>11} {'T_put.p99':>10} "
        f"{'T_slot.mean':>12} {'T_slot.p99':>11} "
        f"{'T_sm.mean':>10} {'T_sm.max':>9} "
        f"{'n_slots':>8} "
        f"{'put_ratio':>10} {'sm_ovhd':>8} {'sm_sprd':>8} "
        f"{'init%':>6} {'body%':>6} {'sync%':>6} "  # v2 shares; n/a on v1
        f"{'ranks':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    cell_indicators = {}
    for key in sorted(cells.keys()):
        ntok, nsms = key
        mechs = cells[key]
        put = mechs.get("T_put", {})
        slot = mechs.get("T_slot", {})
        sm = mechs.get("T_sm", {})
        ind = derive_indicators(mechs)
        cell_indicators[key] = ind
        # Collect the union of rank ids across all three mechs for display.
        rset = set()
        for m in (put, slot, sm):
            for r in m.get("ranks", []) or []:
                rset.add(r)
        ranks_str = ",".join(str(r) for r in sorted(rset)) if rset else "-"
        print(
            f"{ntok:>5} {nsms:>5} "
            f"{_fmt(put.get('mean')):>11} {_fmt(put.get('p99')):>10} "
            f"{_fmt(slot.get('mean')):>12} {_fmt(slot.get('p99')):>11} "
            f"{_fmt(sm.get('mean')):>10} {_fmt(sm.get('max')):>9} "
            f"{_fmt(ind['n_slots_avg'], '.2f'):>8} "
            f"{_fmt(ind['put_ratio']):>10} "
            f"{_fmt(ind['sm_overhead_ratio']):>8} "
            f"{_fmt(ind['sm_spread']):>8} "
            f"{_fmt(ind['init_share'], '.1%'):>6} "
            f"{_fmt(ind['body_share'], '.1%'):>6} "
            f"{_fmt(ind['sync_share'], '.1%'):>6} "
            f"{ranks_str:>6}"
        )
    # Print any per-cell notes (missing-data) below the table.
    any_notes = False
    for key, ind in cell_indicators.items():
        if ind["notes"]:
            if not any_notes:
                print()
                print("Notes (missing / inconsistent data):")
                any_notes = True
            for note in ind["notes"]:
                print(f"  (ntok={key[0]}, nsms={key[1]}): {note}")
    return cell_indicators


def print_decision_table(cells, cell_indicators):
    print()
    print("=== DECISION TABLE (per (ntok, nsms), all fired rules) ===")
    hdr = f"{'ntok':>5} {'nsms':>5}  {'variant':<6}  {'fired':<5}  evidence"
    print(hdr)
    print("-" * len(hdr))
    fired_by_cell: dict = {}
    for key in sorted(cells.keys()):
        ntok, nsms = key
        ind = cell_indicators[key]
        results = apply_rules(ind)
        fired_by_cell[key] = results
        for rr in results:
            mark = "YES" if rr["fired"] else "no"
            print(
                f"{ntok:>5} {nsms:>5}  {rr['variant']:<6}  {mark:<5}  "
                f"{rr['evidence']}  [{rr['desc']}]"
            )
    return fired_by_cell


PREFILL_NTOKS = {256, 512}


def print_recommendation(cells, cell_indicators, fired_by_cell):
    print()
    print("=== K-1 RECOMMENDATION SUMMARY (prefill cells: ntok in {256, 512}) ===")
    prefill_cells = [k for k in cells if k[0] in PREFILL_NTOKS]
    if not prefill_cells:
        print("  No prefill (ntok in {256, 512}) cells in the data.")
        print("  Cannot recommend a K-1 variant without prefill coverage.")
        print()
        print("==> K-1 RECOMMENDATION: INSUFFICIENT DATA (no prefill cells)")
        return

    # For each variant, collect signal strength across prefill cells where we
    # had enough data to evaluate. Rank by (fire_count, mean_strength).
    per_variant: dict = {v: {"fires": 0, "evaluated": 0, "strengths": []}
                         for v, _, _ in DECISION_RULES}
    for key in prefill_cells:
        ind = cell_indicators[key]
        for rr in fired_by_cell[key]:
            v = rr["variant"]
            s = _signal_strength(v, ind)
            if s is None:
                continue
            per_variant[v]["evaluated"] += 1
            per_variant[v]["strengths"].append(s)
            if rr["fired"]:
                per_variant[v]["fires"] += 1

    # Any variant with zero evaluated prefill cells means missing data; flag.
    missing = [v for v, d in per_variant.items() if d["evaluated"] == 0]
    if missing:
        print(f"  WARNING: no prefill data to evaluate variants: {', '.join(missing)}")

    # Build ranking list of variants that fired at least once in prefill.
    rankable = []
    for v, d in per_variant.items():
        if d["fires"] == 0 or not d["strengths"]:
            continue
        mean_strength = statistics.fmean(d["strengths"])
        rankable.append((v, d["fires"], d["evaluated"], mean_strength))

    if not rankable:
        print("  No rules fired on any prefill cell.")
        print("  Either the regression is not captured by the three mechanisms,")
        print("  or probe data is inconsistent. Review the aggregation table.")
        print()
        print("==> K-1 RECOMMENDATION: NONE (no prefill rule fired)")
        return

    # Sort: most fires first, then strongest mean signal.
    rankable.sort(key=lambda t: (t[1], t[3]), reverse=True)
    print()
    print(f"  {'variant':<6} {'fires':>6} {'evaluated':>10} {'mean_strength':>14}")
    for v, fires, evald, ms in rankable:
        print(f"  {v:<6} {fires:>6} {evald:>10} {ms:>14.3f}")

    top_variant, top_fires, top_evald, top_strength = rankable[0]
    if len(rankable) > 1 and rankable[1][1] == top_fires \
            and abs(rankable[1][3] - top_strength) < 0.05:
        tie_variants = [top_variant] + [r[0] for r in rankable[1:]
                                        if r[1] == top_fires
                                        and abs(r[3] - top_strength) < 0.05]
        print()
        print(f"==> K-1 RECOMMENDATION: TIE between {', '.join(tie_variants)}")
        print("    (equal fire count and comparable signal strength; consider")
        print("     implementation cost per the k1_prototype_sketch.md).")
        return

    print()
    print(f"==> K-1 RECOMMENDATION: {top_variant}  "
          f"(fired in {top_fires}/{top_evald} prefill cells, "
          f"mean strength {top_strength:+.3f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Parse bench_overlap.py --mode=probe logs and emit a K-1 "
                    "decision table.",
    )
    parser.add_argument("logs", nargs="+",
                        help="probe log files (e.g. probe-r0.log probe-r1.log)")
    args = parser.parse_args(argv)

    config, rows, warnings = parse_logs(args.logs)
    if warnings:
        print("=== PARSE WARNINGS ===")
        for w in warnings:
            print(f"  {w}")
        print()

    if not rows:
        print("No PROBE rows parsed from input files.", file=sys.stderr)
        print("Nothing to analyze. Check that --mode=probe actually ran and",
              file=sys.stderr)
        print("that stderr was captured into the log files.", file=sys.stderr)
        return 2

    print_config(config)
    cells = aggregate(rows)
    if not cells:
        print()
        print("No (ntok, nsms, mech) cells could be formed from the data.")
        return 2

    cell_indicators = print_aggregation_table(cells)
    fired_by_cell = print_decision_table(cells, cell_indicators)
    print_recommendation(cells, cell_indicators, fired_by_cell)
    return 0


if __name__ == "__main__":
    sys.exit(main())
