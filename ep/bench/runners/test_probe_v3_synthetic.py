"""Synthetic unit tests for probe v3 analyzer.

Does NOT require CUDA / uccl extension. Builds fake probe dicts
that mimic what _parse_probe_buffer returns, feeds them through
_probe_summary, and asserts the derived metrics match three
expected regimes:

    SEND-dominated:   T_recv_wait << T_sm-T_recv_wait → critical_path_share low
    RECV-dominated:   T_recv_wait ≈ T_sm → critical_path_share high
    Balanced:         half-half

Run:
    cd uccl/ep/bench/runners && python3 test_probe_v3_synthetic.py

Exit code 0 = all assertions pass. The test is gate material for
the probe v3 PR — any future schema change that breaks these
expectations should fail here before landing on a GPU session.
"""
from __future__ import annotations

import os
import sys
import numpy as np

# Make the bench package importable without installing.
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, BENCH)

# We only need _probe_summary — importing bench_overlap pulls in torch/uccl.
# Work around by re-parsing the file and pulling the function out.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "bench_overlap_probe_only",
    os.path.join(BENCH, "bench_overlap.py"),
)
# Fake torch/uccl so the import doesn't blow up.
sys.modules.setdefault("torch", _FakeTorch := type(sys)("torch"))
_FakeTorch.Tensor = type("T", (), {})
_FakeTorch.int64 = 0
_FakeTorch.int32 = 0
_FakeTorch.distributed = type(sys)("torch.distributed")
_FakeTorch.cuda = type(sys)("torch.cuda")
uccl_mod = type(sys)("uccl")
ep_mod = type(sys)("uccl.ep")
sys.modules["uccl"] = uccl_mod
sys.modules["uccl.ep"] = ep_mod
uccl_mod.ep = ep_mod

try:
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
except Exception as e:
    # bench_overlap has top-level imports we don't need — extract
    # _probe_summary via direct exec of a sliced source so we avoid
    # pulling in heavy deps.
    src = open(os.path.join(BENCH, "bench_overlap.py")).read()
    # Grab from def _probe_summary to the following def build_inputs.
    start = src.find("def _probe_summary")
    end = src.find("\ndef build_inputs")
    snippet = src[start:end]
    ns = {"np": np}
    exec(snippet, ns)
    _probe_summary = ns["_probe_summary"]
else:
    _probe_summary = module._probe_summary


# ---------------------------------------------------------------------------
# Fake probe builder
# ---------------------------------------------------------------------------

def _mk_probe(
    n_sms: int,
    n_slots_per_sm: int,
    n_recv_slots_per_sm: int,
    send_cycles_per_slot: int,
    recv_wait_cycles: int,
    recv_reduce_cycles: int,
    schema: int = 3,
) -> dict:
    """Build a fake probe dict with deterministic timestamps.

    All cycles are integers. Each SM's timeline:
      sm_start = 0
      slot_start[s] = s * send_cycles_per_slot
      slot_end[s]   = (s+1) * send_cycles_per_slot  (SEND takes N cycles)
      SEND total = n_slots_per_sm * send_cycles_per_slot

      recv_wait_start[r] = SEND_end + r * (recv_wait + recv_reduce)
      recv_wait_end[r]   = recv_wait_start[r] + recv_wait_cycles
      recv_reduce_end[r] = recv_wait_end[r] + recv_reduce_cycles

      sm_end = last recv_reduce_end
    """
    max_sms = 128
    max_slots = 64
    shape = (max_sms, max_slots)
    z64 = np.zeros(shape, dtype=np.int64)
    zi32 = np.zeros(shape, dtype=np.int32)

    sm_start = np.zeros(max_sms, dtype=np.int64)
    sm_end = np.zeros(max_sms, dtype=np.int64)
    slot_start = z64.copy()
    slot_end = z64.copy()
    put_start = z64.copy()
    put_end = z64.copy()
    n_slots = np.zeros(max_sms, dtype=np.int32)
    slot_body_start = z64.copy()
    slot_body_end = z64.copy()
    sync_start = z64.copy()
    sync_end = z64.copy()
    recv_wait_start = z64.copy()
    recv_wait_end = z64.copy()
    recv_reduce_end = z64.copy()
    recv_src_rank = zi32.copy()
    n_recv_slots = np.zeros(max_sms, dtype=np.int32)

    for sm in range(n_sms):
        sm_start[sm] = 0
        cur = 0
        for s in range(n_slots_per_sm):
            slot_start[sm, s] = cur
            slot_body_start[sm, s] = cur + 10
            slot_body_end[sm, s] = cur + send_cycles_per_slot - 20
            sync_start[sm, s] = slot_body_end[sm, s]
            sync_end[sm, s] = cur + send_cycles_per_slot
            slot_end[sm, s] = sync_end[sm, s]
            put_start[sm, s] = slot_body_start[sm, s] + 1
            put_end[sm, s] = slot_body_end[sm, s] - 1
            cur += send_cycles_per_slot
        n_slots[sm] = n_slots_per_sm

        for r in range(n_recv_slots_per_sm):
            recv_wait_start[sm, r] = cur
            recv_wait_end[sm, r] = cur + recv_wait_cycles
            recv_reduce_end[sm, r] = cur + recv_wait_cycles + recv_reduce_cycles
            recv_src_rank[sm, r] = r  # trivial per-peer distribution
            cur += recv_wait_cycles + recv_reduce_cycles
        n_recv_slots[sm] = n_recv_slots_per_sm
        sm_end[sm] = cur

    return dict(
        sm_start=sm_start,
        sm_end=sm_end,
        slot_start=slot_start,
        slot_end=slot_end,
        put_start=put_start,
        put_end=put_end,
        n_slots=n_slots,
        slot_body_start=slot_body_start,
        slot_body_end=slot_body_end,
        sync_start=sync_start,
        sync_end=sync_end,
        recv_wait_start=recv_wait_start,
        recv_wait_end=recv_wait_end,
        recv_reduce_end=recv_reduce_end,
        recv_src_rank=recv_src_rank,
        n_recv_slots=n_recv_slots,
        schema_version=schema,
    )


# ---------------------------------------------------------------------------
# Test regimes
# ---------------------------------------------------------------------------

SM_CLOCK_KHZ = 1_980_000  # H200 boost clock used in prod probes


def _run(name, *, send, recv_wait, recv_reduce, n_sms=16, n_slots=4, n_recv=4,
         expected_share: float, tol: float = 0.05):
    probe = _mk_probe(
        n_sms=n_sms,
        n_slots_per_sm=n_slots,
        n_recv_slots_per_sm=n_recv,
        send_cycles_per_slot=send,
        recv_wait_cycles=recv_wait,
        recv_reduce_cycles=recv_reduce,
    )
    out = _probe_summary([probe], SM_CLOCK_KHZ)
    share = out.get("critical_path_share")
    assert share is not None, f"{name}: missing critical_path_share"
    assert abs(share - expected_share) < tol, (
        f"{name}: critical_path_share={share:.3f}, expected ≈{expected_share:.2f}"
    )
    # Sanity: T_recv_wait stats exist and have the right median.
    recv_stats = out.get("T_recv_wait")
    assert recv_stats is not None, f"{name}: missing T_recv_wait"
    expected_us = recv_wait / SM_CLOCK_KHZ / 1000.0 * 1e6
    got_p50 = recv_stats["p50"]
    assert abs(got_p50 - expected_us) / expected_us < 0.01, (
        f"{name}: T_recv_wait.p50={got_p50:.2f}us, expected {expected_us:.2f}us"
    )
    # T_sync metric should still be reported (v2 field) even though
    # it's semantically suspect — backward compatibility.
    assert out.get("T_sync") is not None, f"{name}: T_sync missing (v2 compat)"
    print(f"  {name}: share={share:.3f} (expected ≈{expected_share:.2f}) OK")


def main():
    print("=== Probe v3 synthetic unit tests ===")

    # 1. SEND-dominated: most time spent in SEND, recv wait is tiny.
    #    4 slots × 10000 cycles SEND, 4 recv × 500 cycles wait.
    #    T_sm = 40000 + 4*(500+500) = 44000
    #    sum_recv_wait per SM = 4*500 = 2000 → share ≈ 0.045
    _run("SEND-dominated",
         send=10_000, recv_wait=500, recv_reduce=500,
         expected_share=2000 / 44000)

    # 2. RECV-dominated: tiny SEND, huge recv wait.
    #    4 slots × 500 SEND, 4 recv × 10000 wait + 500 reduce.
    #    T_sm = 2000 + 4*(10000+500) = 44000
    #    sum_recv_wait = 40000 → share ≈ 0.909
    _run("RECV-dominated",
         send=500, recv_wait=10_000, recv_reduce=500,
         expected_share=40000 / 44000)

    # 3. Balanced: equal SEND and RECV wait.
    #    4 slots × 5000 SEND, 4 recv × 5000 wait + 500 reduce.
    #    T_sm = 20000 + 4*5500 = 42000
    #    sum_recv_wait = 20000 → share ≈ 0.476
    _run("Balanced",
         send=5_000, recv_wait=5_000, recv_reduce=500,
         expected_share=20000 / 42000)

    # 4. Backward-compat: v2-only probe (schema=2) still parses without
    #    error; the v3 metrics should simply not appear.
    probe_v2 = _mk_probe(
        n_sms=8, n_slots_per_sm=4, n_recv_slots_per_sm=4,
        send_cycles_per_slot=10_000, recv_wait_cycles=500,
        recv_reduce_cycles=500, schema=2,
    )
    # Drop v3 fields so the "has_v3" branch is skipped by schema check.
    for k in ("recv_wait_start", "recv_wait_end", "recv_reduce_end",
              "recv_src_rank", "n_recv_slots"):
        probe_v2.pop(k, None)
    out = _probe_summary([probe_v2], SM_CLOCK_KHZ)
    assert out["schema_version"] == 2, \
        f"v2-compat: schema={out['schema_version']}"
    assert out.get("critical_path_share") is None, \
        "v2-compat: critical_path_share should be absent"
    assert out.get("T_body") is not None, \
        "v2-compat: T_body should still be reported"
    print("  v2-compat: schema=2 probe parsed OK, no v3 metrics leaked")

    # 5. Sanity: K-T_sync would have scored zero predicted improvement.
    #    Model: same SEND cost reduced by the "T_sync 50%" slice (fake
    #    10000-cycle "savings" inside SEND), but recv_wait identical to
    #    pre-change. The share should be unchanged because RECV is the
    #    critical path.
    probe_pre = _mk_probe(
        n_sms=16, n_slots_per_sm=4, n_recv_slots_per_sm=4,
        send_cycles_per_slot=10_000, recv_wait_cycles=20_000,
        recv_reduce_cycles=500,
    )
    probe_post = _mk_probe(
        n_sms=16, n_slots_per_sm=4, n_recv_slots_per_sm=4,
        send_cycles_per_slot=5_000,  # ← K-T_sync "saves" 50% of SEND
        recv_wait_cycles=20_000,
        recv_reduce_cycles=500,
    )
    pre = _probe_summary([probe_pre], SM_CLOCK_KHZ)
    post = _probe_summary([probe_post], SM_CLOCK_KHZ)
    assert pre["critical_path_share"] < post["critical_path_share"], (
        "K-T_sync analogue: halving SEND should RAISE critical_path_share "
        "because RECV becomes a larger relative fraction of a shorter "
        "T_sm — proves the metric sees RECV as the binding constraint."
    )
    print(f"  K-T_sync sanity: pre share={pre['critical_path_share']:.3f}, "
          f"post share={post['critical_path_share']:.3f} "
          f"(post > pre → RECV is binding, SEND cut doesn't move wall-time)")

    print("\nALL PASS")


if __name__ == "__main__":
    main()
