# K-1 Kernel Improvement Prototypes (awaiting probe data)

This file stays a SKETCH until we have probe measurements that tell us
which mechanism dominates. Once Sprint B collects probe data via
`--mode=probe`, we pick ONE variant below to turn into a kernel diff.

## Why not implement all three speculatively

Each variant rearranges TMA phase parity, mbarrier lifecycle, or warp
synchronization. Sprint A Gate B failed bit-exact once already when we
were careful; implementing all three and hoping one works is a 4+ day
rabbit hole with high regression risk. Probe data reduces that to 1 day:
the dominant mechanism's T_xxx delta tells us which reordering is
guaranteed to move the needle.

## Variant K-1a — token-intra-slot pipeline

**Idea**: In the inner token loop of each slot, overlap token N's
`ibgda_put` with token N+1's TMA load + compute. Producer-consumer warp
split within a warp-group.

**Kernel change locations**:
- `internode_ll.cu` line ~861..1089 (SEND body per token) —
  split the `for (int token_idx...)` loop into producer/consumer phases.
- `tma_buffer` staging currently has `kNumStages=3`; would need to bump
  to cover up to 2 tokens' worth of in-flight stages.

**Invariants to preserve for Gate B correctness**:
- Each token's `tma_store_1d` must have drained to the send-buffer
  before its `ibgda_put` reads the buffer. Current code enforces this
  via `tma_store_wait();` at line ~1085. In K-1a the wait must be moved
  to **before** each token's put, not before the slot's end.
- `__threadfence_system()` before `ibgda_put` (line ~1093) is required
  for the NIC to see the data. Must stay, but may apply to a different
  warp subset in K-1a.

**Probe signature that picks K-1a**:
`T_put.mean / T_slot.mean > 0.4` in the losing prefill config and
`T_put.mean / T_slot.mean < 0.2` in the winning decode config. Means the
NIC window is what eats the slot wall-time in prefill, and hiding it
behind compute is the right lever.

## Variant K-1b — slot-inter TMA prefetch

**Idea**: At slot end, before `__syncthreads()`, kick off the TMA load
for slot+1's first token. Move `mbarrier_init` from slot head to slot
tail, reset phase parity at the same time.

**Kernel change locations**:
- `internode_ll.cu` line ~885..909 (TMA stage init) — hoist the init
  out of the slot body and run it once before the slot loop, then rotate
  phase parity at the slot boundary.
- Line ~886 currently has the Sprint A `cp.async.bulk.wait_group 0`
  drain — K-1b would remove this (the new phase parity logic handles
  leftover in-flight loads).

**Invariants to preserve**:
- `atomic_clean_flag` decrement (line ~1156) must be visible to the
  **next** slot's reader. Either keep `__syncthreads()` or replace with
  `__threadfence_block()` + explicit fence.
- `finish_counter_per_expert` logic (if any) must be scoped per slot;
  can't leak across.

**Probe signature that picks K-1b**:
`T_SM / (n_slots × T_slot.mean) > 1.3` indicating ~30% of per-SM time
is spent between slots (setup + syncs + atomic), not inside slots.
Equivalently: slot gap dominates over slot body in prefill.

## Variant K-1c — finish-flag warp de-sync

**Idea**: `__syncthreads()` at line ~1162 is there to ensure all warps
see `atomic_clean_flag` decrement before starting the next slot. But
only the finish-flag writer warp (sub_warp_id==1 in line ~1115) actually
writes. Keep sync scoped to that warp + warp-group; let compute warps
advance.

**Kernel change locations**:
- `internode_ll.cu` line ~1161..1166 — replace `__syncthreads()` with a
  scoped `sync_barrier` + per-slot-stage mbarrier.
- `tma_phase[stage_idx]` register array would need to track per-slot
  rotation (currently reset every slot via mbarrier_init).

**Invariants to preserve**:
- No warp can re-init `tma_mbarrier[stage_idx]` while another warp has
  an outstanding TMA arrival pending on it. Today `__syncthreads()`
  handles this trivially; K-1c needs explicit stage phase management.

**Probe signature that picks K-1c**:
`T_SM.max - T_SM.median` > 10% of `T_SM.median` in the losing prefill
config. Means a stray SM is holding up the whole launch — and the sync
is the reason.

## Shared risk — Gate B bit-exact

Any of K-1a/b/c changes the dataflow between TMA store → fence → IBGDA
put. The Sprint A `test_baseline_analytical_oracle` in
`tests/test_low_latency_overlap.py` is the first gate. It must PASS for:
- overlap=True, num_sms ∈ {1, 2, 3, 4, 8, 16, 22, 48, 96}
- all of `(hidden × topk × num_experts) ∈ {(7168, 8, 288)}` (Sprint A
  config) plus the new `(8192, 8, 288)` and `(7168, 4, 60)` Sprint B
  configs

If Gate B fails, the analytical oracle directly flags which (SM, slot)
got the wrong reduction, so the probe's per-(SM, slot) clock data
helps triangulate the breakage.

## Gate C comparison matrix (after K-1 selection)

Once K-1 candidate compiles and passes Gate B:
- Rerun `--mode=workload --workload-tokens=128,256,512 --workload-sms=16,22,48,96`
  on both the **baseline kernel** (current `main` + Sprint A) and the
  **K-1 kernel**. Output goes to:
  ```
  results/sprint-b/baseline-r{0,1}.log
  results/sprint-b/k1-{variant}-r{0,1}.log
  ```
- Acceptance criterion for K-1 merge:
  - **decode (ntok=128) p99**: no worse than Sprint A's −29.6% win
  - **prefill (ntok=512) p99**: strictly improved over Sprint A
    (currently +17% vs baseline); ideal is ≤ 0% (no regression) or
    negative (net win)
  - **any (ntok, num_sms) cell**: K-1 must not make it > 5% worse than
    Sprint A pre-K-1 at the same cell

This is the "no regression across workload envelope" guarantee the PR
reviewer will demand.
