# Sprint B — mechanism-attribution probe plan

## Goal

Turn Sprint A's theoretical 5-mechanism breakdown into **measured**
per-mechanism contribution so that PR reviewers don't see "22 is a magic
number" — they see "here are the three contributing mechanisms and each
one's measured µs impact, and the fix targets mechanism X."

## What the probe captures

Compiled in only when `UCCL_EP_PROBE=1` is set at build time. When off,
the probe macros collapse to no-ops (zero runtime cost, zero memory use).

| Probe point       | Captures                   | Answers                       |
| ----------------- | -------------------------- | ----------------------------- |
| kernel entry/exit | `sm_end - sm_start` per SM | **D-4** load balance          |
| slot start/end    | `T_slot` per (SM, slot)    | **D-2** serial chain length   |
| put first/last    | `T_put` per (SM, slot)     | **D-1** NIC SQ window         |
| `n_slots` per SM  | actual slots processed     | SM stripe granularity sanity  |

`clock64()` is per-SM and not globally synchronized, so **every delta is
computed within the same SM** and never subtracted across SMs.

## How to build and run

```bash
# Build with probes (on the bench host after spot nodes come up).
cd uccl/ep
UCCL_EP_PROBE=1 python3 setup.py install

# Verify the binding is in:
python3 -c "from uccl import ep; print(ep.probe_buffer_enabled())"   # → True
python3 -c "from uccl import ep; print(ep.probe_buffer_bytes())"     # → e.g. 263232

# Run probe mode (2 nodes, 16 ranks). This captures 5 iters per
# (ntok, num_sms) config — small enough that statistics are meaningful
# without turning the run into a full bench.
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$R \
  --master_addr=$MASTER --master_port=12355 \
  bench_overlap.py --mode=probe \
    --probe-tokens=128,256,512 \
    --probe-sms=22,48,96 \
    --probe-iters=5 \
    --hidden=7168 --num-topk=8 --num-experts=288 \
  2> probe-r${R}.log
```

Each `PROBE` row in `probe-r*.log` is self-describing:
```
PROBE rank=0 ntok=128 nsms=22 mech=T_slot n=1650 mean=1.85 p50=1.70 p99=4.12 ...
```

## Decision tree: which K-1 to prototype

After collecting probe data at ntok ∈ {128, 256, 512} × num_sms ∈ {22, 96},
we compare the (ntok=512, nsms=22) **losing** configuration against the
(ntok=128, nsms=22) **winning** configuration:

| Observation                                                | Dominant mechanism              | K-1 variant          |
| ---------------------------------------------------------- | ------------------------------- | -------------------- |
| `T_put / T_slot` large (> 40%) in prefill                  | Token-intra-slot serialization  | **K-1a** token pipe  |
| `T_slot` stable but `T_sm / (n_slots × T_slot)` > 1.3      | Slot-inter-sync overhead        | **K-1b** slot prefetch |
| `T_SM.max - T_SM.median` dominates kernel wall-time        | Tail SM straggler               | **K-1c** work-steal  |

If `T_slot` itself scales linearly with ntok (expected from theory), the
delta from 128 → 512 breaks into three buckets:
```
ΔT_slot(ntok=128 → 512) ≈ ΔT_compute + ΔT_nic_wait + ΔT_setup
                          └──────┬──────┘  └──────┬─────┘  └──┬──┘
                    reading hidden × 4        NIC SQ depth     fixed
                    more tokens               rising
```
and the probe splits these because `T_put` is the NIC-visible component.

## K-1 variant sketches (not yet implemented)

### K-1a — token-intra-slot pipeline

**Idea**: Split warps within a warp-group. Producer warps do TMA load +
compute + TMA store; consumer warps do `tma_store_wait → threadfence →
ibgda_put`. Producer is `num_tokens_to_send` ahead so the NIC window
overlaps with the next token's compute.

**Risk**: The current loop reuses `tma_buffer[stage_idx]` within a token.
Extending reuse across tokens requires doubling the staging buffer or
reallocating phase parity. Gate B bit-exact regression likely if done
wrong.

**Expected win**: prefill `T_put` hidden behind next-token `T_compute`,
should cut per-slot wall time by 20-30% when tokens/slot ≥ 4.

### K-1b — slot-inter TMA prefetch

**Idea**: Before `__syncthreads()` at slot end, issue `tma_load_1d` for
the first hidden-chunk of the next slot's first token. `mbarrier_init`
has to move to the *head* of the slot rather than *after* `syncthreads`.

**Risk**: `atomic_clean_flag` decrement must be visible before next slot
reads it. Requires careful fence ordering. Currently `__syncthreads()`
serves that purpose — we'd replace it with a lighter barrier + explicit
fences.

**Expected win**: fills the ~1-5 µs slot-boundary gap per slot; cumulative
benefit is N_slot × gap, so bigger at high N_slot (22 SM / ntok=512 case).

### K-1c — finish-flag warp de-sync

**Idea**: Keep `__syncthreads()` for the finish-flag writer warp only, let
other warps start TMA load for the next slot immediately. This needs the
mbarrier array to be **per-slot-stage** so the new iteration doesn't race
with the old's in-flight TMA.

**Risk**: Phase-parity bookkeeping becomes per-(slot, stage) instead of
per-stage. Careful about `tma_phase[]` register rotation.

**Expected win**: similar to K-1b but cheaper to implement if the phase-
parity rework is tractable.

## Execution order

1. **Spot nodes up** → build with `UCCL_EP_PROBE=1` → run probe mode
2. **Analyze**: `probe-r0.log` + `probe-r1.log` → emit a decision table
   per (ntok, num_sms) config → pick K-1 variant
3. **Implement** selected K-1 on a fresh branch on top of this one
4. **Gate B regression**: rerun `test_low_latency_overlap.py` analytical
   oracle — must PASS for all probe-data-informed variants
5. **Gate C regression**: rerun `--mode=workload` and compare full
   (ntok, num_sms) grid vs pre-K-1 baseline
6. **Commit** only the winning K-1 variant + its probe evidence

## Caveats

- `clock64()` on Hopper boosts with SM frequency; use
  `cudaDeviceProp.clock_rate` (kHz) captured at kernel-launch time.
- Probe adds ~12 uint64 writes per slot, ~0.5% kernel-time overhead at
  num_sms=22. We do NOT use probe builds for the full regression matrix.
- `n_slots[sm]` upper bound is `kMaxSlotsPerSM = 64`. At num_sms=3 with
  num_experts=288 each SM runs 96 slots — that would overflow. For SM-
  stripe mode we always use num_sms ≥ 16 so max is `ceil(288/16) = 18`,
  well under 64.
