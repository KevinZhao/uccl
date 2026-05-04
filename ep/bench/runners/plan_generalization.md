# Sprint A generalization bench plan

Three tests run on each hardware tier. Each test reuses the same image/repo,
so runners only differ in the `bench_overlap.py` args. All tests are
workload-mode sweeps (dispatch-base + combine-base + combine-overlap-N per
config), and each configuration is self-described in the log's `CONFIG` line
and the per-row `num_tokens=/num_sms=/mode=` stamps.

## Test C — ntok density sweep (core)

Goal: nail the decode↔prefill transition curve on this hardware.

```
--mode=workload
--num-iters=20
--workload-tokens=32,64,128,256,512,1024
--workload-sms=16,22,32,48,64,96
--hidden=7168 --num-topk=8 --num-experts=288
--num-rdma-bytes=21474836480    # 20 GiB, sized for ntok=1024 hidden=7168
```

Expected rows per rank: 20 iters × 6 tokens × (1 dispatch + 1 base + 6 overlap)
= 960. × 16 ranks = 15360 rows.

## Test A — hidden sensitivity

Goal: confirm that bytes-per-slot scales linearly with hidden, and that the
num_sms sweet spot tracks it.

Run three separate torchruns (one per hidden), each with the same ntok/SM set:

```
hidden=4096:
  --workload-tokens=128,512 --workload-sms=16,22,48,96 --hidden=4096
  --num-topk=8 --num-experts=288 --num-rdma-bytes=10737418240
hidden=7168: (identical to Test C subset; can reuse those rows)
hidden=8192:
  --workload-tokens=128,512 --workload-sms=16,22,48,96 --hidden=8192
  --num-topk=8 --num-experts=288 --num-rdma-bytes=21474836480
```

## Test B — MoE granularity

Goal: confirm the per-slot occupancy formula. `tokens_per_slot` =
`ntok × topk / num_experts` varies ~7× between Mixtral-like and DS V3.

```
Mixtral-like:  --num-topk=2 --num-experts=8    (tokens_per_slot ≫)
Qwen MoE-like: --num-topk=4 --num-experts=60
DS V3 (same as C): --num-topk=8 --num-experts=288

Common: --workload-tokens=128,512 --workload-sms=16,22,48,96 --hidden=7168
Buffer: --num-rdma-bytes=10737418240 (10 GiB safe for all three)
```

Note: Buffer's `num_qps_per_rank = num_experts // world_size` — at
`num_experts=8, world_size=16` this is 0, which is invalid. For Mixtral we
need to override `world_size` semantics or run with a different expert count
that respects the `num_experts >= world_size` constraint. Fallback for
Mixtral: use `num_topk=2, num_experts=16` instead (1 expert per rank) to
approximate the coarse-grain regime on the same 2-node setup.

## Tier 1 (p5en usw2-az3) run order

1. Test C (30 min) — primary signal, supports all PR claims
2. Test A (20 min) — hidden scaling
3. Test B (25 min) — MoE granularity

## Tier 2 (p5 usw2, 4 NIC/GPU) run order

1. Test C only (30 min) — the ntok density curve alone answers the "does
   num_sms sweet spot scale with NIC/GPU?" question. Skip A/B here; hidden
   and MoE maxims are NIC-independent.
