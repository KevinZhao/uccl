# Phase 10/11 analysis template

Once logs from `run_phase_10_11.sh` are scp'd to
`efa-validation/results/stage5-p5en/sprint-b-adaptive-probev2-<STAMP>/`,
fill in the below.

## Phase 10 — adaptive num_sms validation

### Run record
- session stamp: `<STAMP>`
- region / AZ: `<region>/<az>` (SPS at launch: `<score>`)
- cluster / nodegroup: `<cluster>/<ng>`
- nodes (instance-id, private IP, leaf-id):
  - rank 0: `<i-…>` / `<ip>` / `<leaf>`
  - rank 1: `<i-…>` / `<ip>` / `<leaf>`
- commits on nodes: `<short-sha>` (expect `14fd7dc3` or later on
  `feat/sprint-a-generalization-bench`)
- per-rank log rows: `$(grep -c '^BENCH ' workload-static-r0.log)`
  static, `$(grep -c '^BENCH ' workload-adaptive-r0.log)` adaptive
  (should both be 16 ranks × (6 ntok × (dispatch+combine-base+N×overlap)
  × 20 iter / 2 files) ≈ 4800 for static and 2880 for adaptive)

### Gate G1–G4 evaluation
Run:
```
cd uccl/ep/bench/runners
python3 analyze_adaptive.py \
  --static ../../../efa-validation/results/stage5-p5en/sprint-b-adaptive-probev2-<STAMP>/workload-static-r0.log \
           ../../../efa-validation/results/stage5-p5en/sprint-b-adaptive-probev2-<STAMP>/workload-static-r1.log \
  --adaptive ../../../efa-validation/results/stage5-p5en/sprint-b-adaptive-probev2-<STAMP>/workload-adaptive-r0.log \
             ../../../efa-validation/results/stage5-p5en/sprint-b-adaptive-probev2-<STAMP>/workload-adaptive-r1.log \
  --iter-warmup 2 \
  --csv ../../../efa-validation/results/stage5-p5en/sprint-b-adaptive-probev2-<STAMP>/adaptive-grid.csv
```

Paste the final G1–G4 block below.

### Decision
- G1 result: PASS / FAIL, affected cells: `<list>`
- G2 result: PASS / FAIL (decode preserved?)
- G3 result: PASS / FAIL (prefill preserved?)
- G4 result: PASS / FAIL (384 boundary OK?)
- **Tier table verdict**: keep / adjust / rollback
  - If adjust: proposed new tiers (ntok, num_sms): `…`

## Phase 11 — probe v2 decomposition

### Run record
- commit on node: `<short-sha>` (must match Phase 10 run)
- build flag: `UCCL_EP_PROBE=1` (verify
  `probe_buffer_bytes()==526912`)
- Gate B result: `max diff=0, nonzero_frac=0` across all num_sms? Yes/No
- probe log rows: `$(grep -c '^PROBE ' probev2-r0.log)` r0, same r1

### Gate P1–P4 evaluation
```
python3 gate_probev2.py \
  ../../../efa-validation/results/stage5-p5en/sprint-b-adaptive-probev2-<STAMP>/probev2-r0.log \
  ../../../efa-validation/results/stage5-p5en/sprint-b-adaptive-probev2-<STAMP>/probev2-r1.log \
  | tee ../../../efa-validation/results/stage5-p5en/sprint-b-adaptive-probev2-<STAMP>/gate_probev2_out.txt
```

Paste the per-cell init/body/sync table + summary here.

### Decision (which kernel variant next?)
Apply the decision rule:
- if `init_share` dominates (>0.20) in decode cells → **K-1a**
  (hoist mbarrier_init only, no persistent phase — K-1b's
  register-pressure trap avoided)
- if `sync_share ≥ 0.10` in any cell → **K-T_sync** (overlap
  finish-flag IBGDA atomic with next slot)
- if `body_share ≥ 0.50` across prefill cells AND `put_ratio > 0.7`
  → **K-1c** (enlarge slot pipelining or token fanout)

**Verdict**: `…`

### Raw full-grid analyzer output
```
python3 analyze_probe.py \
  ../../../efa-validation/results/stage5-p5en/sprint-b-adaptive-probev2-<STAMP>/probev2-r0.log \
  ../../../efa-validation/results/stage5-p5en/sprint-b-adaptive-probev2-<STAMP>/probev2-r1.log \
  | tee ../../../efa-validation/results/stage5-p5en/sprint-b-adaptive-probev2-<STAMP>/analyze_probev2_out.txt
```

Paste the aggregation table here.

## Cross-session comparison (Sprint A → B → this session)

| metric | Sprint A | Sprint B (K-1b) | this session |
|---|---|---|---|
| p99 decode (ntok=128) | 40147 µs | ~41k (K-1b regressed) | `<val>` µs (adaptive) |
| p99 prefill (ntok=512, best static) | 40368 µs | ~40k | `<val>` µs (adaptive) |
| init_share @ decode | unknown | ~N/A (v1 lump) | `<share>` |
| sync_share @ prefill | unknown | ~N/A (v1 lump) | `<share>` |

## Next PR scope
Based on Phase 10 verdict:
- If adaptive G1 PASS: land PR to upstream UCCL with `num_sms=0`
  default for `low_latency_combine(overlap=True)`
- If adjust: iterate tiers offline, no PR yet; add a second session
  at the new boundary

Based on Phase 11 decision:
- K-1a / K-T_sync / K-1c: design sketch in
  `runners/k1_prototype_sketch.md` style doc before burning another
  session

## Cost and teardown log
- total p5en uptime: `<minutes>`
- estimated spend: `<$>` (8/hr × 2 × hours)
- scp size: `<MB>`
- teardown verified at `<UTC>` (nodes Terminated, NG scaled to 0)
