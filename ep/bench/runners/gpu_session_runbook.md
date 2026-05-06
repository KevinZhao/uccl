# GPU Session Runbook — Sprint B mechanism probe + Gate B regression

Goal: a deterministic script of steps to go from "SPS ≥ 7" to "probe
data in hand + Gate B PASS + cluster torn down" with minimum human
attention. Each step has a success signal and a failure-mode note.

Keep this file next to the other Sprint A/B runners so it's always at
hand on the bastion.

## Pre-flight (code-server side, no cost)

Only proceed if:

1. `aws ec2 get-spot-placement-scores --instance-types p5en.48xlarge
   --target-capacity 2 --single-availability-zone --region-names
   <REGION>` returns Score=9 on at least one AZ. Prefer, in order:
   - `ap-northeast-1` (Sprint A baseline parity; preferred for cross-
     session comparability)
   - `us-west-2` (our Sprint A/B fallback; has `gpu-cluster-oregon`
     bastion already)
   - `us-east-2` (Oregon backup)
2. The target AZ has a matching single-subnet in the relevant
   `eks-cluster-deployment` .env (PRIVATE_SUBNET_A/B/C/D).
3. The cluster's launch template for `gpu-p5en-48xlarge-spot` exists
   and includes the 16 EFA NIC spec (16 NICs/node = 2 NICs/GPU × 8
   GPUs). Verify via:
   ```
   aws ec2 describe-launch-template-versions --region <REGION> \
     --launch-template-name gpu-cluster-oregon-gpu-p5en-48xlarge-spot-lt \
     --versions '$Latest' \
     --query 'LaunchTemplateVersions[0].LaunchTemplateData.NetworkInterfaces[].InterfaceType'
   ```
   Must return 1 × `efa` + 15 × `efa-only`.

## Phase 1 — start cluster (targeted ~6 min, cost starts)

### 1.1 Scale or create the single-AZ p5en nodegroup

If an existing single-AZ NG is available (`gpu-p5en-spot-usw2c`,
`gpu-p5en-spot-usw2d` in Oregon), **prefer scaling it**:

```bash
aws eks update-nodegroup-config \
  --cluster-name gpu-cluster-oregon --region us-west-2 \
  --nodegroup-name <single-AZ-p5en-ng> \
  --scaling-config minSize=2,maxSize=4,desiredSize=2
```

If no single-AZ NG exists, create one via EKS API (do **not** rely on
the multi-NG install script — it ignores GPU_INSTANCE_TYPES env and
builds every type in the default list):

```bash
aws eks create-nodegroup --region <REGION> \
  --cli-input-json file:///tmp/create-p5en-ng.json
```

where the JSON references the cluster's existing launch template ID and
the single subnet for the chosen AZ. Template:
```json
{
  "clusterName": "gpu-cluster-oregon",
  "nodegroupName": "gpu-p5en-spot-probe",
  "scalingConfig": {"minSize": 2, "maxSize": 2, "desiredSize": 2},
  "subnets": ["<PRIVATE_SUBNET_C>"],
  "instanceTypes": ["p5en.48xlarge"],
  "amiType": "CUSTOM",
  "capacityType": "SPOT",
  "nodeRole": "arn:aws:iam::<acct>:role/GPUNodeRole-gpu-cluster-oregon",
  "launchTemplate": {"id": "<lt-id>", "version": "<latest>"},
  "labels": {"gpu-instance-type": "p5en.48xlarge",
             "purchase-option": "spot", "workload-type": "gpu"},
  "taints": [{"key": "nvidia.com/gpu", "value": "true",
              "effect": "NO_SCHEDULE"}]
}
```

### 1.2 Wait for nodes to be Ready

```bash
aws eks describe-nodegroup --cluster-name ... --nodegroup-name ... \
  --query 'nodegroup.status' --output text
# loop until == "ACTIVE"

kubectl get nodes -l eks.amazonaws.com/nodegroup=<ng> \
  -L topology.k8s.aws/network-node-layer-3
# loop until both nodes Ready
```

### 1.3 Verify L3 leaf — hard gate

```bash
kubectl get nodes -l eks.amazonaws.com/nodegroup=<ng> \
  -o jsonpath='{range .items[*]}{.metadata.name} {.metadata.labels.topology\.k8s\.aws/network-node-layer-3}{"\n"}{end}'
```

- **Both nodes share layer-3** → proceed to Phase 2.
- **Different layer-3** → Scale NG to 0, wait for termination (EC2
  `describe-instances` filtered by InstanceLifecycle=spot), scale back
  to 2. Up to **2 retries**. If still different, accept the cross-leaf
  baseline and **mark probe data as "cross-L3 leaf"** in the result
  filename (`probe-cross-leaf-r*.log`).

SPOT QUOTA caveat: US accounts typically have a quota of 2 p5-class
spot. Do NOT try to run Tier 1 (p5en) and Tier 2 (p5) concurrently; run
them serially, and fully terminate Tier 1 before starting Tier 2.

## Phase 2 — build uccl.ep on the GPU nodes (targeted ~12 min)

SSM directly to the p5en nodes. The EKS CUSTOM AMI has SSM agent but
may take 60-120 s after Node-Ready before it accepts commands — if
commands stay `Pending`, wait another minute and retry before
assuming it's broken.

For each node:

```bash
# 1. Clone uccl at the probe branch
ssm send-command --instance-ids <p5en-r0> --document-name AWS-RunShellScript \
  --parameters 'commands=["
    set -eo pipefail
    cd /workspace || mkdir -p /workspace && cd /workspace
    rm -rf uccl
    git clone --branch feat/sprint-a-generalization-bench --depth 1 \
      https://github.com/KevinZhao/uccl.git
    cd uccl/ep
    UCCL_EP_PROBE=1 python3 setup.py install 2>&1 | tail -40
    python3 -c \"from uccl import ep; print(ep.probe_buffer_enabled(),
               ep.probe_buffer_bytes(),
               ep.probe_buffer_max_sms(),
               ep.probe_buffer_max_slots_per_sm())\"
  "]'
```

Success signal:
- `python3 -c ...` prints `True 264704 128 64`.

Failure modes to watch:
- `Failed to import uccl.ep` — setup.py didn't finish; inspect last 40
  lines, usually `-DEFA` was missing (libfabric not on LD path) or
  TORCH_CUDA_ARCH_LIST detection failed; rerun with explicit
  `TORCH_CUDA_ARCH_LIST=9.0`.
- Build took > 15 min — the Sprint A bench container's timings suggest
  ~5-10 min is normal; > 15 min usually means `setup.py` is re-
  compiling all sources. Check `git status` didn't leave stale `.o`.

## Phase 3 — run Gate B regression (probe build; targeted ~3 min)

Gate B is non-negotiable: probe must be a no-op at runtime when caller
doesn't pass a buffer, so the analytical oracle must still pass. This
is the correctness-before-speed gate.

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$R \
  --master_addr=$MASTER --master_port=12355 \
  tests/test_low_latency_overlap.py \
    --num-experts=288 --hidden=7168 --num-topk=8 \
    --num-sms-list=1,2,3,4,8,16,22,48,96 \
  2> gate-b-r${R}.log
grep -E "(PASS|FAIL|max diff|nonzero_frac)" gate-b-r${R}.log | tail -30
```

Success signal: every rank prints `max=0 nonzero_frac=0` for each
`num_sms`. Any `max != 0` is a Gate B regression — stop, teardown,
triage.

## Phase 4 — collect probe data (targeted ~12 min)

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$R \
  --master_addr=$MASTER --master_port=12355 \
  bench_overlap.py --mode=probe \
    --probe-tokens=128,256,512 \
    --probe-sms=22,48,96 \
    --probe-iters=5 \
    --hidden=7168 --num-topk=8 --num-experts=288 \
    --num-rdma-bytes=$((20 * 1024**3)) \
  2> probe-r${R}.log

echo "=== probe config + first 20 PROBE rows for sanity ==="
grep -E "^(PROBE_CONFIG|PROBE )" probe-r${R}.log | head -20
echo "=== expected row count: 3 ntok × 3 nsms × 3 mechs × 16 ranks = 432 ==="
grep -c "^PROBE " probe-r${R}.log
```

Success signal: `probe-r0.log` + `probe-r1.log` each contain
**216 `PROBE ` rows** (8 ranks × 3 ntok × 3 nsms × 3 mechs) plus one
`PROBE_CONFIG` + one `=== probe scan DONE ===`.

Failure mode:
- 0 rows → probe buffer was allocated but probe macros compiled out.
  Check `ep.probe_buffer_enabled()` returned True at Phase 2.
- Some cells have `n=0` → `T_put` cell empty may be expected (slots
  resolved to intranode IPC). Check that T_slot and T_sm are non-empty
  for the same (ntok, nsms) — those must be > 0 in all cells.

## Phase 5 — (optional) workload regression (targeted ~10 min)

Only run if Phase 3 and 4 passed and time permits. Confirms the probe
build doesn't break Sprint A's perf numbers.

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$R \
  --master_addr=$MASTER --master_port=12355 \
  bench_overlap.py --mode=workload \
    --workload-tokens=128,256,512 \
    --workload-sms=22,96 \
    --num-iters=10 \
    --hidden=7168 --num-topk=8 --num-experts=288 \
    --num-rdma-bytes=$((20 * 1024**3)) \
  2> workload-probe-r${R}.log
```

Compare `p99` in `workload-probe-r*.log` vs Sprint A's
`workload-r*.log`. Acceptance: probe build within 3% of Sprint A perf
for `mode=combine-overlap-22 num_tokens=128` (the decode winner).
Bigger than that means probe overhead is non-zero enough to skew
future bench runs.

## Phase 6 — scp logs out (code-server side, before teardown)

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${HOME}/workspace/efa-validation/results/stage5-p5en/sprint-b-probe-${STAMP}
mkdir -p "${OUT}"

for R in 0 1; do
  IP=<p5en-node-R-ip>
  ssm-scp <p5en-node-R>:/workspace/probe-r${R}.log "${OUT}/" &
  ssm-scp <p5en-node-R>:/workspace/gate-b-r${R}.log "${OUT}/" &
  ssm-scp <p5en-node-R>:/workspace/workload-probe-r${R}.log "${OUT}/" &
done
wait

# Quick sanity
wc -l "${OUT}"/*.log
```

If SSM doesn't support `scp` directly (most paths don't), fall back to
S3 staging:
```bash
aws s3 cp /workspace/probe-r0.log s3://yanxi-validation-788668107894-oregon/sprint-b/probe-${STAMP}/
# from code-server:
aws s3 cp --recursive s3://yanxi-validation-788668107894-oregon/sprint-b/probe-${STAMP}/ "${OUT}/"
```

## Phase 7 — TEARDOWN (critical: stops the spot burn)

```bash
aws eks update-nodegroup-config --cluster-name <cluster> --region <region> \
  --nodegroup-name <ng> \
  --scaling-config minSize=0,maxSize=4,desiredSize=0

# Force-delete K8s nodes to speed up EC2 terminate:
kubectl delete node <n1> <n2> --force --grace-period=0

# Confirm EC2 is actually terminated (critical):
aws ec2 describe-instances --region <region> \
  --filters "Name=instance-state-name,Values=running,pending,stopping" \
            "Name=instance-type,Values=p5en.48xlarge" \
  --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output table

# If any lingers > 5 min after scale-to-0, terminate explicitly:
aws ec2 terminate-instances --instance-ids <lingering-id> --region <region>
```

Success signal: the final `describe-instances` returns empty. Until
then, we are burning ~$16/hr for p5en spot.

## Phase 8 — analyze (code-server, no cost)

```bash
cd "${OUT}"
python3 ~/workspace/uccl-ep-optimization/uccl/ep/bench/runners/analyze_probe.py \
  probe-r0.log probe-r1.log > analyze_probe_out.txt
cat analyze_probe_out.txt
```

Output ends with `==> K-1 RECOMMENDATION: <variant>` — this is the
input to Sprint B's kernel change, with the evidence (put_ratio,
sm_overhead_ratio, sm_spread per cell) printed above it.

Write `ANALYSIS_MECHANISM.md` alongside the logs, including:
- The decision table rendered as text
- The recommendation + 3 variants-rejected-because lines
- A "no-regression" check comparing `workload-probe-r*.log` p99 against
  Sprint A's p99 on the same (ntok=128, num_sms=22) cell
- Cost and session duration (timestamps from the log files)

## Phase 9 — K-1b A/B regression (next GPU session goal)

Goal: confirm the K-1b kernel variant (commit 2c1c756b+) does not
regress Gate B correctness AND improves at least one prefill cell vs.
the Sprint A baseline kernel. Acceptance comes from
`ANALYSIS_MECHANISM.md`:

> K-1b must not regress any (ntok, num_sms) cell by more than 3% AND
> must improve at least one prefill cell (target: ntok=256 nsms=22,
> expected 5-10% improvement based on the 53 µs slot-boundary overhead
> finding).

### 9.1 Two builds side-by-side

Both builds come from the same commit; the flag is the only diff. Keep
each `install/` in a separate venv or install prefix so we can A/B
swap without a rebuild:

```bash
cd /workspace/uccl/ep
# Baseline — Sprint A kernel, no K-1b, no probe
INSTALL_DIR=/workspace/ep-install-sprintA \
  python3 setup.py install 2>&1 | tail -20

# K-1b variant
python3 setup.py clean
UCCL_EP_K1B=1 \
  INSTALL_DIR=/workspace/ep-install-k1b \
  python3 setup.py install 2>&1 | tail -20
```

Sanity check both install prefixes contain an `ep*.so` with matching
hashes minus the `.so` body:

```bash
ls -la /workspace/ep-install-sprintA/ep.*.so /workspace/ep-install-k1b/ep.*.so
```

### 9.2 Gate B on K-1b (MUST PASS)

Run all 6 Gate B tests under the K-1b `.so` — this is the blocking
gate for shipping K-1b.

```bash
PYTHONPATH=/workspace/ep-install-k1b:$PYTHONPATH \
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$R \
  --master_addr=$MASTER --master_port=12355 \
  tests/test_low_latency_overlap.py \
    --num-experts=288 --hidden=7168 --num-topk=8 \
    --num-sms-list=1,2,3,4,8,16,22,48,96 \
  2> gate-b-k1b-r${R}.log
grep -E "(PASS|FAIL|max diff|nonzero_frac)" gate-b-k1b-r${R}.log
```

Success: every (rank, num_sms) prints `max=0 nonzero_frac=0` AND all 6
test functions (analytical oracle, overlap_num_sms_full,
overlap_bit_exact, overlap_signal_wait, overlap_zero_token_expert,
overlap_bad_kwargs) complete with PASS. Any non-zero diff means
K-1b's phase-parity carry broke correctness — stop and triage.

### 9.3 Workload A/B grid (both .so's, same grid)

For each build, run the same workload grid so we can compare p99
directly:

```bash
for VARIANT in sprintA k1b; do
  PYTHONPATH=/workspace/ep-install-${VARIANT}:$PYTHONPATH \
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$R \
    --master_addr=$MASTER --master_port=12355 \
    bench_overlap.py --mode=workload \
      --workload-tokens=128,256,512 \
      --workload-sms=22,48,96 \
      --num-iters=20 \
      --hidden=7168 --num-topk=8 --num-experts=288 \
      --num-rdma-bytes=$((20 * 1024**3)) \
    2> workload-${VARIANT}-r${R}.log
done
```

Acceptance gate (computed after log retrieval):

| Cell               | Target                      |
|--------------------|-----------------------------|
| (128, 22) decode   | K-1b p99 ≤ Sprint A p99 ×1.03 |
| (256, 22) prefill  | K-1b p99 ≤ Sprint A p99 ×0.95 (improvement) |
| (512, 22) prefill  | K-1b p99 ≤ Sprint A p99 ×1.03 |
| all other cells    | K-1b p99 ≤ Sprint A p99 ×1.03 |

**If the (256, 22) improvement lands**, the probe's mechanism claim
is confirmed and Sprint B can ship K-1b.

### 9.4 Optional — probe on K-1b

Not required for acceptance but valuable as evidence that K-1b
actually shrunk `sm_overhead_ratio` (the mechanism quantity). Build a
third variant `UCCL_EP_K1B=1 UCCL_EP_PROBE=1 python3 setup.py install`
and rerun Phase 4's probe scan. Expected signal: `sm_ovhd` drops from
~1.43 (Sprint A) toward 1.10-1.20. If it doesn't drop, the slot-body
`__syncthreads()` between iterations may still be serializing — a
K-1b follow-up.

## Phase 10 — adaptive num_sms end-to-end validation (next session)

Goal: prove that `low_latency_combine(num_sms=0)` dispatcher change in
commit `7c925ea6` (Python-side adaptive tiers 22/22/48) preserves or
beats Sprint A's best *static* choice at every workload cell, with zero
> 3% regression anywhere in the grid.

Why: the adaptive thresholds were derived from Sprint A's offline log,
not from an in-situ A/B. We need one clean session where the exact
same call site is run with (a) Sprint A default `num_sms=static`
values and (b) adaptive `num_sms=0`, both compiled from the same
commit, to rule out variance and confirm the tier boundaries.

### 10.1 Two install prefixes (same .so, different call path)

Only one build is needed — adaptive lives in Python. Install once:

```bash
cd /workspace/uccl/ep
python3 setup.py install 2>&1 | tail -10
```

### 10.2 Grid A — static num_sms baseline (reproduces Sprint A numbers)

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$R \
  --master_addr=$MASTER --master_port=12355 \
  bench_overlap.py --mode=workload \
    --workload-tokens=128,256,384,512,768,1024 \
    --workload-sms=22,48,96 \
    --num-iters=20 \
    --hidden=7168 --num-topk=8 --num-experts=288 \
    --num-rdma-bytes=$((20 * 1024**3)) \
  2> workload-static-r${R}.log
```

This yields 6 ntok × 3 nsms = 18 cells × 16 ranks × 20 iter.
Note the new rows 384/768/1024 — they fill the tier boundaries so we
can see the U-curve inside each tier (not just at the old 128/256/512
sample points).

### 10.3 Grid B — adaptive num_sms (num_sms=0)

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$R \
  --master_addr=$MASTER --master_port=12355 \
  bench_overlap.py --mode=workload \
    --workload-tokens=128,256,384,512,768,1024 \
    --workload-sms=0 \
    --num-iters=20 \
    --hidden=7168 --num-topk=8 --num-experts=288 \
    --num-rdma-bytes=$((20 * 1024**3)) \
  2> workload-adaptive-r${R}.log
```

`--workload-sms=0` triggers `_pick_overlap_num_sms`; each of the 6
ntok cells uses the tier lookup. Bench prints the actual resolved
`num_sms` in the per-row `BENCH` line, so the log is self-documenting.

### 10.4 Acceptance gates (computed offline, code-server side)

For each ntok:

| Gate | Rule |
|---|---|
| G1 — no regress vs best static | `p99(adaptive, ntok) ≤ min_{nsms∈{22,48,96}} p99(static, ntok, nsms) × 1.03` |
| G2 — decode preserved | `p99(adaptive, 128) ≤ p99(static, 128, 96) × 0.75` (must keep the 29% Sprint A win) |
| G3 — prefill preserved | `p99(adaptive, 512) ≤ p99(static, 512, 22) × 1.00` (must still beat the 512×22 case it was supposed to fix) |
| G4 — interpolation sanity | `p99(adaptive, 384)` within ±5% of `min(p99(static, 384, 22), p99(static, 384, 48))` — confirms the (192,384] tier boundary is in a flat zone |

If G1 fails at any ntok, the tier boundary is wrong; re-derive from
this session's Grid A and land a follow-up PR before shipping
adaptive. Do NOT change tier values mid-session — collect the data,
teardown, decide offline.

### 10.5 Output artifacts

```
stage5-p5en/sprint-b-adaptive-<STAMP>/
  workload-static-r{0,1}.log         # raw bench logs
  workload-adaptive-r{0,1}.log
  ANALYSIS_ADAPTIVE.md                # G1–G4 results + decision
  adaptive-grid.csv                   # ntok,cfg,p50,p99,p99.9,delta_vs_best
```

## Phase 11 — probe v2 data collection (next session, parallel to Phase 10)

Goal: get `init_share / body_share / sync_share` per (ntok, num_sms)
cell on a clean Sprint A kernel, so we know which *mechanism* to
attack next. Probe v1 told us "slot-level overhead ~37%" but couldn't
separate the hoistable init from the sticky sync; v2 resolves that.

### 11.1 Build with probe v2 enabled

```bash
cd /workspace/uccl/ep
python3 setup.py clean
UCCL_EP_PROBE=1 python3 setup.py install 2>&1 | tail -10
python3 -c "from uccl import ep; \
  print(ep.probe_buffer_enabled(), ep.probe_buffer_bytes(), \
        ep.probe_buffer_max_sms(), ep.probe_buffer_max_slots_per_sm())"
```

Expected: `True 526912 128 64` (v2 buffer is larger than v1's 264704
due to the four new timestamp arrays + schema tag + padding).

### 11.2 Gate B regression on probe v2 build (MUST PASS)

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$R \
  --master_addr=$MASTER --master_port=12355 \
  tests/test_low_latency_overlap.py \
    --num-experts=288 --hidden=7168 --num-topk=8 \
    --num-sms-list=1,2,3,4,8,16,22,48,96 \
  2> gate-b-probev2-r${R}.log
grep -E "(PASS|FAIL|max diff|nonzero_frac)" gate-b-probev2-r${R}.log
```

Success: every rank/num_sms prints `max=0 nonzero_frac=0`. The v2
macros are no-syncs-added (same 4 program points as v1), so the
probability of a correctness regression is near zero — but Gate B is
cheap and the signal is high.

### 11.3 Probe v2 workload scan

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$R \
  --master_addr=$MASTER --master_port=12355 \
  bench_overlap.py --mode=probe \
    --probe-tokens=128,256,512 \
    --probe-sms=22,48,96 \
    --probe-iters=5 \
    --hidden=7168 --num-topk=8 --num-experts=288 \
    --num-rdma-bytes=$((20 * 1024**3)) \
  2> probev2-r${R}.log

echo "=== schema sanity ==="
grep -E "^PROBE_SCHEMA" probev2-r${R}.log | head -2
# Expected: PROBE_SCHEMA version=2 (at least one row per rank)

grep -c "^PROBE " probev2-r${R}.log
# Expected: 216 rows per log (8 ranks × 3 ntok × 3 nsms × 3 mechs)
```

### 11.4 Acceptance gates

| Gate | Rule |
|---|---|
| P1 — schema | Every log contains `PROBE_SCHEMA version=2` |
| P2 — non-trivial body | For every cell, `body_share > 0.20` (else the probe placement is wrong, not real mechanism data) |
| P3 — init_share ordering | `init_share(128,22) > init_share(512,22)` — decode is init-heavy, prefill is body-heavy. If violated, the hoisted-init hypothesis from Sprint B probe v1 is wrong |
| P4 — sync_share visibility | At least one cell has `sync_share ≥ 0.10` — if sync is always < 5% then K-1b's failure was purely register pressure, not T_sync residual |

### 11.5 Offline analysis

```bash
python3 ~/workspace/uccl-ep-optimization/uccl/ep/bench/runners/analyze_probe.py \
  probev2-r0.log probev2-r1.log > analyze_probev2_out.txt
cat analyze_probev2_out.txt
```

Output should end with one of:
- `==> NEXT KERNEL CHANGE: K-1a (hoist mbarrier_init only, no persistent phase)` — if init_share is the top contributor AND K-1b's regression was driven by persistent-phase register cost
- `==> NEXT KERNEL CHANGE: K-T_sync (overlap finish-flag atomic with next slot)` — if sync_share dominates
- `==> NEXT KERNEL CHANGE: K-1c (larger slot pipelining)` — if body_share is flat but aggregate is still NIC-bound (put_ratio > 0.7)

### 11.6 Output artifacts

```
stage5-p5en/sprint-b-probev2-<STAMP>/
  probev2-r{0,1}.log
  gate-b-probev2-r{0,1}.log
  analyze_probev2_out.txt
  ANALYSIS_PROBE_V2.md       # init/body/sync decomposition + next-kernel decision
```

## Phase 10/11 combined session flow

Because adaptive is pure-Python and probe v2 is a separate build, the
cleanest single-session layout is:

1. Phase 1 — start cluster, verify leaf.
2. Phase 10.1 — one clean build (no probe).
3. Phase 10.2 + 10.3 — Grid A + Grid B under same .so (same
   `torchrun` pair, different `--workload-sms`).
4. `python3 setup.py clean` — tear down .so to avoid cache collision.
5. Phase 11.1 — probe v2 build.
6. Phase 11.2 — Gate B on probe v2 build.
7. Phase 11.3 — probe scan.
8. Phase 6 — scp all 8 logs out.
9. Phase 7 — teardown.

Step 4 is the one trap: `setup.py install` with a different
`UCCL_EP_PROBE` must not share the build cache, else probe macros
won't actually fire. The `clean` between builds is mandatory.

## Total session budget

| Phase | Duration | Cost (p5en spot ~$8/hr × 2 nodes) |
|---|---|---|
| 1 start + leaf retry | ~6-15 min | $1.60 - $4.00 |
| 2 build (single variant) | ~12 min | $3.20 |
| 3 Gate B | ~3 min | $0.80 |
| 4 probe | ~12 min | $3.20 |
| 5 workload | ~10 min | $2.67 |
| 6 scp | ~2 min | $0.53 |
| 7 teardown | ~3 min | ~$0 |
| 9 K-1b A/B (Phase 9 = 2 builds + Gate B + 2 workload runs) | ~35 min | ~$9.30 |
| 10 adaptive A/B (1 build + 2 workload grids, 6×3 + 6×1 cells) | ~22 min | ~$5.90 |
| 11 probe v2 (1 build + Gate B + probe scan) | ~20 min | ~$5.30 |
| **Total (probe only)** | **~50 min** | **~$16** |
| **Total (K-1b A/B session, Phase 1+9+6+7)** | **~60 min** | **~$19** |
| **Total (adaptive + probe v2, Phase 1+10+11+6+7)** | **~70 min** | **~$19** |

If any phase exits non-zero and can't be recovered in 3 min, skip to
Phase 7. Don't let the spot burn while you debug.
