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

## Phase 0 — bench hygiene pre-flight (BLOCKING, every session)

Lesson from the 2026-05-05 session (`ANALYSIS_40MS_POSTMORTEM.md`):
a degraded libfabric / IBGDA submission path on one pod produced
p99 ≈ 40 000 µs across every UCCL kernel, without any runtime error.
Two days of "K-1b rejected" conclusions were derived from that
contaminated data. Every GPU session now runs two independent gates
before a single metric is published.

### 0.1 Pod-side `verify_efa.sh` (pre-build)

`runners/verify_efa.sh` shells out to `fi_info -p efa -l` and exits
non-zero if libfabric has no EFA provider registered (the
`fi_getinfo: No data available` signature). Runtime ~100 ms.

`runners/run_phase_10_11.sh` calls this as **Phase 0** on both pods
before the build step; a non-zero rc aborts the session without
paying for the build.

```bash
# Manual invocation on a suspicious pod:
kubectl exec <pod> -- bash /path/to/verify_efa.sh
# PASS: VERIFY_EFA PASS providers=16 efa_devs=16
# FAIL: VERIFY_EFA FAIL libfabric has no EFA provider registered
```

### 0.2 Bench-side smoke gate (implicit pre-scan)

`bench_overlap.py` runs a one-cell smoke check at (ntok=128,
num_sms=22) before every `--mode=workload` or `--mode=probe`
invocation (unless `--skip-smoke` is passed). Two accept criteria:

| gate | threshold | normal observed | pathological |
|---|---|---|---|
| `median(p99) / median(min)` | ≤ 10× | ~2× | ~190× (2026-05-05) |
| `median(p50)` | ≤ 1000 µs | ~330 µs | ~18 000 µs (2026-05-05) |

Both have 5× headroom over normal jitter. Fail → `sys.exit(2)` on
every rank, logs labelled `SMOKE_FAIL` with the reason.

```bash
# Explicit smoke-only invocation (used by teardown / mid-session sanity):
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$R \
  --master_addr=$MASTER --master_port=12360 \
  bench_overlap.py --mode=smoke \
    --hidden=7168 --num-topk=8 --num-experts=288 \
    --num-rdma-bytes=$((20 * 1024**3))
```

`--skip-smoke` exists for kernel-debug sessions where the bench is
known-broken; any numbers from such a session **must not** be used
for PR / ship decisions.

## Phase 10/11 — probe v2 mechanism scan (replaces probe v1)

Goal: capture T_slot decomposition into (T_init, T_body, T_sync) per
(ntok, num_sms) cell so the next kernel variant can be picked from
measured shares rather than v1's lumped slot_ovhd estimate.

The 2026-05-06 apne1 session using this flow produced the per-cell
shares recorded in `efa-validation/results/stage5-p5en/
sprint-b-adaptive-probev2-20260506T023000Z/ANALYSIS.md` — body
dominated at 82–94 %, init sat at 3–11 %, sync at 26–55 %. Those
numbers obsolete the Sprint B probe v1 reading that motivated the
K-1b (init-hoist) direction; that kernel variant is therefore *not*
carried on this branch.

### 10.1 Build

```bash
cd /workspace/uccl/ep
python3 setup.py clean
TORCH_CUDA_ARCH_LIST=9.0 UCCL_EP_PROBE=1 python3 setup.py install 2>&1 | tail -5
python3 -c "from uccl import ep; \
  print(ep.probe_buffer_enabled(), ep.probe_buffer_bytes())"
# Expected: True 526912  (v2 buffer size)
```

The `TORCH_CUDA_ARCH_LIST=9.0` override is required on nvcr 25.10-py3
images where `nvidia-smi` is not wired up and torch's autodetect
otherwise emits `compute_75` PTX that ptxas rejects for the Hopper
intrinsics used in the SM-stripe kernel.

### 10.2 Static-grid workload scan

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$R \
  --master_addr=$MASTER --master_port=12360 \
  bench_overlap.py --mode=workload \
    --workload-tokens=128,256,384,512,768,1024 \
    --workload-sms=22,48,96 \
    --num-iters=20 \
    --hidden=7168 --num-topk=8 --num-experts=288 \
    --num-rdma-bytes=$((20 * 1024**3)) \
  2> workload-static-r${R}.log
```

Expected row count per rank: 6 ntok × (1 dispatch + 1 combine-base +
3 combine-overlap) × 20 iter = 600.

### 10.3 Probe v2 scan

```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$R \
  --master_addr=$MASTER --master_port=12362 \
  bench_overlap.py --mode=probe \
    --probe-tokens=128,256,512 \
    --probe-sms=22,48,96 \
    --probe-iters=5 \
    --hidden=7168 --num-topk=8 --num-experts=288 \
    --num-rdma-bytes=$((20 * 1024**3)) \
  2> probev2-r${R}.log

grep "^PROBE_SCHEMA" probev2-r${R}.log | head -1
# Expected: PROBE_SCHEMA ntok=128 nsms=22 version=2 (rank 0 only)
grep -c "^PROBE " probev2-r${R}.log
# Expected: 216 per rank = 8 × 3 × 3 × 3 mechs (v1) + 3 extra v2 mechs
# Observed 2026-05-06: 432 per log (two ranks contributing)
```

### 10.4 Acceptance gates (offline)

| Gate | Rule |
|---|---|
| P1 | At least one log has `PROBE_SCHEMA version=2` (rank-0 only field) |
| P2 | Every cell has `body_share > 0.20` — else probe placement is wrong |
| P3 | `init_share(128,22) > init_share(512,22)` — decode is init-heavier |
| P4 | `max sync_share >= 0.10` across cells — sync is attackable |

Gate evaluator: `runners/gate_probev2.py probev2-r{0,1}.log`.

### 10.5 Deciding the next kernel variant

Read `body/init/sync` shares:

- `init_share` ≥ 20 % and dominates → K-1a (pure mbarrier_init hoist,
  no persistent phase; K-1b's register-pressure trap avoided)
- `sync_share` ≥ 20 % and attackable → K-T_sync (overlap finish-flag
  IBGDA atomic with next slot body)
- `body_share` is everything (> 80 %) and `put_ratio > 0.7` → K-1c
  (larger slot pipelining)

2026-05-06 data pointed at K-T_sync (sync 26–55 %, init 3–11 %).

## Phase 12 — K-T_sync A/B session (acceptance for the kernel variant)

Goal: decide whether to keep the `UCCL_EP_K_T_SYNC` kernel variant on
the SM-stripe branch by running it head-to-head against the same
commit with the flag off. No decision gets made from code review
alone — probe v2 already told us the sync region is 26–55 % of slot
time, but the actual `__syncthreads()` contribution within that
region is only knowable from hardware.

### 12.1 Two builds on the same commit (one branch)

Both configs come from `feat/k-t-sync` tip. Build and install into
separate site-packages paths so A/B swaps do not require a rebuild:

```bash
cd /workspace/uccl/ep
# Baseline: same as feat/sm-stripe-overlap, no K-T_sync flag
python3 setup.py clean
TORCH_CUDA_ARCH_LIST=9.0 python3 setup.py install --prefix=/workspace/install-baseline 2>&1 | tail -5

# K-T_sync variant
python3 setup.py clean
TORCH_CUDA_ARCH_LIST=9.0 UCCL_EP_K_T_SYNC=1 python3 setup.py install --prefix=/workspace/install-ktsync 2>&1 | tail -5

# Probe build (UCCL_EP_K_T_SYNC=1 UCCL_EP_PROBE=1) for mechanism-level A/B
python3 setup.py clean
TORCH_CUDA_ARCH_LIST=9.0 UCCL_EP_K_T_SYNC=1 UCCL_EP_PROBE=1 python3 setup.py install --prefix=/workspace/install-ktsync-probe 2>&1 | tail -5
```

Flip between builds by prepending the install prefix to PYTHONPATH
(`PYTHONPATH=/workspace/install-ktsync/lib/python3.12/site-packages` etc.).

### 12.2 Gate B regression on the K-T_sync build (BLOCKING)

```bash
PYTHONPATH=/workspace/install-ktsync/lib/python3.12/site-packages \
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$R \
  --master_addr=$MASTER --master_port=12360 \
  tests/test_low_latency_overlap.py \
    --num-experts=288 --hidden=7168 --num-topk=8 \
    --num-sms-list=1,2,3,4,8,16,22,48,96 \
  2> gateB-ktsync-r${R}.log

grep -E "max diff|nonzero_frac|PASS|FAIL" gateB-ktsync-r${R}.log | head -20
```

Success criterion: every (rank, num_sms) prints `max=0
nonzero_frac=0`. Any nonzero is an I1 / I2 / I3 violation — stop the
session, revert `f57159cc`, investigate which invariant broke.

### 12.3 Workload A/B grid

Same grid as `workload-static` in Phase 10.2 (128/256/384/512/768/1024
× 22/48/96 × 20 iter) so the result is directly comparable to the
`feat/sm-stripe-overlap` smoke numbers (`sprint-b-smoke-20260506T093000Z`)
and the earlier session (`sprint-b-adaptive-probev2-20260506T023000Z`).

```bash
for VARIANT in baseline ktsync; do
  PYTHONPATH=/workspace/install-${VARIANT}/lib/python3.12/site-packages \
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$R \
    --master_addr=$MASTER --master_port=12360 \
    bench_overlap.py --mode=workload \
      --workload-tokens=128,256,384,512,768,1024 \
      --workload-sms=22,48,96 \
      --num-iters=20 \
      --hidden=7168 --num-topk=8 --num-experts=288 \
      --num-rdma-bytes=$((20 * 1024**3)) \
    2> workload-${VARIANT}-r${R}.log
done
```

### 12.4 Probe v2 A/B — which piece of sync_share actually shrank

```bash
PYTHONPATH=/workspace/install-ktsync-probe/lib/python3.12/site-packages \
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$R \
  --master_addr=$MASTER --master_port=12362 \
  bench_overlap.py --mode=probe \
    --probe-tokens=128,256,512 \
    --probe-sms=22,48,96 \
    --probe-iters=5 \
    --hidden=7168 --num-topk=8 --num-experts=288 \
    --num-rdma-bytes=$((20 * 1024**3)) \
  2> probev2-ktsync-r${R}.log
```

Compare cell-by-cell against `sprint-b-adaptive-probev2-20260506T023000Z/
probev2-r{0,1}.log` via `gate_probev2.py`. Expected signature of
K-T_sync working:

- `sync_share(128,22)`: drops from **51.7 %** toward 15–25 %
- `body_share(128,22)`: rises from 82.8 % toward 85-92 %
- `init_share(128,22)`: unchanged (~11 %, K-T_sync doesn't touch this)

### 12.5 Acceptance gates (all required for ship)

| gate | rule | notes |
|---|---|---|
| **K1 correctness** | Gate B `max=0 nonzero_frac=0` across all cells | BLOCKING; any nonzero ⇒ revert |
| **K2 no regression** | No workload cell's K-T_sync median p99 > 1.03 × baseline | avoid net-negative variants like K-1b was |
| **K3 meaningful gain** | At least one cell shows K-T_sync median p99 ≤ 0.90 × baseline | ≥10 % improvement somewhere — otherwise the removed barrier was not the bottleneck |
| **K4 mechanism match** | Probe v2 `sync_share` at (128,22) drops by ≥ 20 percentage points | confirms the `__syncthreads()` was the dominant sync cost, not sync_barrier or the atomic |

If K1 fails → revert the commit, open an issue with the failing cell
and the raw max-diff number.

If K2 or K3 fails but K1 passes → the code is correct but the speed
win is not where we expected. Do not ship; reassess whether
`sync_barrier<true>` or the remote IBGDA atomic is the real cost
(those are the next targets, not `__syncthreads()`).

If K4 is ambiguous (sync_share drops only 5–15 pp) → ship K-T_sync
only if K3 still fires; otherwise treat the commit as a bench
artefact and keep looking.

### 12.6 Output artefacts

```
stage5-p5en/sprint-c-ktsync-<STAMP>/
  gateB-ktsync-r{0,1}.log
  workload-baseline-r{0,1}.log        # headline A/B for K2 / K3
  workload-ktsync-r{0,1}.log
  probev2-ktsync-r{0,1}.log           # K4 mechanism check
  ANALYSIS_KTSYNC.md                  # gate verdict + next step
  ktsync-grid.csv                     # per-cell p99 delta
```

### 12.7 Session budget

One session ≈ **~35 min / ≈ \$9**:
- 2 min — scale NG + leaf verify
- 9 min — 3 builds (baseline / ktsync / ktsync+probe)
- 3 min — Gate B
- 8 min — workload A/B (2 × 6 × 3 cells)
- 6 min — probe v2 scan
- 5 min — scp + teardown

Same spot discipline as previous sessions: teardown fires immediately
after `scp`, verify `State=shutting-down` before walking away.

## Total session budget

| Phase | Duration | Cost (p5en spot ~$8/hr × 2 nodes) |
|---|---|---|
| 1 start + leaf retry | ~6-15 min | $1.60 - $4.00 |
| 2 build | ~12 min | $3.20 |
| 3 Gate B | ~3 min | $0.80 |
| 4 probe | ~12 min | $3.20 |
| 5 workload | ~10 min | $2.67 |
| 6 scp | ~2 min | $0.53 |
| 7 teardown | ~3 min | ~$0 |
| **Total** | **~50 min** | **~$16** |

If any phase exits non-zero and can't be recovered in 3 min, skip to
Phase 7. Don't let the spot burn while you debug.
