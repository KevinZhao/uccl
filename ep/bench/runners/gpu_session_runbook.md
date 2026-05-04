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
