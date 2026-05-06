#!/usr/bin/env bash
# Fire-and-forget orchestration for Phase 10 (adaptive num_sms) + Phase 11
# (probe v2) after SPS HIT. Assumes GPU pods are already scheduled on two
# same-leaf p5en nodes (see reference_eks_gpu_pod_libcuda.md for the pod
# pattern) and that MASTER_ADDR is set to node 0's pod IP.
#
# Required env:
#   STAMP       run identifier (e.g. 20260506T020000Z)
#   NODE_R0     pod name on rank-0 node
#   NODE_R1     pod name on rank-1 node
#   MASTER      master pod IP (typically NODE_R0 pod IP)
#   OUT_DIR     host-side results dir on bastion
#
# This script drives both nodes via kubectl exec. Each phase writes its
# logs into $OUT_DIR/<phase>-<rank>.log for later scp.
#
# Usage (on bastion):
#   cd /home/ec2-user/workspace/uccl-ep-optimization/uccl/ep/bench/runners
#   STAMP=$(date -u +%Y%m%dT%H%M%SZ) NODE_R0=... NODE_R1=... MASTER=10.x.x.x \
#     OUT_DIR=/tmp/phase-10-11-${STAMP} bash run_phase_10_11.sh

set -euo pipefail

: "${STAMP:?}"; : "${NODE_R0:?}"; : "${NODE_R1:?}"; : "${MASTER:?}"; : "${OUT_DIR:?}"
mkdir -p "${OUT_DIR}"

HIDDEN="${HIDDEN:-7168}"
NUM_TOPK="${NUM_TOPK:-8}"
NUM_EXPERTS="${NUM_EXPERTS:-288}"
NUM_ITERS="${NUM_ITERS:-20}"
PROBE_ITERS="${PROBE_ITERS:-5}"
NUM_RDMA_BYTES="${NUM_RDMA_BYTES:-$((20*1024*1024*1024))}"

TORCHRUN="torchrun --nnodes=2 --nproc_per_node=8 --master_addr=${MASTER} --master_port=12355"
BENCH="python3 -u bench_overlap.py"

# Helper — run one command on each pod in parallel with node_rank 0/1 split.
run_both() {
  local label="$1"; shift
  local cmd_common="$*"
  kubectl exec "${NODE_R0}" -- bash -lc \
    "cd /workspace/uccl/ep/bench && \
     NCCL_SOCKET_IFNAME=enp71s0 GLOO_SOCKET_IFNAME=enp71s0 \
     ${TORCHRUN} --node_rank=0 ${cmd_common}" \
    > "${OUT_DIR}/${label}-r0.log" 2>&1 &
  local p0=$!
  kubectl exec "${NODE_R1}" -- bash -lc \
    "cd /workspace/uccl/ep/bench && \
     NCCL_SOCKET_IFNAME=enp71s0 GLOO_SOCKET_IFNAME=enp71s0 \
     ${TORCHRUN} --node_rank=1 ${cmd_common}" \
    > "${OUT_DIR}/${label}-r1.log" 2>&1 &
  local p1=$!
  wait $p0 && wait $p1
  echo "[${label}] both ranks done"
}

# ---------- Phase 0: per-pod EFA / libfabric health check ----------------
# Refuses to waste an entire GPU session on a pod whose libfabric
# lost its EFA provider (2026-05-05 postmortem). Runs in ~100 ms.
echo "=== $(date -u +%FT%TZ) Phase 0 verify_efa on both pods ==="
for POD in "${NODE_R0}" "${NODE_R1}"; do
  kubectl cp /tmp/verify_efa.sh "${POD}:/tmp/verify_efa.sh" 2>/dev/null || true
  kubectl exec "${POD}" -- bash /tmp/verify_efa.sh \
    > "${OUT_DIR}/verify-efa-${POD}.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[verify_efa] FAIL on ${POD} (rc=${rc}). Aborting session."
    cat "${OUT_DIR}/verify-efa-${POD}.log"
    exit "$rc"
  fi
  echo "[verify_efa] ${POD}: $(cat "${OUT_DIR}/verify-efa-${POD}.log")"
done

# ---------- Phase 10.1: build adaptive-only .so on both nodes ------------
echo "=== $(date -u +%FT%TZ) Phase 10.1 build (no probe) ==="
for POD in "${NODE_R0}" "${NODE_R1}"; do
  kubectl exec "${POD}" -- bash -lc \
    "cd /workspace/uccl/ep && python3 setup.py clean && python3 setup.py install 2>&1 | tail -5" \
    > "${OUT_DIR}/build-adaptive-${POD}.log" 2>&1 &
done
wait
echo "[build-adaptive] done"

# ---------- Phase 10.2: Grid A static -------------------------------------
echo "=== $(date -u +%FT%TZ) Phase 10.2 Grid A static ==="
run_both workload-static \
  "${BENCH} --mode=workload \
    --workload-tokens=128,256,384,512,768,1024 \
    --workload-sms=22,48,96 \
    --num-iters=${NUM_ITERS} \
    --hidden=${HIDDEN} --num-topk=${NUM_TOPK} --num-experts=${NUM_EXPERTS} \
    --num-rdma-bytes=${NUM_RDMA_BYTES}"

# ---------- Phase 10.3: Grid B adaptive -----------------------------------
echo "=== $(date -u +%FT%TZ) Phase 10.3 Grid B adaptive ==="
run_both workload-adaptive \
  "${BENCH} --mode=workload \
    --workload-tokens=128,256,384,512,768,1024 \
    --workload-sms=0 \
    --num-iters=${NUM_ITERS} \
    --hidden=${HIDDEN} --num-topk=${NUM_TOPK} --num-experts=${NUM_EXPERTS} \
    --num-rdma-bytes=${NUM_RDMA_BYTES}"

# ---------- Phase 11.1: rebuild with probe v2 -----------------------------
echo "=== $(date -u +%FT%TZ) Phase 11.1 build (probe v2) ==="
for POD in "${NODE_R0}" "${NODE_R1}"; do
  kubectl exec "${POD}" -- bash -lc \
    "cd /workspace/uccl/ep && python3 setup.py clean && UCCL_EP_PROBE=1 python3 setup.py install 2>&1 | tail -5 && \
     python3 -c 'from uccl import ep; print(\"probe_enabled\", ep.probe_buffer_enabled(), \"bytes\", ep.probe_buffer_bytes())'" \
    > "${OUT_DIR}/build-probev2-${POD}.log" 2>&1 &
done
wait
echo "[build-probev2] done"

# ---------- Phase 11.2: Gate B on probe build -----------------------------
echo "=== $(date -u +%FT%TZ) Phase 11.2 Gate B ==="
run_both gateB-probev2 \
  "python3 -u tests/test_low_latency_overlap.py \
    --num-experts=${NUM_EXPERTS} --hidden=${HIDDEN} --num-topk=${NUM_TOPK} \
    --num-sms-list=1,2,3,4,8,16,22,48,96"

# ---------- Phase 11.3: probe v2 scan -------------------------------------
echo "=== $(date -u +%FT%TZ) Phase 11.3 probe scan ==="
run_both probev2 \
  "${BENCH} --mode=probe \
    --probe-tokens=128,256,512 \
    --probe-sms=22,48,96 \
    --probe-iters=${PROBE_ITERS} \
    --hidden=${HIDDEN} --num-topk=${NUM_TOPK} --num-experts=${NUM_EXPERTS} \
    --num-rdma-bytes=${NUM_RDMA_BYTES}"

echo "=== $(date -u +%FT%TZ) All phases complete. Logs in ${OUT_DIR} ==="
ls -l "${OUT_DIR}"
