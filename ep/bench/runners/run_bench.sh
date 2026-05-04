#!/bin/bash
# Generic 2-node bench runner. One set of CLI flags, one log file per rank.
#
# Usage (from each GPU node):
#   run_bench.sh <rank> <master_host> <tag> -- <bench_overlap.py args...>
#
# - <tag> is used to name log files: /workspace/bench-<tag>-r<rank>.log
# - Everything after `--` is passed verbatim to bench_overlap.py
#
# Env:
#   BRANCH       branch to pull (default: feat/sprint-a-generalization-bench)
#   REPO_URL     fork URL (default: https://github.com/KevinZhao/uccl.git)
#   MASTER_PORT  torchrun master port (default: 12355)
set -eo pipefail

RANK="${1:?rank}"; shift
MASTER_HOST="${1:?master host}"; shift
TAG="${1:?tag}"; shift
if [ "$1" != "--" ]; then
  echo "Usage: run_bench.sh <rank> <master_host> <tag> -- <bench args...>"
  exit 2
fi
shift  # consume the --
BENCH_ARGS=("$@")

BRANCH="${BRANCH:-feat/sprint-a-generalization-bench}"
REPO_URL="${REPO_URL:-https://github.com/KevinZhao/uccl.git}"
MASTER_PORT="${MASTER_PORT:-12355}"
WORK=/workspace/build-generalization
BUILD_LOG=/workspace/build-${TAG}-r${RANK}.log
TEST_LOG=/workspace/bench-${TAG}-r${RANK}.log

mkdir -p "$WORK"
cd "$WORK"
if [ -d uccl/.git ]; then
  cd uccl
  git fetch origin "$BRANCH" 2>&1 | tee -a "$BUILD_LOG"
  git reset --hard "origin/$BRANCH" 2>&1 | tee -a "$BUILD_LOG"
else
  git clone --branch "$BRANCH" --depth 5 "$REPO_URL" uccl 2>&1 | tee -a "$BUILD_LOG"
  cd uccl
fi

cd ep
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH:-}"

# Only rebuild if the .so is missing or the branch advanced since last build
NEED_BUILD=1
if [ -f /workspace/.last-built-commit ] && [ "$(cat /workspace/.last-built-commit)" = "$(git rev-parse HEAD)" ]; then
  NEED_BUILD=0
fi
if [ "$NEED_BUILD" = "1" ]; then
  python3 setup.py install 2>&1 | tee -a "$BUILD_LOG"
  git rev-parse HEAD > /workspace/.last-built-commit
fi

cd bench

echo "=== $(date -u) bench tag=$TAG rank=$RANK args='${BENCH_ARGS[*]}' ===" \
    | tee -a "$TEST_LOG"
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export UCCL_IB_HCA="${UCCL_IB_HCA:-rdmap}"

torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$RANK \
    --master_addr="$MASTER_HOST" --master_port="$MASTER_PORT" \
    bench_overlap.py "${BENCH_ARGS[@]}" >> "$TEST_LOG" 2>&1
EC=$?
echo "=== $(date -u) torchrun tag=$TAG rank=$RANK EC=$EC ===" | tee -a "$TEST_LOG"
exit $EC
