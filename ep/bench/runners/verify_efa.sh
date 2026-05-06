#!/bin/bash
# verify_efa.sh — pod-side EFA / libfabric health check.
#
# Lives alongside verify_topology.sh. Runs inside a GPU pod before any
# benchmark. Exits 0 if libfabric has registered at least one EFA
# provider, 2 otherwise. Prints a one-line summary either way.
#
# Catches the 2026-05-05 failure mode where aws-ofi-nccl printed
# `fi_getinfo: No data available` across every rank and the whole
# session's p99 was dominated by a degraded IBGDA submission path
# (see efa-validation/results/stage5-p5en/ANALYSIS_40MS_POSTMORTEM.md).
#
# Usage:
#   bash verify_efa.sh                # exits 0 or 2
#   bash verify_efa.sh --strict       # also requires N providers == num_EFA_devs
#
# Designed to be cheap (<100ms) so it can run before every session.

set -u

strict=0
if [ "${1:-}" = "--strict" ]; then
  strict=1
fi

# 1. fi_info comes from aws-ofi-nccl / libfabric-devel. If absent we can
#    still proceed — nvshmem uses in-kernel IBGDA directly — but the
#    typical p5en pod image ships fi_info and its absence suggests a
#    broken image baseline.
if ! command -v fi_info >/dev/null 2>&1; then
  echo "VERIFY_EFA WARN fi_info not installed; cannot verify libfabric EFA provider"
  # Not fatal: skip instead of fail, so pods without fi_info still run.
  exit 0
fi

# 2. List EFA providers. `fi_info -p efa -l` returns non-zero when no
#    provider is registered (the 2026-05-05 pathology).
providers="$(fi_info -p efa -l 2>&1)"
rc=$?
if [ "$rc" -ne 0 ] || ! echo "$providers" | grep -q "efa"; then
  echo "VERIFY_EFA FAIL libfabric has no EFA provider registered"
  echo "  fi_info output:"
  echo "$providers" | sed 's/^/    /'
  echo "  This matches the 2026-05-05 degraded-pod signature."
  echo "  DO NOT trust any p99/latency numbers from this pod."
  exit 2
fi

n_providers="$(echo "$providers" | grep -c "^efa")"

# 3. Count actual EFA PCI devices. On p5en: 16 EFAs (1 per NIC).
n_efa_devs=0
if command -v lspci >/dev/null 2>&1; then
  n_efa_devs=$(lspci 2>/dev/null | grep -ci "Elastic Fabric Adapter" || true)
fi

echo "VERIFY_EFA PASS providers=${n_providers} efa_devs=${n_efa_devs}"

if [ "$strict" -eq 1 ] && [ "$n_efa_devs" -gt 0 ] && [ "$n_providers" -lt "$n_efa_devs" ]; then
  echo "VERIFY_EFA FAIL strict mode: ${n_providers} providers < ${n_efa_devs} devices"
  exit 2
fi

exit 0
