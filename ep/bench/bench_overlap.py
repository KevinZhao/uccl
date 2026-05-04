"""
Sprint A Gate C — performance bench comparing overlap=False vs overlap=True.

Measures combine-kernel latency only (SEND phase). The overlap kernel uses
3 SMs with a pre-filled comp_signal so the spin returns immediately; this
isolates the "overlap overhead" (drain + syncthreads + spin check) from
the DeepGemm co-execution benefit.

For the full-SBO e2e benefit (combine overlapping with DeepGemm down_gemm),
a separate SGLang e2e bench is needed; that is NOT this test.

Usage (torchrun, 2 nodes):
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$R \\
    --master_addr=$MASTER --master_port=12355 \\
    bench_overlap.py --num-iters=30 --mode=both
"""

import argparse
import os
import sys
import time
from functools import partial
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist

from buffer import Buffer
from utils import bench, init_dist_under_torchrun, destroy_uccl


def bench_detailed(fn, num_warmups=20, num_tests=50):
    """Like utils.bench but returns the full array of per-call latencies so
    we can compute p50/p99/p99.9 tails, not just min/avg/max."""
    torch.cuda.synchronize()
    current_device = torch.cuda.current_device()
    cache = torch.empty(
        int(256e6 // 4), dtype=torch.int, device=f"cuda:{current_device}"
    )
    for _ in range(num_warmups):
        fn()
    cache.zero_()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times = np.array(
        [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    )[1:]
    return times  # seconds


try:
    from uccl import ep
except ImportError:
    sys.stderr.write("Failed to import uccl.ep\n")
    raise


def _report(msg: str):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _alloc_probe_buffer(device):
    """Allocate a uint8 CUDA buffer sized exactly for ProbeBuffer.

    Returns (buffer_tensor, None) when the extension was built without
    -DUCCL_EP_PROBE; caller must skip probe capture in that case.
    """
    if not hasattr(ep, "probe_buffer_bytes"):
        return None, None
    n_bytes = ep.probe_buffer_bytes()
    buf = torch.zeros(n_bytes, dtype=torch.uint8, device=device)
    return buf, n_bytes


def _parse_probe_buffer(buf: torch.Tensor) -> dict:
    """Reinterpret the raw uint8 probe buffer into the ProbeBuffer layout.

    Layout must stay in sync with combine_probe.cuh: sm_start[kMaxSMs],
    sm_end[kMaxSMs], slot_start[kMaxSMs][kMaxSlotsPerSM], slot_end[...],
    put_start[...], put_end[...], n_slots[kMaxSMs], pad[3].
    All timestamp fields are uint64; n_slots is int32.
    """
    max_sms = ep.probe_buffer_max_sms()
    max_slots = ep.probe_buffer_max_slots_per_sm()
    host = buf.cpu()
    offset = 0

    def take_u64(count):
        nonlocal offset
        view = host[offset : offset + count * 8].view(torch.int64)
        offset += count * 8
        return view.numpy().astype("int64")

    def take_i32(count):
        nonlocal offset
        view = host[offset : offset + count * 4].view(torch.int32)
        offset += count * 4
        return view.numpy().astype("int32")

    sm_start = take_u64(max_sms)
    sm_end = take_u64(max_sms)
    slot_start = take_u64(max_sms * max_slots).reshape(max_sms, max_slots)
    slot_end = take_u64(max_sms * max_slots).reshape(max_sms, max_slots)
    put_start = take_u64(max_sms * max_slots).reshape(max_sms, max_slots)
    put_end = take_u64(max_sms * max_slots).reshape(max_sms, max_slots)
    n_slots = take_i32(max_sms)
    return dict(
        sm_start=sm_start,
        sm_end=sm_end,
        slot_start=slot_start,
        slot_end=slot_end,
        put_start=put_start,
        put_end=put_end,
        n_slots=n_slots,
    )


def _probe_summary(probes: list, sm_clock_khz: int) -> dict:
    """Aggregate per-iter probes into per-mechanism statistics (µs).

    Conventions:
      - Only clock64 deltas within the same SM are meaningful (different SMs
        may run at different frequencies, and the clock is not globally
        synchronized). We never subtract across SMs.
      - sm_clock_khz is the device boost clockRate from cudaDeviceProp
        (already in kHz), so cycles / sm_clock_khz / 1000 → µs.
      - n_slots[sm] may be 0 for SMs that the grid didn't spin up (rare)
        or when UCCL_EP_PROBE was compiled out — caller must check.

    Returns dict with keys T_slot, T_put, T_sm each mapping to a list of
    values (µs) across all (iter, sm, slot) samples; caller computes CDF.
    """
    T_slot = []  # D-2: slot_end - slot_start (whole slot wall-time)
    T_put = []  # D-1: put_end - put_start (IBGDA NIC window)
    T_sm = []  # D-4: sm_end - sm_start (per-SM total)
    for p in probes:
        n_slots = p["n_slots"]
        max_sms = len(n_slots)
        for sm in range(max_sms):
            n = int(n_slots[sm])
            if n <= 0:
                continue
            sm_delta = int(p["sm_end"][sm]) - int(p["sm_start"][sm])
            if sm_delta > 0:
                T_sm.append(sm_delta)
            for slot in range(min(n, p["slot_start"].shape[1])):
                slot_delta = int(p["slot_end"][sm, slot]) - int(
                    p["slot_start"][sm, slot]
                )
                if slot_delta > 0:
                    T_slot.append(slot_delta)
                ps = int(p["put_start"][sm, slot])
                pe = int(p["put_end"][sm, slot])
                if ps > 0 and pe > ps:
                    T_put.append(pe - ps)
    cycles_to_us = 1.0 / (sm_clock_khz * 1000.0) * 1e6

    def stats(arr):
        if not arr:
            return None
        a = np.asarray(arr, dtype=np.int64) * cycles_to_us
        return dict(
            n=len(a),
            mean=float(a.mean()),
            p50=float(np.percentile(a, 50)),
            p99=float(np.percentile(a, 99)),
            p999=float(np.percentile(a, 99.9)),
            max=float(a.max()),
            stdev=float(a.std()),
        )

    return dict(T_slot=stats(T_slot), T_put=stats(T_put), T_sm=stats(T_sm))


def build_inputs(rank, num_tokens, hidden, num_topk, num_experts, device):
    g = torch.Generator(device=device).manual_seed(42 + rank)
    x = torch.randn(
        (num_tokens, hidden), dtype=torch.bfloat16, device=device, generator=g
    )
    topk_idx = torch.randint(
        0,
        num_experts,
        (num_tokens, num_topk),
        device=device,
        generator=g,
        dtype=torch.int64,
    )
    topk_weights = (
        torch.ones((num_tokens, num_topk), dtype=torch.float, device=device) / num_topk
    )
    return x, topk_idx, topk_weights


def make_comp_signal(num_local_experts, max_rx_tokens, block_m, device, value):
    max_blocks = (max_rx_tokens + block_m - 1) // block_m
    return torch.full(
        (num_local_experts * max_blocks,),
        value,
        dtype=torch.int32,
        device=device,
    )


def run_mode_one(
    buffer,
    x,
    topk_idx,
    topk_weights,
    num_tokens,
    num_experts,
    overlap,
    num_sms,
    block_m=64,
    threshold=1,
):
    """Runs ONE dispatch+combine cycle. Returns (combine_avg_t_us)."""
    # Dispatch
    recv_x, recv_count, handle, _, _ = buffer.low_latency_dispatch(
        x,
        topk_idx,
        num_tokens,
        num_experts,
        use_fp8=True,
        async_finish=False,
        return_recv_hook=False,
    )
    rx_ref = recv_x[0] if isinstance(recv_x, tuple) else recv_x
    simulated_gemm_x = torch.ones(rx_ref.shape, dtype=torch.bfloat16, device=x.device)
    num_local_experts = num_experts // dist.get_world_size()
    comp_signal = None
    packed_recv_count = None
    return_recv_hook = overlap
    if overlap:
        comp_signal = make_comp_signal(
            num_local_experts,
            num_tokens * dist.get_world_size(),
            block_m,
            x.device,
            threshold,  # pre-filled, kernel never spins
        )
        packed_recv_count = recv_count.to(torch.int32)

    def combine_fn():
        combined_x, event, hook = buffer.low_latency_combine(
            simulated_gemm_x,
            topk_idx,
            topk_weights,
            handle,
            return_recv_hook=return_recv_hook,
            overlap=overlap,
            packed_recv_count=packed_recv_count,
            comp_signal=comp_signal,
            block_m=block_m,
            threshold=threshold,
            num_sms=num_sms,
        )
        if return_recv_hook:
            hook()

    times_s = bench_detailed(combine_fn, num_warmups=20, num_tests=50)
    times_us = times_s * 1e6
    return (
        float(times_us.mean()),
        float(np.percentile(times_us, 50)),
        float(np.percentile(times_us, 99)),
        float(np.percentile(times_us, 99.9)),
        float(times_us.min()),
        float(times_us.max()),
    )


def run_probe_one(
    buffer,
    x,
    topk_idx,
    topk_weights,
    num_tokens,
    num_experts,
    num_sms,
    num_iters,
    block_m=64,
    threshold=1,
):
    """Run combine(overlap=True) num_iters times with a probe buffer attached.

    Returns a list of per-iter parsed ProbeBuffer dicts. Caller is expected
    to aggregate across iters (and across ranks via torch.distributed). The
    combine wall-time per iter is NOT measured here — use --mode=workload
    for latency numbers; probe mode is strictly for mechanism attribution.
    """
    if not hasattr(ep, "probe_buffer_bytes"):
        raise RuntimeError(
            "probe buffer API not available — rebuild uccl.ep with "
            "UCCL_EP_PROBE=1 python3 setup.py install"
        )
    if not ep.probe_buffer_enabled():
        raise RuntimeError(
            "probe compiled out (UCCL_EP_PROBE not defined at build time)"
        )

    recv_x, recv_count, handle, _, _ = buffer.low_latency_dispatch(
        x,
        topk_idx,
        num_tokens,
        num_experts,
        use_fp8=True,
        async_finish=False,
        return_recv_hook=False,
    )
    rx_ref = recv_x[0] if isinstance(recv_x, tuple) else recv_x
    simulated_gemm_x = torch.ones(rx_ref.shape, dtype=torch.bfloat16, device=x.device)
    num_local_experts = num_experts // dist.get_world_size()
    comp_signal = make_comp_signal(
        num_local_experts,
        num_tokens * dist.get_world_size(),
        block_m,
        x.device,
        threshold,
    )
    packed_recv_count = recv_count.to(torch.int32)

    probe_buf, _ = _alloc_probe_buffer(x.device)
    probes = []
    # 3 warmup calls to settle NIC state, then num_iters captured.
    for warmup in range(3):
        combined_x, event, hook = buffer.low_latency_combine(
            simulated_gemm_x,
            topk_idx,
            topk_weights,
            handle,
            return_recv_hook=True,
            overlap=True,
            packed_recv_count=packed_recv_count,
            comp_signal=comp_signal,
            block_m=block_m,
            threshold=threshold,
            num_sms=num_sms,
        )
        hook()
    torch.cuda.synchronize()
    for it in range(num_iters):
        probe_buf.zero_()
        combined_x, event, hook = buffer.low_latency_combine(
            simulated_gemm_x,
            topk_idx,
            topk_weights,
            handle,
            return_recv_hook=True,
            overlap=True,
            packed_recv_count=packed_recv_count,
            comp_signal=comp_signal,
            block_m=block_m,
            threshold=threshold,
            num_sms=num_sms,
            probe_buffer=probe_buf,
        )
        hook()
        torch.cuda.synchronize()
        probes.append(_parse_probe_buffer(probe_buf))
    return probes


def run_dispatch_bench(buffer, x, topk_idx, num_tokens, num_experts):
    """Measure dispatch() latency distribution. No overlap variants exist
    yet for dispatch — this establishes the baseline for L3."""

    def dispatch_fn():
        recv_x, recv_count, handle, _, _ = buffer.low_latency_dispatch(
            x,
            topk_idx,
            num_tokens,
            num_experts,
            use_fp8=True,
            async_finish=False,
            return_recv_hook=False,
        )
        # recv_x / handle are reused; nothing to free explicitly

    times_s = bench_detailed(dispatch_fn, num_warmups=20, num_tests=50)
    times_us = times_s * 1e6
    return (
        float(times_us.mean()),
        float(np.percentile(times_us, 50)),
        float(np.percentile(times_us, 99)),
        float(np.percentile(times_us, 99.9)),
        float(times_us.min()),
        float(times_us.max()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=288)
    parser.add_argument(
        "--num-iters",
        type=int,
        default=30,
        help="How many full bench() calls per mode (each bench() already runs 50 tests)",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "overlap", "both", "sweep", "workload", "probe"],
        default="both",
    )
    parser.add_argument(
        "--probe-tokens",
        type=str,
        default="128,256,512",
        help="--mode=probe: comma-separated num_tokens values to capture probes for",
    )
    parser.add_argument(
        "--probe-sms",
        type=str,
        default="22,48,96",
        help="--mode=probe: comma-separated num_sms values to capture probes for",
    )
    parser.add_argument(
        "--probe-iters",
        type=int,
        default=5,
        help="--mode=probe: combine calls per (ntok, num_sms) config to aggregate",
    )
    parser.add_argument("--num-sms", type=int, default=3)
    parser.add_argument(
        "--sweep-sms",
        type=str,
        default="3,6,8,12,16,24,32",
        help="Comma-separated num_sms values to sweep in --mode=sweep",
    )
    parser.add_argument(
        "--workload-tokens",
        type=str,
        default="128,256,512",
        help="Comma-separated num_tokens values for --mode=workload",
    )
    parser.add_argument(
        "--workload-sms",
        type=str,
        default="22",
        help="Comma-separated num_sms values swept per ntok in --mode=workload",
    )
    parser.add_argument(
        "--num-rdma-bytes",
        type=int,
        default=20 * 1024 * 1024 * 1024,
        help="RDMA buffer size; must cover max(ntok)*world_size*hidden*2 across modes",
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    num_local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
    rank, world_size, group = init_dist_under_torchrun(local_rank, num_local_ranks)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    buffer = Buffer(
        group,
        num_rdma_bytes=args.num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=args.num_experts // world_size,
        allow_nvlink_for_low_latency_mode=True,
        explicitly_destroy=True,
    )

    if args.mode == "workload":
        # For each num_tokens × each num_sms: bench dispatch baseline, combine
        # baseline, combine overlap. Inputs rebuilt per ntok config. Buffer
        # layout is sized once at max(token_list).
        token_list = [int(t) for t in args.workload_tokens.split(",") if t.strip()]
        sms_list = [int(s) for s in args.workload_sms.split(",") if s.strip()]
        max_ntok = max(token_list)
        # Stamp the static config so log rows stay self-describing.
        if rank == 0:
            _report(
                f"CONFIG hidden={args.hidden} num_topk={args.num_topk} "
                f"num_experts={args.num_experts} world_size={world_size} "
                f"token_list={token_list} sms_list={sms_list} "
                f"num_rdma_bytes={args.num_rdma_bytes}"
            )
        for iteration in range(args.num_iters):
            for ntok in token_list:
                x_w, topk_idx_w, topk_weights_w = build_inputs(
                    rank, ntok, args.hidden, args.num_topk, args.num_experts, device
                )
                dist.barrier()
                # Dispatch latency (baseline only)
                avg, p50, p99, p999, mn, mx = run_dispatch_bench(
                    buffer, x_w, topk_idx_w, max_ntok, args.num_experts
                )
                _report(
                    f"BENCH rank={rank} iter={iteration} mode=dispatch-base "
                    f"num_tokens={ntok} num_sms=0 avg={avg:.2f} p50={p50:.2f} "
                    f"p99={p99:.2f} p999={p999:.2f} min={mn:.2f} max={mx:.2f}"
                )
                dist.barrier()
                # Combine baseline
                avg, p50, p99, p999, mn, mx = run_mode_one(
                    buffer,
                    x_w,
                    topk_idx_w,
                    topk_weights_w,
                    max_ntok,
                    args.num_experts,
                    overlap=False,
                    num_sms=0,
                )
                _report(
                    f"BENCH rank={rank} iter={iteration} mode=combine-base "
                    f"num_tokens={ntok} num_sms=0 avg={avg:.2f} p50={p50:.2f} "
                    f"p99={p99:.2f} p999={p999:.2f} min={mn:.2f} max={mx:.2f}"
                )
                # Combine overlap swept across all num_sms in sms_list
                for nsms in sms_list:
                    dist.barrier()
                    avg, p50, p99, p999, mn, mx = run_mode_one(
                        buffer,
                        x_w,
                        topk_idx_w,
                        topk_weights_w,
                        max_ntok,
                        args.num_experts,
                        overlap=True,
                        num_sms=nsms,
                    )
                    _report(
                        f"BENCH rank={rank} iter={iteration} "
                        f"mode=combine-overlap-{nsms} "
                        f"num_tokens={ntok} num_sms={nsms} "
                        f"avg={avg:.2f} p50={p50:.2f} p99={p99:.2f} "
                        f"p999={p999:.2f} min={mn:.2f} max={mx:.2f}"
                    )
        dist.barrier()
        if rank == 0:
            _report("=== workload scan DONE ===")
        buffer.destroy() if hasattr(buffer, "destroy") else None
        destroy_uccl()
        return

    if args.mode == "probe":
        # Mechanism-attribution probe. Each (ntok, num_sms) config captures
        # probe_iters combine calls. We print per-rank summary rows so host
        # side can aggregate across ranks later (clock64 is per-SM and is
        # not comparable across GPUs — cross-rank aggregation only makes
        # sense in histogram form, not averaged).
        sm_clock_khz = torch.cuda.get_device_properties(device).clock_rate
        ntok_list = [int(t) for t in args.probe_tokens.split(",") if t.strip()]
        sms_list = [int(s) for s in args.probe_sms.split(",") if s.strip()]
        max_ntok = max(ntok_list)
        if rank == 0:
            _report(
                f"PROBE_CONFIG sm_clock_khz={sm_clock_khz} "
                f"hidden={args.hidden} num_topk={args.num_topk} "
                f"num_experts={args.num_experts} world_size={world_size} "
                f"probe_iters={args.probe_iters} "
                f"ntok_list={ntok_list} sms_list={sms_list} "
                f"probe_enabled={ep.probe_buffer_enabled()}"
            )
        for ntok in ntok_list:
            x_w, topk_idx_w, topk_weights_w = build_inputs(
                rank, ntok, args.hidden, args.num_topk, args.num_experts, device
            )
            for nsms in sms_list:
                dist.barrier()
                try:
                    probes = run_probe_one(
                        buffer,
                        x_w,
                        topk_idx_w,
                        topk_weights_w,
                        max_ntok,
                        args.num_experts,
                        num_sms=nsms,
                        num_iters=args.probe_iters,
                    )
                except RuntimeError as e:
                    _report(f"PROBE_ERROR rank={rank} ntok={ntok} nsms={nsms}: {e}")
                    continue
                summary = _probe_summary(probes, sm_clock_khz)
                for mech, stats in summary.items():
                    if stats is None:
                        _report(
                            f"PROBE rank={rank} ntok={ntok} nsms={nsms} "
                            f"mech={mech} n=0 (no samples)"
                        )
                        continue
                    _report(
                        f"PROBE rank={rank} ntok={ntok} nsms={nsms} "
                        f"mech={mech} n={stats['n']} "
                        f"mean={stats['mean']:.2f} p50={stats['p50']:.2f} "
                        f"p99={stats['p99']:.2f} p999={stats['p999']:.2f} "
                        f"max={stats['max']:.2f} stdev={stats['stdev']:.2f}"
                    )
        dist.barrier()
        if rank == 0:
            _report("=== probe scan DONE ===")
        buffer.destroy() if hasattr(buffer, "destroy") else None
        destroy_uccl()
        return

    x, topk_idx, topk_weights = build_inputs(
        rank, args.num_tokens, args.hidden, args.num_topk, args.num_experts, device
    )

    if args.mode == "baseline":
        modes = [("baseline", False, 0)]
    elif args.mode == "overlap":
        modes = [("overlap", True, args.num_sms)]
    elif args.mode == "sweep":
        sweep = [int(x) for x in args.sweep_sms.split(",") if x.strip()]
        modes = [("baseline", False, 0)] + [(f"overlap-{n}", True, n) for n in sweep]
    else:
        modes = [("baseline", False, 0), ("overlap", True, args.num_sms)]

    for iteration in range(args.num_iters):
        for mode_name, overlap, num_sms_arg in modes:
            dist.barrier()
            avg_us, p50_us, p99_us, p999_us, min_us, max_us = run_mode_one(
                buffer,
                x,
                topk_idx,
                topk_weights,
                args.num_tokens,
                args.num_experts,
                overlap=overlap,
                num_sms=num_sms_arg,
            )
            _report(
                f"BENCH rank={rank} iter={iteration} mode={mode_name} "
                f"num_sms={num_sms_arg} avg={avg_us:.2f} p50={p50_us:.2f} "
                f"p99={p99_us:.2f} p999={p999_us:.2f} min={min_us:.2f} max={max_us:.2f}"
            )

    dist.barrier()
    if rank == 0:
        _report("=== Gate C bench DONE ===")
    buffer.destroy() if hasattr(buffer, "destroy") else None
    destroy_uccl()


if __name__ == "__main__":
    main()
