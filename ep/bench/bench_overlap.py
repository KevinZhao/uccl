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
        choices=["baseline", "overlap", "both", "sweep", "workload"],
        default="both",
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
        type=int,
        default=22,
        help="Fixed num_sms used in --mode=workload overlap comparison",
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    num_local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
    rank, world_size, group = init_dist_under_torchrun(local_rank, num_local_ranks)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Sized for num_max_dispatch_tokens_per_rank up to 512 with hidden=7168.
    # combine_recv_buffer = 288 experts * 512 tokens * 14336 B ≈ 2.1 GB;
    # total ~= 8.4 GB across send/recv × 2 buffers. Round up to 10 GB.
    num_rdma_bytes = 10 * 1024 * 1024 * 1024
    buffer = Buffer(
        group,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=args.num_experts // world_size,
        allow_nvlink_for_low_latency_mode=True,
        explicitly_destroy=True,
    )

    if args.mode == "workload":
        # For each num_tokens: bench dispatch baseline, combine baseline,
        # combine overlap-N (N = workload_sms). Inputs rebuilt per config.
        # num_max_dispatch_tokens_per_rank is fixed to max(token_list) so the
        # buffer layout is sized once; actual dispatched count is x.size(0).
        token_list = [int(t) for t in args.workload_tokens.split(",") if t.strip()]
        max_ntok = max(token_list)
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
                dist.barrier()
                # Combine overlap at fixed num_sms
                avg, p50, p99, p999, mn, mx = run_mode_one(
                    buffer,
                    x_w,
                    topk_idx_w,
                    topk_weights_w,
                    max_ntok,
                    args.num_experts,
                    overlap=True,
                    num_sms=args.workload_sms,
                )
                _report(
                    f"BENCH rank={rank} iter={iteration} "
                    f"mode=combine-overlap-{args.workload_sms} "
                    f"num_tokens={ntok} num_sms={args.workload_sms} "
                    f"avg={avg:.2f} p50={p50:.2f} p99={p99:.2f} "
                    f"p999={p999:.2f} min={mn:.2f} max={mx:.2f}"
                )
        dist.barrier()
        if rank == 0:
            _report("=== workload scan DONE ===")
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
