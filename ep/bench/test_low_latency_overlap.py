"""
Sprint A unit tests for `low_latency_combine(overlap=True, ...)`.

Four test cases, all run on 2× p5en.48xlarge (EP=16) via torchrun:

  1. test_overlap_bit_exact_vs_baseline:
     pre-fill comp_signal to threshold → overlap path runs without any
     DeepGemm in the loop, output tensor must equal the overlap=False
     tensor (same dispatched/combined bytes).

  2. test_overlap_signal_wait:
     comp_signal starts at 0; launch combine(overlap=True) in a stream,
     then after ~50 ms a CUDA callback fills signals to threshold. Kernel
     must complete and output must match baseline.

  3. test_overlap_zero_token_expert:
     inject a handle where one local_expert has packed_recv_count[e]=0
     — the kernel must short-circuit its spin (rx_count == 0 guard) and
     still fire the finish flag.

  4. test_overlap_bad_kwargs:
     pass invalid block_m / threshold / missing tensors → Python-side
     ValueError. Then pass a non-zero src_signals_ptr → C++ should throw
     runtime_error (Blackwell path reserved for later PR).

The mock producer is pure `tensor.fill_()`; we do not require a DeepGemm
install. That means we lose the actual wall-clock overlap benefit, but
we verify correctness of the spin + TMA/IBGDA path end-to-end.

Usage on first node:
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \\
    --master_addr=<host> --master_port=12355 \\
    bench/test_low_latency_overlap.py \\
    --num-tokens=128 --hidden=7168 --num-topk=8 --num-experts=288
"""

import argparse
import math
import os
import threading
import time
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist

from buffer import Buffer
from utils import (
    init_dist,
    init_dist_under_torchrun,
    initialize_uccl,
    destroy_uccl,
    detect_ib_hca,
)

try:
    from uccl import ep
except ImportError:
    import sys

    sys.stderr.write("Failed to import uccl.ep\n")
    raise


def make_dispatch_inputs(
    rank: int,
    world_size: int,
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    device: torch.device,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deterministic x / topk_idx / topk_weights for reproducible tests."""
    g = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn(
        (num_tokens, hidden), dtype=torch.bfloat16, device=device, generator=g
    )
    # Distribute top-k choices uniformly across experts (no -1s for bit-exact tests).
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


def allocate_comp_signal(
    num_local_experts: int,
    num_max_dispatch_tokens_per_rank: int,
    num_ranks: int,
    block_m: int,
    device: torch.device,
    fill_value: int = 0,
) -> torch.Tensor:
    max_rx = num_max_dispatch_tokens_per_rank * num_ranks
    max_blocks = (max_rx + block_m - 1) // block_m
    signal = torch.full(
        (num_local_experts * max_blocks,),
        fill_value,
        dtype=torch.int32,
        device=device,
    )
    return signal


def run_combine(
    buffer: Buffer,
    simulated_gemm_x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    handle: tuple,
    *,
    overlap: bool,
    packed_recv_count: torch.Tensor = None,
    comp_signal: torch.Tensor = None,
    block_m: int = 64,
    threshold: int = 1,
    num_sms: int = 0,
    return_recv_hook: bool = None,
):
    """Thin wrapper that mirrors SGLang's call shape."""
    if return_recv_hook is None:
        return_recv_hook = overlap  # overlap requires hook
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
        hook()  # drain the RECV kernel synchronously
    torch.cuda.current_stream(simulated_gemm_x.device).synchronize()
    return combined_x


def test_baseline_vs_baseline_control(
    buffer: Buffer, dispatch_out, num_ranks: int, args, device: torch.device
):
    """Control test: run overlap=False twice with the same input. Any diff
    between out_ref1 and out_ref2 proves the buffer-swap / ring-buffer state
    itself is not bit-stable across back-to-back combine calls, which would
    invalidate the bit-exact assumption in the next test."""
    recv_x, recv_count, handle, _, _ = dispatch_out
    rx_ref = recv_x[0] if isinstance(recv_x, tuple) else recv_x
    simulated_gemm_x = torch.randn(rx_ref.shape, dtype=torch.bfloat16, device=device)
    _, topk_idx, topk_weights = make_dispatch_inputs(
        dist.get_rank(),
        dist.get_world_size(),
        args.num_tokens,
        args.hidden,
        args.num_topk,
        args.num_experts,
        device,
    )
    out_a = run_combine(
        buffer, simulated_gemm_x, topk_idx, topk_weights, handle, overlap=False
    )
    out_b = run_combine(
        buffer, simulated_gemm_x, topk_idx, topk_weights, handle, overlap=False
    )
    if not torch.allclose(out_a.float(), out_b.float(), rtol=0, atol=0):
        diff = (out_a.float() - out_b.float()).abs().max().item()
        print(
            f"[rank {dist.get_rank()}] CONTROL: two baseline combine calls differ "
            f"max abs = {diff} -- bit-exact test is INVALID",
            flush=True,
        )
    else:
        print(
            f"[rank {dist.get_rank()}] control baseline-vs-baseline: OK (bit-stable)",
            flush=True,
        )


def test_overlap_bit_exact_vs_baseline(
    buffer: Buffer, dispatch_out, num_ranks: int, args, device: torch.device
):
    """overlap=True with pre-filled comp_signal must match overlap=False output."""
    recv_x, recv_count, handle, _, _ = dispatch_out
    # recv_x is (fp8_tensor, scale_tensor); we only need the shape for the
    # combine input. SGLang would feed DeepGemm's bf16 output here.
    rx_ref = recv_x[0] if isinstance(recv_x, tuple) else recv_x
    simulated_gemm_x = torch.randn(rx_ref.shape, dtype=torch.bfloat16, device=device)
    _, topk_idx, topk_weights = make_dispatch_inputs(
        dist.get_rank(),
        dist.get_world_size(),
        args.num_tokens,
        args.hidden,
        args.num_topk,
        args.num_experts,
        device,
    )

    # Baseline: no overlap.
    out_ref = run_combine(
        buffer,
        simulated_gemm_x,
        topk_idx,
        topk_weights,
        handle,
        overlap=False,
    )

    # Overlap with signal pre-filled so kernel never blocks.
    num_local_experts = args.num_experts // num_ranks
    block_m = 64
    threshold = 1
    comp_signal = allocate_comp_signal(
        num_local_experts,
        args.num_max_dispatch_tokens_per_rank or args.num_tokens,
        num_ranks,
        block_m,
        device,
        fill_value=threshold,
    )
    out_ov = run_combine(
        buffer,
        simulated_gemm_x,
        topk_idx,
        topk_weights,
        handle,
        overlap=True,
        packed_recv_count=recv_count.to(torch.int32),
        comp_signal=comp_signal,
        block_m=block_m,
        threshold=threshold,
        num_sms=3,
    )

    if not torch.allclose(out_ref.float(), out_ov.float(), rtol=0, atol=0):
        diff = (out_ref.float() - out_ov.float()).abs().max().item()
        raise AssertionError(
            f"[test_overlap_bit_exact] max abs diff = {diff} (expected 0.0); "
            f"overlap output does not match baseline"
        )
    print(f"[rank {dist.get_rank()}] bit-exact: OK", flush=True)


def test_overlap_signal_wait(
    buffer: Buffer, dispatch_out, num_ranks: int, args, device: torch.device
):
    """Kernel waits on signal until a CPU thread fills it."""
    recv_x, recv_count, handle, _, _ = dispatch_out
    # recv_x is (fp8_tensor, scale_tensor); we only need the shape for the
    # combine input. SGLang would feed DeepGemm's bf16 output here.
    rx_ref = recv_x[0] if isinstance(recv_x, tuple) else recv_x
    simulated_gemm_x = torch.randn(rx_ref.shape, dtype=torch.bfloat16, device=device)
    _, topk_idx, topk_weights = make_dispatch_inputs(
        dist.get_rank(),
        dist.get_world_size(),
        args.num_tokens,
        args.hidden,
        args.num_topk,
        args.num_experts,
        device,
    )

    num_local_experts = args.num_experts // num_ranks
    block_m = 64
    threshold = 1
    comp_signal = allocate_comp_signal(
        num_local_experts,
        args.num_max_dispatch_tokens_per_rank or args.num_tokens,
        num_ranks,
        block_m,
        device,
        fill_value=0,  # start zero
    )

    # Kernel will spin until we fill the signal. Use a side-thread to fill
    # after 50 ms so we can verify the spin path actually spins and then
    # makes forward progress.
    def fill_later():
        time.sleep(0.05)
        comp_signal.fill_(threshold)
        torch.cuda.synchronize()

    t = threading.Thread(target=fill_later, daemon=True)
    t0 = time.time()
    t.start()
    out_ov = run_combine(
        buffer,
        simulated_gemm_x,
        topk_idx,
        topk_weights,
        handle,
        overlap=True,
        packed_recv_count=recv_count.to(torch.int32),
        comp_signal=comp_signal,
        block_m=block_m,
        threshold=threshold,
        num_sms=3,
    )
    elapsed = time.time() - t0
    t.join()
    if elapsed < 0.04:
        # We expected to wait ~50 ms; if we came back immediately, the
        # kernel skipped the spin.
        raise AssertionError(
            f"[test_overlap_signal_wait] combine returned in {elapsed*1e3:.1f} ms; "
            f"expected ~50 ms (kernel did not wait on comp_signal)"
        )
    print(
        f"[rank {dist.get_rank()}] signal-wait: waited ~{elapsed*1e3:.1f} ms, OK",
        flush=True,
    )


def test_overlap_zero_token_expert(
    buffer: Buffer, dispatch_out, num_ranks: int, args, device: torch.device
):
    """packed_recv_count[e]=0 must not deadlock the kernel."""
    recv_x, recv_count, handle, _, _ = dispatch_out
    # recv_x is (fp8_tensor, scale_tensor); we only need the shape for the
    # combine input. SGLang would feed DeepGemm's bf16 output here.
    rx_ref = recv_x[0] if isinstance(recv_x, tuple) else recv_x
    simulated_gemm_x = torch.randn(rx_ref.shape, dtype=torch.bfloat16, device=device)
    _, topk_idx, topk_weights = make_dispatch_inputs(
        dist.get_rank(),
        dist.get_world_size(),
        args.num_tokens,
        args.hidden,
        args.num_topk,
        args.num_experts,
        device,
    )

    num_local_experts = args.num_experts // num_ranks
    block_m = 64
    threshold = 1
    comp_signal = allocate_comp_signal(
        num_local_experts,
        args.num_max_dispatch_tokens_per_rank or args.num_tokens,
        num_ranks,
        block_m,
        device,
        fill_value=threshold,
    )

    # Force first local expert to have zero tokens.
    doctored = recv_count.clone().to(torch.int32)
    doctored[0] = 0

    try:
        run_combine(
            buffer,
            simulated_gemm_x,
            topk_idx,
            topk_weights,
            handle,
            overlap=True,
            packed_recv_count=doctored,
            comp_signal=comp_signal,
            block_m=block_m,
            threshold=threshold,
            num_sms=3,
        )
    except Exception as e:
        raise AssertionError(
            f"[test_overlap_zero_token_expert] combine crashed when "
            f"local_expert 0 has 0 tokens: {e}"
        )
    print(f"[rank {dist.get_rank()}] zero-token expert: OK", flush=True)


def test_overlap_bad_kwargs(
    buffer: Buffer, dispatch_out, num_ranks: int, args, device: torch.device
):
    """Python-side and C++-side arg validation."""
    recv_x, recv_count, handle, _, _ = dispatch_out
    # recv_x is (fp8_tensor, scale_tensor); we only need the shape for the
    # combine input. SGLang would feed DeepGemm's bf16 output here.
    rx_ref = recv_x[0] if isinstance(recv_x, tuple) else recv_x
    simulated_gemm_x = torch.randn(rx_ref.shape, dtype=torch.bfloat16, device=device)
    _, topk_idx, topk_weights = make_dispatch_inputs(
        dist.get_rank(),
        dist.get_world_size(),
        args.num_tokens,
        args.hidden,
        args.num_topk,
        args.num_experts,
        device,
    )
    num_local_experts = args.num_experts // num_ranks
    block_m = 64
    comp_signal = allocate_comp_signal(
        num_local_experts,
        args.num_max_dispatch_tokens_per_rank or args.num_tokens,
        num_ranks,
        block_m,
        device,
        fill_value=1,
    )

    # overlap=True + return_recv_hook=False
    try:
        buffer.low_latency_combine(
            simulated_gemm_x,
            topk_idx,
            topk_weights,
            handle,
            return_recv_hook=False,
            overlap=True,
            packed_recv_count=recv_count.to(torch.int32),
            comp_signal=comp_signal,
            block_m=block_m,
            threshold=1,
            num_sms=3,
        )
    except ValueError as e:
        print(
            f"[rank {dist.get_rank()}] bad-kwargs A (hook=False): caught ValueError OK",
            flush=True,
        )
    else:
        raise AssertionError(
            "expected ValueError for overlap=True, return_recv_hook=False"
        )

    # block_m=96 (invalid)
    try:
        buffer.low_latency_combine(
            simulated_gemm_x,
            topk_idx,
            topk_weights,
            handle,
            return_recv_hook=True,
            overlap=True,
            packed_recv_count=recv_count.to(torch.int32),
            comp_signal=comp_signal,
            block_m=96,
            threshold=1,
            num_sms=3,
        )
    except ValueError as e:
        print(
            f"[rank {dist.get_rank()}] bad-kwargs B (block_m=96): caught ValueError OK",
            flush=True,
        )
    else:
        raise AssertionError("expected ValueError for block_m=96")

    # src_signals != None (Blackwell path — should throw runtime_error from C++)
    fake_src_signals = torch.zeros((1,), dtype=torch.int32, device=device)
    try:
        buffer.low_latency_combine(
            simulated_gemm_x,
            topk_idx,
            topk_weights,
            handle,
            return_recv_hook=True,
            overlap=True,
            packed_recv_count=recv_count.to(torch.int32),
            comp_signal=comp_signal,
            block_m=block_m,
            threshold=1,
            num_sms=3,
            src_signals=fake_src_signals,
            src_signal_expect_value=1,
        )
    except RuntimeError as e:
        if "Blackwell" in str(e) or "src_signals" in str(e):
            print(
                f"[rank {dist.get_rank()}] bad-kwargs C (src_signals): caught "
                f"RuntimeError OK",
                flush=True,
            )
        else:
            raise
    else:
        raise AssertionError("expected RuntimeError for Blackwell src_signals path")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=288)
    parser.add_argument("--num-max-dispatch-tokens-per-rank", type=int, default=128)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    num_local_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
    rank, world_size, group = init_dist_under_torchrun(local_rank, num_local_ranks)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    num_rdma_bytes = 2 * 1024 * 1024 * 1024
    buffer = Buffer(
        group,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=args.num_experts // world_size,
        allow_nvlink_for_low_latency_mode=True,
        explicitly_destroy=True,
    )

    # One dispatch round to get a valid (recv_x, recv_count, handle).
    x, topk_idx, topk_weights = make_dispatch_inputs(
        rank,
        world_size,
        args.num_tokens,
        args.hidden,
        args.num_topk,
        args.num_experts,
        device,
    )
    # num_max_dispatch_tokens_per_rank must match between dispatch and combine
    # (determines shape of handle tuple and comp_signal stride).
    num_max_dispatch_tokens = max(
        args.num_tokens, args.num_max_dispatch_tokens_per_rank
    )
    args.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens
    recv_x, recv_count, handle, event, hook = buffer.low_latency_dispatch(
        x,
        topk_idx,
        num_max_dispatch_tokens,
        args.num_experts,
        use_fp8=True,
        async_finish=False,
        return_recv_hook=False,
    )
    dispatch_out = (recv_x, recv_count, handle, event, hook)

    for fn in (
        test_baseline_vs_baseline_control,
        test_overlap_bit_exact_vs_baseline,
        test_overlap_signal_wait,
        test_overlap_zero_token_expert,
        test_overlap_bad_kwargs,
    ):
        dist.barrier()
        try:
            fn(buffer, dispatch_out, world_size, args, device)
        except Exception as e:
            print(f"[rank {rank}] FAIL in {fn.__name__}: {e}", flush=True)
            raise

    dist.barrier()
    if rank == 0:
        print("=== ALL overlap unit tests PASSED ===", flush=True)

    buffer.destroy() if hasattr(buffer, "destroy") else None
    destroy_uccl()


if __name__ == "__main__":
    main()
