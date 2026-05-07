// combine_probe.cuh — mechanism-attribution probe.
//
// Enabled at compile time via -DUCCL_EP_PROBE. When disabled, the probe
// macros collapse to no-ops and there is zero runtime cost or memory use.
//
// v1 (Sprint B) captured four points per slot so we could attribute time to:
//   D-1  NIC queue contention: put_end - put_start per slot  (T_put)
//   D-2  Serial chain length:  slot_end - slot_start per slot (T_slot)
//   D-4  SM load imbalance:    sm_end - sm_start per SM      (T_sm)
//
// v2 (Sprint C planning) adds two more pairs to decompose T_slot further.
// Sprint B probe's sm_ovhd = T_sm / (n_slots × T_slot) overestimated the
// hoistable fraction because it lumped the slot-end `__syncthreads()` +
// finish-flag IBGDA atomic into the "between slots" residual. K-1b attacked
// only the `mbarrier_init` burst and underperformed its prediction.
//
// v2 measurements (written at the same four call sites, zero extra syncs):
//   D-2a body:  slot_body_start → slot_body_end  — pure token pipeline
//   D-2b sync:  sync_start      → sync_end       — cost of the CTA barrier
//                                                  + finish-flag IBGDA at slot end
//
// Absolute µs derivation unchanged: divide by SM clock (≈ 1.98 GHz on H200).

#pragma once

#include <cstdint>

namespace uccl::ep::probe {

// Upper bounds. combine is launched with num_sms ≤ 96 on H200 (one launch
// per combine call), and each SM iterates over
// ceil(num_experts / num_sms) slots → worst case with num_experts=288 and
// num_sms=3 is 96 slots, but overlap-mode min num_sms is 16 which caps at
// 18 slots. 64 is a comfortable upper bound with 2× headroom.
static constexpr int kMaxSMs = 128;
static constexpr int kMaxSlotsPerSM = 64;

struct ProbeBuffer {
  // v1 fields (Sprint B) ---------------------------------------------------
  // Per-SM totals (D-4).
  uint64_t sm_start[kMaxSMs];
  uint64_t sm_end[kMaxSMs];
  // Per-(SM, slot) breakdown (D-1, D-2).
  uint64_t slot_start[kMaxSMs][kMaxSlotsPerSM];
  uint64_t slot_end[kMaxSMs][kMaxSlotsPerSM];
  uint64_t put_start[kMaxSMs][kMaxSlotsPerSM];
  uint64_t put_end[kMaxSMs][kMaxSlotsPerSM];
  // Number of slots each SM actually processed. The tail SMs may process
  // one fewer slot than the head SMs when num_experts is not divisible by
  // num_sms.
  int32_t n_slots[kMaxSMs];

  // v2 fields (Sprint C) ---------------------------------------------------
  // Probe v1 showed that `T_slot = slot_end - slot_start` aggregates:
  //   (a) mbarrier_init burst at slot start (hoisted by K-1b)
  //   (b) pure token-pipeline body
  //   (c) end-of-slot `__syncthreads()` + finish-flag IBGDA atomic
  // v2 adds two pairs to separate (b) from (c). The (a) region is the span
  // between slot_start and slot_body_start, so it's derivable without a
  // fifth pair. All v2 writes hit the same call sites as v1, no new syncs.
  //
  // NOTE on sync_start/sync_end (IMPORTANT for readers):
  // Sprint C 2026-05-06 K-T_sync A/B session empirically showed that the
  // sync_start→sync_end window measures writer-lane IBGDA-atomic RTT, not
  // blocking time on non-writer warps. The v3 schema retains these field
  // names for backward compatibility but the analyzer should label the
  // derived metric "T_finish_flag" rather than "T_sync", and it should
  // NOT be reported as an attackable fraction. See docs/PROBE_V3_DESIGN.md.
  uint64_t slot_body_start[kMaxSMs][kMaxSlotsPerSM];  // after per-slot init
  uint64_t slot_body_end[kMaxSMs][kMaxSlotsPerSM];    // before slot-end sync
  uint64_t sync_start[kMaxSMs][kMaxSlotsPerSM];       // same as slot_body_end
                                                      // (v3: finish_flag_start)
  uint64_t sync_end[kMaxSMs][kMaxSlotsPerSM];         // same as slot_end
                                                      // (v3: finish_flag_end)

  // v3 fields (Sprint C+ planning) -----------------------------------------
  // v2 timed SEND phase only and missed the `while (rdma_recv_flag == 0)`
  // spin in RECV phase — the actual cross-rank critical path. K-T_sync
  // landed with sync_share 26-55% → 0.8-2.3% per probe v2 but workload p99
  // moved ±3%, consistent with RECV wait dominating T_kernel.
  //
  // v3 adds three timestamps per RECV iteration. Indexing is per-(SM,
  // recv_slot_iter) where recv_slot_iter is a fresh 0-based counter for
  // the RECV phase (independent from SEND's slot_iter because RECV uses
  // a different responsible_expert_idx mapping).
  uint64_t recv_wait_start[kMaxSMs][kMaxSlotsPerSM];   // entry to RECV spin
  uint64_t recv_wait_end[kMaxSMs][kMaxSlotsPerSM];     // flag arrived, exit spin
  uint64_t recv_reduce_end[kMaxSMs][kMaxSlotsPerSM];   // after local reduction
  // Which peer rank each RECV slot was waiting on. Lets the analyzer
  // answer "is the tail always coming from the same slow peer?"
  int32_t recv_src_rank[kMaxSMs][kMaxSlotsPerSM];
  // Number of RECV iterations each SM actually processed (parallel to
  // n_slots for SEND phase).
  int32_t n_recv_slots[kMaxSMs];

  // Schema tag. Writer sets:
  //   1 — v1 (Sprint B probe)
  //   2 — v2 (Sprint C planning; adds slot_body_*/sync_*)
  //   3 — v3 (Sprint C+; adds recv_*)
  // Analyzers must gate on schema_version and degrade gracefully for
  // older buffers. Zero → no probe wrote anything.
  int32_t schema_version;
  int32_t _pad_for_64B_alignment[15];  // keep total 64B-aligned (756800B)
};

static_assert(sizeof(ProbeBuffer) % 64 == 0,
              "ProbeBuffer must stay 64B-aligned for device memset safety");

}  // namespace uccl::ep::probe

#ifdef UCCL_EP_PROBE

#define UCCL_EP_PROBE_ENABLED 1

// Device-side helpers. Each macro expects a lane_0 / warp_0 gate from the
// caller so we write exactly once per (sm, slot). We intentionally avoid
// nested atomics — the writes are race-free because each (sm_id, slot_iter)
// tuple is owned by exactly one warp.
// Note on SM_START placement: fires BEFORE `next_clean` buffer init (which
// only SM 0 performs). Thus D-4's T_SM for SM 0 includes the init work and
// will be slightly higher than other SMs. That's the real latency SM 0 pays,
// but downstream analysis should treat SM 0 as a known slight outlier.
#define UCCL_EP_PROBE_SM_START(probe_ptr, sm_id)                              \
  do {                                                                        \
    if ((probe_ptr) != nullptr && threadIdx.x == 0 &&                         \
        (sm_id) < ::uccl::ep::probe::kMaxSMs) {                               \
      (probe_ptr)->sm_start[sm_id] = clock64();                               \
    }                                                                         \
  } while (0)

#define UCCL_EP_PROBE_SM_END(probe_ptr, sm_id, n_slots_done)                  \
  do {                                                                        \
    if ((probe_ptr) != nullptr && threadIdx.x == 0 &&                         \
        (sm_id) < ::uccl::ep::probe::kMaxSMs) {                               \
      (probe_ptr)->sm_end[sm_id] = clock64();                                 \
      (probe_ptr)->n_slots[sm_id] = (n_slots_done);                           \
    }                                                                         \
  } while (0)

#define UCCL_EP_PROBE_SLOT_START(probe_ptr, sm_id, slot_iter)                 \
  do {                                                                        \
    if ((probe_ptr) != nullptr && threadIdx.x == 0 &&                         \
        (sm_id) < ::uccl::ep::probe::kMaxSMs &&                               \
        (slot_iter) < ::uccl::ep::probe::kMaxSlotsPerSM) {                    \
      (probe_ptr)->slot_start[sm_id][slot_iter] = clock64();                  \
    }                                                                         \
  } while (0)

#define UCCL_EP_PROBE_SLOT_END(probe_ptr, sm_id, slot_iter)                   \
  do {                                                                        \
    if ((probe_ptr) != nullptr && threadIdx.x == 0 &&                         \
        (sm_id) < ::uccl::ep::probe::kMaxSMs &&                               \
        (slot_iter) < ::uccl::ep::probe::kMaxSlotsPerSM) {                    \
      (probe_ptr)->slot_end[sm_id][slot_iter] = clock64();                    \
    }                                                                         \
  } while (0)

// Put probes are written once per token-loop by one elected lane per warp.
// We record only the first and last put of each slot iteration — that's
// enough to bound the slot's NIC-write window without N_token extra writes.
// NOTE: T_put captures only inter-node IBGDA puts (guarded by
// `dst_p2p_ptr == 0` at the call site). Intra-node IPC peers use NVLink
// and are skipped — in mixed topologies the T_put distribution is
// conditional on the inter-node subset.
#define UCCL_EP_PROBE_PUT_FIRST(probe_ptr, sm_id, slot_iter, lane0_cond)      \
  do {                                                                        \
    if ((probe_ptr) != nullptr && (lane0_cond) &&                             \
        (sm_id) < ::uccl::ep::probe::kMaxSMs &&                               \
        (slot_iter) < ::uccl::ep::probe::kMaxSlotsPerSM &&                    \
        (probe_ptr)->put_start[sm_id][slot_iter] == 0) {                      \
      (probe_ptr)->put_start[sm_id][slot_iter] = clock64();                   \
    }                                                                         \
  } while (0)

#define UCCL_EP_PROBE_PUT_LAST(probe_ptr, sm_id, slot_iter, lane0_cond)       \
  do {                                                                        \
    if ((probe_ptr) != nullptr && (lane0_cond) &&                             \
        (sm_id) < ::uccl::ep::probe::kMaxSMs &&                               \
        (slot_iter) < ::uccl::ep::probe::kMaxSlotsPerSM) {                    \
      (probe_ptr)->put_end[sm_id][slot_iter] = clock64();                     \
    }                                                                         \
  } while (0)

// v2 macros (T_slot decomposition). Call from the same threadIdx.x==0 /
// warp-scoped gate as v1 macros. Cost is one clock64() + one uint64 store
// each, equivalent to v1; the total 6 timestamps per slot (up from 4) is
// still 6 fused clock reads, which on Hopper is ≈30 cycles total.
//
// Placement guide for callers (internode_ll.cu SEND phase):
//   slot_start    — right at the top of the slot for-loop (BEFORE the
//                   comp_signal spin and BEFORE the per-slot mbarrier_init)
//   slot_body_start — AFTER the per-slot mbarrier_init burst, immediately
//                   BEFORE the token for-loop begins. Under K-1b+kOverlap
//                   there is no per-slot init; body_start == slot_start.
//   slot_body_end — AFTER the last token's IBGDA put, BEFORE the
//                   slot-end __syncthreads()
//   sync_start    — same program point as slot_body_end (write both)
//   sync_end      — AFTER the slot-end __syncthreads() and AFTER the
//                   atomic_clean_flag decrement, right BEFORE slot_end
//   slot_end      — same as sync_end (write both; v2 keeps both to let a
//                   v1-only analyzer still work)
//
// The timestamp pairs (sync_start, sync_end) and (slot_body_end, sync_start)
// are intentionally redundant with the slot_body/slot boundaries; they let
// a v1-only analyzer read slot_start/slot_end while a v2 analyzer reads
// slot_body_*/sync_* for the finer decomposition.
#define UCCL_EP_PROBE_SLOT_BODY_START(probe_ptr, sm_id, slot_iter)            \
  do {                                                                        \
    if ((probe_ptr) != nullptr && threadIdx.x == 0 &&                         \
        (sm_id) < ::uccl::ep::probe::kMaxSMs &&                               \
        (slot_iter) < ::uccl::ep::probe::kMaxSlotsPerSM) {                    \
      (probe_ptr)->slot_body_start[sm_id][slot_iter] = clock64();             \
    }                                                                         \
  } while (0)

#define UCCL_EP_PROBE_SLOT_BODY_END(probe_ptr, sm_id, slot_iter)              \
  do {                                                                        \
    if ((probe_ptr) != nullptr && threadIdx.x == 0 &&                         \
        (sm_id) < ::uccl::ep::probe::kMaxSMs &&                               \
        (slot_iter) < ::uccl::ep::probe::kMaxSlotsPerSM) {                    \
      uint64_t now = clock64();                                               \
      (probe_ptr)->slot_body_end[sm_id][slot_iter] = now;                     \
      (probe_ptr)->sync_start[sm_id][slot_iter] = now;                        \
    }                                                                         \
  } while (0)

#define UCCL_EP_PROBE_SYNC_END(probe_ptr, sm_id, slot_iter)                   \
  do {                                                                        \
    if ((probe_ptr) != nullptr && threadIdx.x == 0 &&                         \
        (sm_id) < ::uccl::ep::probe::kMaxSMs &&                               \
        (slot_iter) < ::uccl::ep::probe::kMaxSlotsPerSM) {                    \
      (probe_ptr)->sync_end[sm_id][slot_iter] = clock64();                    \
    }                                                                         \
  } while (0)

// Called once by the first SM to advertise schema_version to the reader.
// v3 supersedes v2 (schema_version=3 implies all v2 fields are also
// populated, plus the RECV fields). Analyzers reading schema_version=2
// see only SEND-phase fields.
#define UCCL_EP_PROBE_SCHEMA_V2(probe_ptr, sm_id)                             \
  do {                                                                        \
    if ((probe_ptr) != nullptr && (sm_id) == 0 && threadIdx.x == 0) {         \
      (probe_ptr)->schema_version = 2;                                        \
    }                                                                         \
  } while (0)

#define UCCL_EP_PROBE_SCHEMA_V3(probe_ptr, sm_id)                             \
  do {                                                                        \
    if ((probe_ptr) != nullptr && (sm_id) == 0 && threadIdx.x == 0) {         \
      (probe_ptr)->schema_version = 3;                                        \
    }                                                                         \
  } while (0)

// v3 macros — RECV phase timing. Called with the same lane-0 gate as the
// existing RECV spin block in internode_ll.cu combine kernel.
//
// Placement guide (internode_ll.cu RECV phase):
//   UCCL_EP_PROBE_RECV_WAIT_START — immediately before the
//     `while (ld_acquire_sys_global(rdma_recv_flag + ...) == 0)` spin;
//     also writes recv_src_rank (caller passes the computed src_rank).
//   UCCL_EP_PROBE_RECV_WAIT_END — after the spin exits (remote flag arrived).
//   UCCL_EP_PROBE_RECV_REDUCE_END — after the local receive-side reduction
//     finishes. The gap wait_end→reduce_end is pure local work.
#define UCCL_EP_PROBE_RECV_WAIT_START(probe_ptr, sm_id, recv_iter, src_rank)  \
  do {                                                                        \
    if ((probe_ptr) != nullptr &&                                             \
        (sm_id) < ::uccl::ep::probe::kMaxSMs &&                               \
        (recv_iter) < ::uccl::ep::probe::kMaxSlotsPerSM) {                    \
      (probe_ptr)->recv_wait_start[sm_id][recv_iter] = clock64();             \
      (probe_ptr)->recv_src_rank[sm_id][recv_iter] = (src_rank);              \
    }                                                                         \
  } while (0)

#define UCCL_EP_PROBE_RECV_WAIT_END(probe_ptr, sm_id, recv_iter)              \
  do {                                                                        \
    if ((probe_ptr) != nullptr &&                                             \
        (sm_id) < ::uccl::ep::probe::kMaxSMs &&                               \
        (recv_iter) < ::uccl::ep::probe::kMaxSlotsPerSM) {                    \
      (probe_ptr)->recv_wait_end[sm_id][recv_iter] = clock64();               \
    }                                                                         \
  } while (0)

#define UCCL_EP_PROBE_RECV_REDUCE_END(probe_ptr, sm_id, recv_iter)            \
  do {                                                                        \
    if ((probe_ptr) != nullptr &&                                             \
        (sm_id) < ::uccl::ep::probe::kMaxSMs &&                               \
        (recv_iter) < ::uccl::ep::probe::kMaxSlotsPerSM) {                    \
      (probe_ptr)->recv_reduce_end[sm_id][recv_iter] = clock64();             \
    }                                                                         \
  } while (0)

// Record how many RECV iterations this SM completed. Fires once at
// RECV phase exit on lane 0 of the owning warp.
#define UCCL_EP_PROBE_N_RECV_SLOTS(probe_ptr, sm_id, n_recv_done)             \
  do {                                                                        \
    if ((probe_ptr) != nullptr &&                                             \
        (sm_id) < ::uccl::ep::probe::kMaxSMs) {                               \
      (probe_ptr)->n_recv_slots[sm_id] = (n_recv_done);                       \
    }                                                                         \
  } while (0)

#else  // UCCL_EP_PROBE not defined

#define UCCL_EP_PROBE_ENABLED 0

#define UCCL_EP_PROBE_SM_START(probe_ptr, sm_id) ((void)0)
#define UCCL_EP_PROBE_SM_END(probe_ptr, sm_id, n_slots_done) ((void)0)
#define UCCL_EP_PROBE_SLOT_START(probe_ptr, sm_id, slot_iter) ((void)0)
#define UCCL_EP_PROBE_SLOT_END(probe_ptr, sm_id, slot_iter) ((void)0)
#define UCCL_EP_PROBE_PUT_FIRST(probe_ptr, sm_id, slot_iter, lane0_cond) \
  ((void)0)
#define UCCL_EP_PROBE_PUT_LAST(probe_ptr, sm_id, slot_iter, lane0_cond) \
  ((void)0)
#define UCCL_EP_PROBE_SLOT_BODY_START(probe_ptr, sm_id, slot_iter) ((void)0)
#define UCCL_EP_PROBE_SLOT_BODY_END(probe_ptr, sm_id, slot_iter) ((void)0)
#define UCCL_EP_PROBE_SYNC_END(probe_ptr, sm_id, slot_iter) ((void)0)
#define UCCL_EP_PROBE_SCHEMA_V2(probe_ptr, sm_id) ((void)0)
#define UCCL_EP_PROBE_SCHEMA_V3(probe_ptr, sm_id) ((void)0)
#define UCCL_EP_PROBE_RECV_WAIT_START(probe_ptr, sm_id, recv_iter, src_rank) \
  ((void)0)
#define UCCL_EP_PROBE_RECV_WAIT_END(probe_ptr, sm_id, recv_iter) ((void)0)
#define UCCL_EP_PROBE_RECV_REDUCE_END(probe_ptr, sm_id, recv_iter) ((void)0)
#define UCCL_EP_PROBE_N_RECV_SLOTS(probe_ptr, sm_id, n_recv_done) ((void)0)

#endif  // UCCL_EP_PROBE
