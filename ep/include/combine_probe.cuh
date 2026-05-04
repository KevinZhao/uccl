// combine_probe.cuh — Sprint B mechanism-attribution probe.
//
// Enabled at compile time via -DUCCL_EP_PROBE. When disabled, the probe
// macros collapse to no-ops and there is zero runtime cost or memory use.
//
// The probe captures per-(SM, slot) clock64() timestamps at four points in
// the combine SEND phase so we can separate:
//   D-1  NIC queue contention: put_end - put_start per slot
//   D-2  Serial chain length:  slot_end - slot_start per slot
//   D-4  SM load imbalance:    sm_end - sm_start per SM
//
// Layout is fixed-size so a single cudaMalloc + cudaMemset is enough
// per bench run; no per-call allocation.
//
// Semantics are best-effort. clock64() reads the SM clock, which on Hopper
// runs at the boost frequency (≈ 1.98 GHz). Absolute µs are derived in the
// host-side dump by dividing by cudaDeviceProp.clockRate (in kHz).

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
  // num_sms. With kMaxSMs=128 this int32 array is 512 B, which keeps the
  // overall struct size at 264 704 B = 2^12 · 64.5 — naturally 64B-aligned
  // without explicit padding.
  int32_t n_slots[kMaxSMs];
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
#define UCCL_EP_PROBE_SM_START(probe_ptr, sm_id)                              \
  do {                                                                        \
    if ((probe_ptr) != nullptr && threadIdx.x == 0) {                         \
      (probe_ptr)->sm_start[sm_id] = clock64();                               \
    }                                                                         \
  } while (0)

#define UCCL_EP_PROBE_SM_END(probe_ptr, sm_id, n_slots_done)                  \
  do {                                                                        \
    if ((probe_ptr) != nullptr && threadIdx.x == 0) {                         \
      (probe_ptr)->sm_end[sm_id] = clock64();                                 \
      (probe_ptr)->n_slots[sm_id] = (n_slots_done);                           \
    }                                                                         \
  } while (0)

#define UCCL_EP_PROBE_SLOT_START(probe_ptr, sm_id, slot_iter)                 \
  do {                                                                        \
    if ((probe_ptr) != nullptr && threadIdx.x == 0 &&                         \
        (slot_iter) < ::uccl::ep::probe::kMaxSlotsPerSM) {                    \
      (probe_ptr)->slot_start[sm_id][slot_iter] = clock64();                  \
    }                                                                         \
  } while (0)

#define UCCL_EP_PROBE_SLOT_END(probe_ptr, sm_id, slot_iter)                   \
  do {                                                                        \
    if ((probe_ptr) != nullptr && threadIdx.x == 0 &&                         \
        (slot_iter) < ::uccl::ep::probe::kMaxSlotsPerSM) {                    \
      (probe_ptr)->slot_end[sm_id][slot_iter] = clock64();                    \
    }                                                                         \
  } while (0)

// Put probes are written once per token-loop by one elected lane per warp.
// We record only the first and last put of each slot iteration — that's
// enough to bound the slot's NIC-write window without N_token extra writes.
#define UCCL_EP_PROBE_PUT_FIRST(probe_ptr, sm_id, slot_iter, lane0_cond)      \
  do {                                                                        \
    if ((probe_ptr) != nullptr && (lane0_cond) &&                             \
        (slot_iter) < ::uccl::ep::probe::kMaxSlotsPerSM &&                    \
        (probe_ptr)->put_start[sm_id][slot_iter] == 0) {                      \
      (probe_ptr)->put_start[sm_id][slot_iter] = clock64();                   \
    }                                                                         \
  } while (0)

#define UCCL_EP_PROBE_PUT_LAST(probe_ptr, sm_id, slot_iter, lane0_cond)       \
  do {                                                                        \
    if ((probe_ptr) != nullptr && (lane0_cond) &&                             \
        (slot_iter) < ::uccl::ep::probe::kMaxSlotsPerSM) {                    \
      (probe_ptr)->put_end[sm_id][slot_iter] = clock64();                     \
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

#endif  // UCCL_EP_PROBE
