// FramePace - monotonic clock, spin hints, and sub-millisecond pacing sleep.
#pragma once

#include <chrono>
#include <cstdint>
#include <thread>

#include "framepace/config.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

namespace framepace {

using Clock = std::chrono::steady_clock;
using Timestamp = Clock::time_point;
using Duration = Clock::duration;

static_assert(Clock::is_steady, "frame pacing requires a monotonic clock");

using namespace std::chrono_literals;

/// 120 FPS == 8.3333ms. Expressed exactly in nanoseconds to avoid the drift
/// that accumulates from rounding to whole microseconds every frame.
inline constexpr Duration kFrameBudget120Hz = std::chrono::nanoseconds{8'333'333};
inline constexpr Duration kFrameBudget60Hz = std::chrono::nanoseconds{16'666'667};

/// Nanoseconds since an arbitrary monotonic epoch. Used inside POD messages
/// that cross the SPSC queue, because `Timestamp` is not guaranteed to be a
/// trivially copyable layout across toolchains.
[[nodiscard]] FRAMEPACE_ALWAYS_INLINE std::int64_t to_ns(Timestamp t) noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch())
      .count();
}

[[nodiscard]] FRAMEPACE_ALWAYS_INLINE Timestamp from_ns(std::int64_t ns) noexcept {
  return Timestamp{std::chrono::nanoseconds{ns}};
}

[[nodiscard]] FRAMEPACE_ALWAYS_INLINE double to_ms(Duration d) noexcept {
  return std::chrono::duration<double, std::milli>{d}.count();
}

/// Architectural "I am spinning" hint. Lowers power draw and, on SMT parts,
/// yields issue slots to the sibling thread. Not a memory barrier.
FRAMEPACE_ALWAYS_INLINE void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  _mm_pause();
#elif defined(__aarch64__) || defined(__arm64__)
  __asm__ __volatile__("yield" ::: "memory");
#elif defined(__arm__)
  __asm__ __volatile__("yield" ::: "memory");
#else
  // Nothing portable to do; the compiler barrier keeps the spin load live.
  __asm__ __volatile__("" ::: "memory");
#endif
}

/// Sleep until `deadline` with far tighter accuracy than `sleep_until` alone.
///
/// `nanosleep`/`sleep_until` on a general-purpose scheduler routinely overshoot
/// by 1-2ms, which is 12-24% of a 120Hz frame. So we sleep coarsely to
/// `deadline - spin_margin` and burn the remainder on a relaxed spin. The
/// margin trades CPU for accuracy; 300us is enough to absorb typical macOS and
/// Linux CFS wakeup latency.
///
/// Allocation-free and signal-safe.
inline void precise_sleep_until(Timestamp deadline,
                                Duration spin_margin = 300us) noexcept {
  const Timestamp coarse_deadline = deadline - spin_margin;
  if (Clock::now() < coarse_deadline) {
    std::this_thread::sleep_until(coarse_deadline);
  }
  while (Clock::now() < deadline) {
    cpu_relax();
  }
}

}  // namespace framepace
