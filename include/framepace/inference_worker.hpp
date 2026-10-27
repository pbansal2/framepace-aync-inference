// FramePace - the inference-thread side of the pipeline.
#pragma once

#include <atomic>
#include <cstdint>

#include "framepace/clock.hpp"
#include "framepace/config.hpp"
#include "framepace/frame.hpp"
#include "framepace/pacer.hpp"

namespace framepace {

/// Deterministic allocation-free PRNG (xorshift64*).
///
/// Deliberately not `std::mt19937`: that is 2.5KB of state, and more
/// importantly the distribution adapters (`uniform_int_distribution`) have
/// implementation-defined behaviour, so a "15-30ms" latency profile would not
/// reproduce across standard libraries. Benchmarks need to be comparable.
class Xorshift64 {
 public:
  explicit constexpr Xorshift64(std::uint64_t seed) noexcept
      : state_(seed != 0 ? seed : 0x9E3779B97F4A7C15ULL) {}

  constexpr std::uint64_t next() noexcept {
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    return state_ * 0x2545F4914F6CDD1DULL;
  }

  /// Uniform in [lo, hi]. Uses the high bits, which have the best spectral
  /// properties in this generator.
  constexpr std::int64_t range(std::int64_t lo, std::int64_t hi) noexcept {
    if (hi <= lo) {
      return lo;
    }
    const std::uint64_t span = static_cast<std::uint64_t>(hi - lo) + 1;
    return lo + static_cast<std::int64_t>((next() >> 1) % span);
  }

 private:
  std::uint64_t state_;
};

/// Simulated on-device ML inference: 15-30ms per frame, asynchronous to the
/// render loop.
///
/// Does genuine data-dependent work on the tensor (block reductions into the
/// feature vector) before padding out to the target latency. The real work
/// matters for two reasons: it keeps the optimiser from eliding the tensor the
/// pacer just built, and it makes the feature values a continuous function of
/// the input, so the pacer's velocity extrapolation is exercised with
/// realistically-correlated data rather than noise.
class InferenceWorker {
 public:
  struct Config {
    /// Inference latency envelope, sampled uniformly per frame.
    Duration min_latency{15ms};
    Duration max_latency{30ms};
    std::uint64_t seed{0xC0FFEE123456789ULL};
    /// How long to sleep when the request queue is empty. Keeps an idle worker
    /// off the CPU without adding meaningful latency to the next request.
    Duration idle_backoff{200us};
    /// Burn the latency on a spin instead of sleeping. Truer to a real NPU/GPU
    /// submission that pins a core, but it costs a core for the whole run.
    bool busy_spin{false};
  };

  struct Counters {
    std::uint64_t processed{0};
    /// Results we computed but could not publish because the render thread had
    /// not drained the result queue. Means the renderer is starved or stalled.
    std::uint64_t results_dropped_full{0};
    /// Queued requests skipped because a fresher frame was already waiting.
    /// Expected to be large at 120Hz in / 22ms out: it is the measure of how
    /// much work the latest-wins policy correctly declined to do.
    std::uint64_t requests_skipped_stale{0};
    std::uint64_t idle_polls{0};
  };

  InferenceWorker(RequestQueue& requests, ResultQueue& results,
                  const Config& config) noexcept;

  /// Process at most one request. Returns false if none was waiting.
  /// Allocation-free.
  [[nodiscard]] bool step() noexcept;

  /// Blocking loop until `stop` is set. Intended as a thread entry point.
  /// Allocation-free after entry.
  void run(const std::atomic<bool>& stop) noexcept;

  [[nodiscard]] const Counters& counters() const noexcept { return counters_; }

 private:
  RequestQueue& requests_;
  ResultQueue& results_;
  Config config_;
  Counters counters_{};
  Xorshift64 rng_;
};

}  // namespace framepace
