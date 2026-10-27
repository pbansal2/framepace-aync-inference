// FramePace - fixed-capacity latency sample recorder.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "framepace/clock.hpp"
#include "framepace/config.hpp"

namespace framepace {

/// Records up to `Capacity` samples with an O(1) allocation-free `record()`.
///
/// Stores raw samples rather than HDR-histogram buckets: at a few hundred
/// thousand frames the memory is trivial and exact tail percentiles matter more
/// than compactness here -- p99.9 of a bucketed histogram is only as precise as
/// its bucket width, and frame pacing lives or dies in the tail.
///
/// Overflow is counted, not wrapped. A reservoir or ring would silently bias
/// the distribution (a ring keeps only the most recent samples, which is
/// exactly the window most likely to be atypical); an explicit `dropped()`
/// count makes a too-small capacity visible instead.
///
/// `Capacity` samples are inline, so large instances belong on the heap --
/// construct via `make_recorder` at startup, never in the hot path.
template<std::size_t Capacity>
class SampleRecorder {
 public:
  static constexpr std::size_t kCapacity = Capacity;

  /// Hot-path entry point: a bounds check and a store.
  FRAMEPACE_ALWAYS_INLINE void record(std::int64_t sample) noexcept {
    if (FRAMEPACE_LIKELY(count_ < Capacity)) {
      samples_[count_++] = sample;
    } else {
      ++dropped_;
    }
  }

  FRAMEPACE_ALWAYS_INLINE void record(Duration d) noexcept {
    record(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
  }

  [[nodiscard]] std::size_t count() const noexcept { return count_; }
  [[nodiscard]] std::uint64_t dropped() const noexcept { return dropped_; }
  [[nodiscard]] bool empty() const noexcept { return count_ == 0; }

  void clear() noexcept {
    count_ = 0;
    dropped_ = 0;
    sorted_ = false;
  }

  /// Sorts in place. Must be called before any percentile query; separate from
  /// the queries so N percentiles cost one sort, not N.
  void finalize() noexcept {
    if (!sorted_) {
      std::sort(samples_.begin(), samples_.begin() + count_);
      sorted_ = true;
    }
  }

  /// Nearest-rank percentile, `p` in [0, 100]. Requires `finalize()`.
  [[nodiscard]] std::int64_t percentile(double p) const noexcept {
    if (count_ == 0) {
      return 0;
    }
    const double clamped = std::clamp(p, 0.0, 100.0);
    auto index = static_cast<std::size_t>(
        (clamped / 100.0) * static_cast<double>(count_ - 1) + 0.5);
    index = std::min(index, count_ - 1);
    return samples_[index];
  }

  [[nodiscard]] std::int64_t min() const noexcept {
    return count_ == 0 ? 0 : samples_[0];
  }
  [[nodiscard]] std::int64_t max() const noexcept {
    return count_ == 0 ? 0 : samples_[count_ - 1];
  }

  [[nodiscard]] double mean() const noexcept {
    if (count_ == 0) {
      return 0.0;
    }
    // Sum in double. The samples are nanosecond counts; a few hundred thousand
    // of them overflow nothing in int64, but double keeps this valid if the
    // recorder is reused for other units.
    double total = 0.0;
    for (std::size_t i = 0; i < count_; ++i) {
      total += static_cast<double>(samples_[i]);
    }
    return total / static_cast<double>(count_);
  }

  [[nodiscard]] double stddev() const noexcept {
    if (count_ < 2) {
      return 0.0;
    }
    const double mu = mean();
    double acc = 0.0;
    for (std::size_t i = 0; i < count_; ++i) {
      const double d = static_cast<double>(samples_[i]) - mu;
      acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(count_ - 1));
  }

 private:
  std::size_t count_{0};
  std::uint64_t dropped_{0};
  bool sorted_{false};
  std::array<std::int64_t, Capacity> samples_{};
};

/// Heap-construct a recorder at startup. Explicit so the one allocation is
/// visible at the call site rather than hidden in a hot-path constructor.
template<std::size_t Capacity>
[[nodiscard]] std::unique_ptr<SampleRecorder<Capacity>> make_recorder() {
  return std::make_unique<SampleRecorder<Capacity>>();
}

}  // namespace framepace
