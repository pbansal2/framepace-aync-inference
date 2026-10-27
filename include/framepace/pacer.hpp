// FramePace - the render-thread side of the pipeline.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "framepace/clock.hpp"
#include "framepace/config.hpp"
#include "framepace/downsample.hpp"
#include "framepace/frame.hpp"
#include "framepace/spsc_queue.hpp"

namespace framepace {

using RequestQueue = SpscQueue<InferenceRequest, kRequestQueueDepth>;
using ResultQueue = SpscQueue<InferenceResult, kResultQueueDepth>;

/// A per-frame feature estimate handed to the renderer.
struct Estimate {
  std::array<float, kMaxFeatures> features{};
  std::uint32_t feature_count{0};
  /// Age of the newest real inference this was derived from.
  Duration source_age{};
  /// How far past that inference we extrapolated. Zero when interpolating or
  /// when the estimate is exact.
  Duration extrapolated_by{};
  /// False until the first inference lands -- the renderer must have a
  /// defined-but-unusable state for the first ~20ms after startup.
  bool valid{false};
  /// True when `extrapolated_by` hit the configured ceiling, meaning the
  /// estimate is being held rather than projected further.
  bool clamped{false};
};

/// Tracks the two newest results and projects them forward in time.
///
/// This is the component that actually hides inference latency. A result that
/// lands at T describes the world at `capture_ns`, which is 15-30ms (2-4 frames)
/// in the past. Presenting it directly makes motion lag the input visibly. So we
/// hold the two most recent results, estimate a per-feature velocity from them,
/// and project to the frame's presentation time.
///
/// Extrapolation is clamped to a horizon because velocity from two samples is
/// noisy; projecting 100ms on a stale pair produces wild values that look worse
/// than lag. Past the horizon we hold the last projected value instead.
class ResultTracker {
 public:
  /// Ingest a completed inference. Out-of-order and duplicate results are
  /// rejected: the queue preserves order, but a caller draining several queues
  /// or replaying could hand us an older frame_id, and silently accepting it
  /// would invert the velocity estimate.
  bool ingest(const InferenceResult& result) noexcept;

  /// Project to `t`. Allocation-free, no branches on the heap, safe to call
  /// every frame.
  [[nodiscard]] Estimate estimate_at(Timestamp t,
                                     Duration max_extrapolation) const noexcept;

  [[nodiscard]] bool has_value() const noexcept { return have_current_; }
  [[nodiscard]] std::uint64_t newest_frame_id() const noexcept {
    return current_.frame_id;
  }
  /// Capture time of the newest result. Together with `Clock::now()` this gives
  /// true capture-to-visible pipeline latency, which is strictly less than
  /// `Estimate::source_age` (that one is measured to the *presentation*
  /// deadline and so includes a frame of look-ahead).
  [[nodiscard]] std::int64_t newest_capture_ns() const noexcept {
    return current_.capture_ns;
  }

  void reset() noexcept {
    have_current_ = false;
    have_previous_ = false;
  }

 private:
  InferenceResult current_{};
  InferenceResult previous_{};
  bool have_current_{false};
  bool have_previous_{false};
};

/// Render-thread pacing counters. Plain integers: only the render thread
/// touches them.
struct PacerCounters {
  std::uint64_t frames{0};
  std::uint64_t submitted{0};
  /// Frames where we wanted to submit but the request queue was full, i.e.
  /// inference is saturated. Expected and healthy at 120Hz in, 30ms/inference
  /// out -- this is backpressure working, not an error.
  std::uint64_t submit_skipped_full{0};
  /// Frames where submission was skipped by the cadence throttle.
  std::uint64_t submit_skipped_cadence{0};
  std::uint64_t results_ingested{0};
  /// Results discarded by `pop_latest` because a fresher one was behind them.
  std::uint64_t results_dropped_stale{0};
  std::uint64_t results_rejected_out_of_order{0};
  /// Frames presented before the first inference landed.
  std::uint64_t frames_without_estimate{0};
  std::uint64_t frames_extrapolation_clamped{0};
};

/// Drives one 120Hz render frame: ingest results, project an estimate, and
/// (at the configured cadence) downsample and submit a new inference request.
///
/// Owns its downsample scratch inline, so `step()` performs no allocation. The
/// scratch is ~1MB for 1080p, which makes `Pacer` far too large for a thread
/// stack -- construct it once at startup via `std::make_unique<Pacer>(cfg)`.
/// That single allocation is explicitly outside the hot path.
class alignas(kCacheLineSize) Pacer {
 public:
  struct Config {
    Duration frame_budget{kFrameBudget120Hz};
    /// Spin window at the tail of `sleep_until_next_frame`.
    Duration spin_margin{300us};
    /// Submit inference every Nth frame. 1 = every frame and let the queue
    /// apply backpressure; 2-3 reduces wasted downsampling when the model is
    /// known to be slower than the frame rate.
    unsigned submit_interval{1};
    /// Extrapolation ceiling; past this, hold the last projection.
    ///
    /// Sized from the steady-state age of the newest result, which is
    /// `end_to_end_latency + inference_period / 2` -- not from the inference
    /// latency alone. For a 15-30ms model that is roughly 25ms + 11ms, so a
    /// 33ms ceiling (the obvious "4 frames at 120Hz" choice) clamps on almost
    /// every frame and the latency hiding silently stops working. 50ms covers
    /// the normal distribution and leaves the clamp as the exception it is
    /// meant to be.
    ///
    /// Raising it further is not free: the velocity comes from two samples
    /// ~22ms apart, so a 50ms horizon is already a ~2.3x extrapolation, and
    /// beyond that the projection diverges faster than the lag it removes.
    /// Watch `PacerCounters::frames_extrapolation_clamped` when retuning.
    Duration max_extrapolation{50ms};
  };

  explicit Pacer(const Config& config) noexcept;

  Pacer(const Pacer&) = delete;
  Pacer& operator=(const Pacer&) = delete;

  /// Queues shared with the inference worker. The worker is the consumer of
  /// `requests()` and the producer of `results()`.
  [[nodiscard]] RequestQueue& requests() noexcept { return requests_; }
  [[nodiscard]] ResultQueue& results() noexcept { return results_; }

  // -------------------------------------------------------------------------
  // Hot path -- all noexcept and allocation-free
  // -------------------------------------------------------------------------

  /// Drain the result queue, keeping only the freshest result.
  void ingest_results() noexcept;

  /// Downsample `frame` and submit it for inference, subject to the cadence
  /// throttle and queue space. Returns true if a request was published.
  ///
  /// Builds the tensor directly in the ring slot via `write_slot()`, so the
  /// 32KB payload is never copied.
  [[nodiscard]] bool submit(const SourceFrame& frame, std::uint64_t frame_id,
                            Timestamp capture) noexcept;

  /// Project the newest inference to `present_at`.
  [[nodiscard]] Estimate estimate_at(Timestamp present_at) noexcept;

  /// One full frame: ingest, submit, estimate. Returns the estimate the
  /// renderer should draw with.
  [[nodiscard]] Estimate step(const SourceFrame& frame, std::uint64_t frame_id,
                              Timestamp capture, Timestamp present_at) noexcept;

  /// Hold the loop until `deadline`, then return the next deadline. Keeping the
  /// cadence on an absolute schedule rather than `now + budget` is what stops
  /// per-frame overshoot from accumulating into drift.
  Timestamp sleep_until_next_frame(Timestamp deadline) noexcept;

  [[nodiscard]] const PacerCounters& counters() const noexcept { return counters_; }
  [[nodiscard]] const Config& config() const noexcept { return config_; }
  [[nodiscard]] const ResultTracker& tracker() const noexcept { return tracker_; }

 private:
  Config config_;
  PacerCounters counters_{};
  ResultTracker tracker_{};

  // Queues first and cache-line aligned (by their own alignas) so neither
  // shares a line with the render-thread-private state below.
  RequestQueue requests_{};
  ResultQueue results_{};

  /// Ping-pong scratch for the halving chain. Inline, hence no hot-path
  /// allocation; ~1MB at 1080p.
  static constexpr std::size_t kScratchBytes =
      pow2_scratch_bytes(kSourceWidth, kSourceHeight);
  alignas(kCacheLineSize) std::array<std::uint8_t, kScratchBytes> scratch_{};
};

}  // namespace framepace
