#include "framepace/pacer.hpp"

#include <algorithm>

namespace framepace {

// ---------------------------------------------------------------------------
// ResultTracker
// ---------------------------------------------------------------------------

bool ResultTracker::ingest(const InferenceResult& result) noexcept {
  if (have_current_ && result.frame_id <= current_.frame_id) {
    // Stale or duplicate. Accepting it would make `current_` older than
    // `previous_` and flip the sign of every velocity we derive.
    return false;
  }
  if (have_current_) {
    previous_ = current_;
    have_previous_ = true;
  }
  current_ = result;
  have_current_ = true;
  return true;
}

Estimate ResultTracker::estimate_at(Timestamp t,
                                    Duration max_extrapolation) const noexcept {
  Estimate out{};
  if (!have_current_) {
    return out;  // valid == false
  }

  const std::int64_t now_ns = to_ns(t);
  const std::uint32_t count = std::min<std::uint32_t>(
      current_.feature_count, static_cast<std::uint32_t>(kMaxFeatures));

  out.valid = true;
  out.feature_count = count;
  out.source_age = std::chrono::nanoseconds{now_ns - current_.capture_ns};

  const std::int64_t dt_ns = now_ns - current_.capture_ns;
  const std::int64_t span_ns =
      have_previous_ ? current_.capture_ns - previous_.capture_ns : 0;

  // No usable velocity: one sample only, a non-positive sample span, or a
  // presentation time at/behind the capture. Hand back the raw features.
  if (!have_previous_ || span_ns <= 0 || dt_ns <= 0) {
    std::copy_n(current_.features.begin(), count, out.features.begin());
    return out;
  }

  const std::int64_t max_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(max_extrapolation).count();
  const std::int64_t clamped_ns = std::min(dt_ns, std::max<std::int64_t>(max_ns, 0));
  out.clamped = clamped_ns < dt_ns;
  out.extrapolated_by = std::chrono::nanoseconds{clamped_ns};

  // Linear projection from the two newest samples. `inv_span` is computed once
  // and reused across features so the per-feature cost is one FMA.
  const float inv_span = 1.0F / static_cast<float>(span_ns);
  const float horizon = static_cast<float>(clamped_ns);
  for (std::uint32_t i = 0; i < count; ++i) {
    const float velocity = (current_.features[i] - previous_.features[i]) * inv_span;
    out.features[i] = current_.features[i] + velocity * horizon;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Pacer
// ---------------------------------------------------------------------------

Pacer::Pacer(const Config& config) noexcept : config_(config) {
  if (config_.submit_interval == 0) {
    config_.submit_interval = 1;
  }
}

void Pacer::ingest_results() noexcept {
  InferenceResult latest{};
  bool had_value = false;
  const std::size_t dropped = results_.pop_latest(latest, had_value);
  counters_.results_dropped_stale += dropped;
  if (!had_value) {
    return;
  }
  if (tracker_.ingest(latest)) {
    ++counters_.results_ingested;
  } else {
    ++counters_.results_rejected_out_of_order;
  }
}

bool Pacer::submit(const SourceFrame& frame, std::uint64_t frame_id,
                   Timestamp capture) noexcept {
  if (config_.submit_interval > 1 && (frame_id % config_.submit_interval) != 0) {
    ++counters_.submit_skipped_cadence;
    return false;
  }

  // Borrow the ring slot first, then build the tensor straight into it. If the
  // queue is full we discover that before paying for the downsample.
  InferenceRequest* slot = requests_.write_slot();
  if (slot == nullptr) {
    ++counters_.submit_skipped_full;
    return false;
  }

  slot->frame_id = frame_id;
  slot->capture_ns = to_ns(capture);
  slot->width = static_cast<std::uint32_t>(kTensorWidth);
  slot->height = static_cast<std::uint32_t>(kTensorHeight);

  const bool ok = downsample_pow2_u8(frame.pixels(), kSourceWidth, kSourceHeight,
                                     SourceFrame::kStride, kDownsampleLog2,
                                     slot->tensor.data(), kTensorWidth, scratch_);
  if (!ok) {
    // Never commit a partially built slot; the producer still owns it, so
    // simply not committing makes this a clean no-op.
    return false;
  }

  requests_.commit();
  ++counters_.submitted;
  return true;
}

Estimate Pacer::estimate_at(Timestamp present_at) noexcept {
  const Estimate est = tracker_.estimate_at(present_at, config_.max_extrapolation);
  if (!est.valid) {
    ++counters_.frames_without_estimate;
  } else if (est.clamped) {
    ++counters_.frames_extrapolation_clamped;
  }
  return est;
}

Estimate Pacer::step(const SourceFrame& frame, std::uint64_t frame_id,
                     Timestamp capture, Timestamp present_at) noexcept {
  ++counters_.frames;
  ingest_results();
  (void)submit(frame, frame_id, capture);
  return estimate_at(present_at);
}

Timestamp Pacer::sleep_until_next_frame(Timestamp deadline) noexcept {
  precise_sleep_until(deadline, config_.spin_margin);

  Timestamp next = deadline + config_.frame_budget;
  // If we blew the budget badly enough to fall a whole frame behind, resync to
  // the phase-aligned future rather than trying to "catch up" by presenting a
  // burst of frames with no sleep -- that converts one late frame into a run of
  // them and makes the stutter far more visible.
  const Timestamp now = Clock::now();
  if (next < now) {
    const auto budget_ns = config_.frame_budget.count();
    if (budget_ns > 0) {
      const auto behind = (now - next).count();
      const auto whole_frames = behind / budget_ns + 1;
      next += Duration{whole_frames * budget_ns};
    } else {
      next = now;
    }
  }
  return next;
}

}  // namespace framepace
