// FramePace - pacing, extrapolation, and end-to-end pipeline behaviour.

#include "framepace/pacer.hpp"

#include <atomic>
#include <cstdio>
#include <memory>
#include <thread>

#include "framepace/inference_worker.hpp"
#include "test_util.hpp"

using namespace framepace;

namespace {

constexpr std::int64_t kMs = 1'000'000;  // ns per ms

InferenceResult make_result(std::uint64_t frame_id, std::int64_t capture_ms,
                            float first_feature, float slope = 0.0F) {
  InferenceResult r{};
  r.frame_id = frame_id;
  r.capture_ns = capture_ms * kMs;
  r.complete_ns = (capture_ms + 20) * kMs;
  r.feature_count = static_cast<std::uint32_t>(kMaxFeatures);
  for (std::size_t i = 0; i < kMaxFeatures; ++i) {
    r.features[i] = first_feature + slope * static_cast<float>(i);
  }
  return r;
}

void fill_frame(SourceFrame& frame, std::uint8_t value) {
  frame.data.fill(value);
}

}  // namespace

// ---------------------------------------------------------------------------
// ResultTracker
// ---------------------------------------------------------------------------

FP_TEST(tracker_has_no_estimate_before_first_result) {
  ResultTracker tracker;
  FP_CHECK(!tracker.has_value());
  const Estimate est = tracker.estimate_at(from_ns(5 * kMs), 33ms);
  FP_CHECK(!est.valid);
  FP_CHECK_EQ(est.feature_count, std::uint32_t{0});
}

FP_TEST(tracker_returns_raw_features_from_a_single_result) {
  // One sample gives no velocity, so the only defensible answer is the sample
  // itself -- extrapolating from nothing would be invention.
  ResultTracker tracker;
  FP_CHECK(tracker.ingest(make_result(1, 0, 4.0F)));
  const Estimate est = tracker.estimate_at(from_ns(20 * kMs), 33ms);
  FP_REQUIRE(est.valid);
  FP_CHECK_NEAR(est.features[0], 4.0, 1e-6);
  FP_CHECK_EQ(est.extrapolated_by.count(), Duration::rep{0});
  FP_CHECK_NEAR(to_ms(est.source_age), 20.0, 1e-6);
}

FP_TEST(tracker_extrapolates_linearly_from_two_results) {
  // prev: capture 0ms,  feature 1.0
  // cur:  capture 10ms, feature 2.0   -> velocity 0.1 per ms
  // estimate at 25ms   -> dt 15ms     -> 2.0 + 0.1*15 = 3.5
  ResultTracker tracker;
  FP_CHECK(tracker.ingest(make_result(1, 0, 1.0F)));
  FP_CHECK(tracker.ingest(make_result(2, 10, 2.0F)));

  const Estimate est = tracker.estimate_at(from_ns(25 * kMs), 33ms);
  FP_REQUIRE(est.valid);
  FP_CHECK_NEAR(est.features[0], 3.5, 1e-4);
  FP_CHECK(!est.clamped);
  FP_CHECK_NEAR(to_ms(est.extrapolated_by), 15.0, 1e-6);
  FP_CHECK_NEAR(to_ms(est.source_age), 15.0, 1e-6);
}

FP_TEST(tracker_clamps_extrapolation_at_the_horizon) {
  // Same velocity, but 90ms past the last capture. Projecting the full 90ms
  // would give 11.0; the horizon holds it at 2.0 + 0.1*33 = 5.3.
  ResultTracker tracker;
  FP_CHECK(tracker.ingest(make_result(1, 0, 1.0F)));
  FP_CHECK(tracker.ingest(make_result(2, 10, 2.0F)));

  const Estimate est = tracker.estimate_at(from_ns(100 * kMs), 33ms);
  FP_REQUIRE(est.valid);
  FP_CHECK(est.clamped);
  FP_CHECK_NEAR(est.features[0], 5.3, 1e-3);
  FP_CHECK_NEAR(to_ms(est.extrapolated_by), 33.0, 1e-6);
  FP_CHECK_NEAR(to_ms(est.source_age), 90.0, 1e-6);
}

FP_TEST(tracker_does_not_extrapolate_backwards) {
  // Presentation time at or before the newest capture: hand back the raw
  // features rather than projecting a negative dt.
  ResultTracker tracker;
  FP_CHECK(tracker.ingest(make_result(1, 0, 1.0F)));
  FP_CHECK(tracker.ingest(make_result(2, 10, 2.0F)));

  const Estimate at_capture = tracker.estimate_at(from_ns(10 * kMs), 33ms);
  FP_REQUIRE(at_capture.valid);
  FP_CHECK_NEAR(at_capture.features[0], 2.0, 1e-6);

  const Estimate before = tracker.estimate_at(from_ns(5 * kMs), 33ms);
  FP_REQUIRE(before.valid);
  FP_CHECK_NEAR(before.features[0], 2.0, 1e-6);
}

FP_TEST(tracker_rejects_stale_and_duplicate_results) {
  // Accepting an older frame_id would make `current` older than `previous` and
  // invert every velocity derived from the pair.
  ResultTracker tracker;
  FP_CHECK(tracker.ingest(make_result(5, 50, 1.0F)));
  FP_CHECK(tracker.ingest(make_result(6, 60, 2.0F)));

  FP_CHECK(!tracker.ingest(make_result(6, 60, 9.0F)));  // duplicate id
  FP_CHECK(!tracker.ingest(make_result(3, 30, 9.0F)));  // older id
  FP_CHECK_EQ(tracker.newest_frame_id(), std::uint64_t{6});

  // The rejected results must not have perturbed the velocity.
  const Estimate est = tracker.estimate_at(from_ns(70 * kMs), 33ms);
  FP_CHECK_NEAR(est.features[0], 3.0, 1e-4);

  FP_CHECK(tracker.ingest(make_result(7, 70, 3.0F)));
  FP_CHECK_EQ(tracker.newest_frame_id(), std::uint64_t{7});
}

FP_TEST(tracker_handles_zero_span_between_results) {
  // Two captures with the same timestamp give an infinite velocity. Must
  // degrade to the raw features, not produce inf/NaN.
  ResultTracker tracker;
  FP_CHECK(tracker.ingest(make_result(1, 10, 1.0F)));
  FP_CHECK(tracker.ingest(make_result(2, 10, 5.0F)));
  const Estimate est = tracker.estimate_at(from_ns(20 * kMs), 33ms);
  FP_REQUIRE(est.valid);
  FP_CHECK_NEAR(est.features[0], 5.0, 1e-6);
  FP_CHECK(std::isfinite(est.features[0]));
}

FP_TEST(tracker_respects_feature_count) {
  ResultTracker tracker;
  InferenceResult r = make_result(1, 0, 1.0F);
  r.feature_count = 4;
  FP_CHECK(tracker.ingest(r));
  const Estimate est = tracker.estimate_at(from_ns(10 * kMs), 33ms);
  FP_CHECK_EQ(est.feature_count, std::uint32_t{4});
}

// ---------------------------------------------------------------------------
// Pacer plumbing
// ---------------------------------------------------------------------------

FP_TEST(pacer_submits_a_downsampled_tensor) {
  auto pacer = std::make_unique<Pacer>(Pacer::Config{});
  auto frame = std::make_unique<SourceFrame>();
  fill_frame(*frame, 200);

  FP_REQUIRE(pacer->submit(*frame, 0, Clock::now()));
  FP_CHECK_EQ(pacer->counters().submitted, std::uint64_t{1});
  FP_CHECK_EQ(pacer->requests().size(), std::size_t{1});

  const InferenceRequest* req = pacer->requests().read_slot();
  FP_REQUIRE(req != nullptr);
  FP_CHECK_EQ(req->frame_id, std::uint64_t{0});
  FP_CHECK_EQ(req->width, std::uint32_t{kTensorWidth});
  FP_CHECK_EQ(req->height, std::uint32_t{kTensorHeight});
  // A constant source must survive the box filter exactly.
  bool all_200 = true;
  for (std::uint8_t px : req->tensor) {
    all_200 = all_200 && (px == 200);
  }
  FP_CHECK_MSG(all_200, "uniform input did not survive downsampling");
  pacer->requests().release();
}

FP_TEST(pacer_applies_backpressure_when_inference_saturates) {
  // Fill the request queue, then confirm further submits are skipped rather
  // than blocking the render thread. This is the whole point of the design.
  auto pacer = std::make_unique<Pacer>(Pacer::Config{});
  auto frame = std::make_unique<SourceFrame>();
  fill_frame(*frame, 10);

  for (std::size_t i = 0; i < kRequestQueueDepth; ++i) {
    FP_CHECK(pacer->submit(*frame, i, Clock::now()));
  }
  FP_CHECK(pacer->requests().full());
  FP_CHECK(!pacer->submit(*frame, 99, Clock::now()));
  FP_CHECK(!pacer->submit(*frame, 100, Clock::now()));
  FP_CHECK_EQ(pacer->counters().submit_skipped_full, std::uint64_t{2});
  FP_CHECK_EQ(pacer->counters().submitted, kRequestQueueDepth);
}

FP_TEST(pacer_honours_submit_cadence) {
  Pacer::Config cfg;
  cfg.submit_interval = 3;
  auto pacer = std::make_unique<Pacer>(cfg);
  auto frame = std::make_unique<SourceFrame>();
  fill_frame(*frame, 1);

  // Frames 0 and 3 submit; 1, 2, 4 are throttled.
  FP_CHECK(pacer->submit(*frame, 0, Clock::now()));
  FP_CHECK(!pacer->submit(*frame, 1, Clock::now()));
  FP_CHECK(!pacer->submit(*frame, 2, Clock::now()));
  FP_CHECK(pacer->submit(*frame, 3, Clock::now()));
  FP_CHECK(!pacer->submit(*frame, 4, Clock::now()));
  FP_CHECK_EQ(pacer->counters().submitted, std::uint64_t{2});
  FP_CHECK_EQ(pacer->counters().submit_skipped_cadence, std::uint64_t{3});
}

FP_TEST(pacer_zero_submit_interval_is_normalised) {
  Pacer::Config cfg;
  cfg.submit_interval = 0;
  auto pacer = std::make_unique<Pacer>(cfg);
  FP_CHECK_EQ(pacer->config().submit_interval, 1U);
}

FP_TEST(pacer_ingest_keeps_only_the_freshest_result) {
  auto pacer = std::make_unique<Pacer>(Pacer::Config{});
  for (std::uint64_t i = 1; i <= 4; ++i) {
    FP_REQUIRE(pacer->results().try_push(
        make_result(i, static_cast<std::int64_t>(i) * 10, static_cast<float>(i))));
  }
  pacer->ingest_results();

  FP_CHECK_EQ(pacer->counters().results_ingested, std::uint64_t{1});
  FP_CHECK_EQ(pacer->counters().results_dropped_stale, std::uint64_t{3});
  FP_CHECK_EQ(pacer->tracker().newest_frame_id(), std::uint64_t{4});
  FP_CHECK(pacer->results().empty());
}

FP_TEST(pacer_counts_frames_without_an_estimate) {
  auto pacer = std::make_unique<Pacer>(Pacer::Config{});
  for (int i = 0; i < 3; ++i) {
    const Estimate est = pacer->estimate_at(Clock::now());
    FP_CHECK(!est.valid);
  }
  FP_CHECK_EQ(pacer->counters().frames_without_estimate, std::uint64_t{3});
}

FP_TEST(pacer_step_advances_all_counters) {
  auto pacer = std::make_unique<Pacer>(Pacer::Config{});
  auto frame = std::make_unique<SourceFrame>();
  fill_frame(*frame, 128);
  const Timestamp now = Clock::now();
  const Estimate est = pacer->step(*frame, 0, now, now + kFrameBudget120Hz);
  FP_CHECK_EQ(pacer->counters().frames, std::uint64_t{1});
  FP_CHECK_EQ(pacer->counters().submitted, std::uint64_t{1});
  FP_CHECK(!est.valid);  // nothing has come back yet
}

// ---------------------------------------------------------------------------
// Pacing cadence
// ---------------------------------------------------------------------------

FP_TEST(pacer_schedule_does_not_drift) {
  // Deadlines must advance on an absolute schedule. Computing `now + budget`
  // each frame would let every overshoot accumulate; after 60 frames at 120Hz
  // a 0.2ms per-frame bias would already be a 12ms phase error.
  Pacer::Config cfg;
  cfg.spin_margin = 200us;
  auto pacer = std::make_unique<Pacer>(cfg);

  constexpr int kFrames = 60;
  const Timestamp start = Clock::now();
  Timestamp deadline = start + cfg.frame_budget;
  for (int i = 0; i < kFrames; ++i) {
    deadline = pacer->sleep_until_next_frame(deadline);
  }
  const Timestamp end = Clock::now();

  const double expected_ms = to_ms(cfg.frame_budget) * kFrames;
  const double actual_ms = to_ms(end - start);
  std::printf("         %d frames: expected %.2fms, actual %.2fms (%.2f%% error)\n",
              kFrames, expected_ms, actual_ms,
              100.0 * (actual_ms - expected_ms) / expected_ms);

  // Generous bound: this runs on a non-realtime scheduler alongside whatever
  // else the machine is doing, so the assertion targets drift, not jitter.
  FP_CHECK_MSG(actual_ms >= expected_ms * 0.95,
               "finished early: " + std::to_string(actual_ms));
  FP_CHECK_MSG(actual_ms < expected_ms * 1.60,
               "drifted late: " + std::to_string(actual_ms));
}

FP_TEST(pacer_resyncs_after_a_missed_deadline) {
  // A deadline already far in the past must resync to a phase-aligned future
  // point, not return a still-past deadline (which would make the next frames
  // spin with no sleep at all).
  auto pacer = std::make_unique<Pacer>(Pacer::Config{});
  const Timestamp stale = Clock::now() - 500ms;
  const Timestamp next = pacer->sleep_until_next_frame(stale);
  FP_CHECK_MSG(next > Clock::now(), "resynced deadline is still in the past");
  FP_CHECK(to_ms(next - Clock::now()) <= to_ms(kFrameBudget120Hz) + 1.0);
}

// ---------------------------------------------------------------------------
// End to end
// ---------------------------------------------------------------------------

FP_TEST(pipeline_runs_end_to_end_with_a_real_worker) {
  // Full topology on two threads with a fast model, so several results land
  // inside a short test. Also a second TSan surface: the pacer and worker touch
  // both queues concurrently here, which test_spsc exercises only in the
  // abstract.
  auto pacer = std::make_unique<Pacer>(Pacer::Config{});
  auto frame = std::make_unique<SourceFrame>();

  InferenceWorker::Config wcfg;
  wcfg.min_latency = 2ms;
  wcfg.max_latency = 5ms;
  wcfg.idle_backoff = 100us;
  InferenceWorker worker(pacer->requests(), pacer->results(), wcfg);

  std::atomic<bool> stop{false};
  std::thread worker_thread([&] { worker.run(stop); });

  constexpr int kFrames = 120;  // ~1s at 120Hz
  Xorshift64 rng(4242);
  Timestamp deadline = Clock::now() + kFrameBudget120Hz;
  int valid_estimates = 0;

  for (int i = 0; i < kFrames; ++i) {
    // Vary the frame so features move and extrapolation has real velocity.
    fill_frame(*frame, static_cast<std::uint8_t>(i * 2));
    const Timestamp capture = Clock::now();
    const Estimate est =
        pacer->step(*frame, static_cast<std::uint64_t>(i), capture, deadline);
    valid_estimates += est.valid ? 1 : 0;
    for (std::uint32_t f = 0; f < est.feature_count; ++f) {
      FP_CHECK(std::isfinite(est.features[f]));
    }
    deadline = pacer->sleep_until_next_frame(deadline);
    (void)rng;
  }

  stop.store(true, std::memory_order_release);
  worker_thread.join();

  const PacerCounters& c = pacer->counters();
  std::printf(
      "         frames=%llu submitted=%llu ingested=%llu "
      "stale_dropped=%llu full_skips=%llu processed=%llu\n",
      static_cast<unsigned long long>(c.frames),
      static_cast<unsigned long long>(c.submitted),
      static_cast<unsigned long long>(c.results_ingested),
      static_cast<unsigned long long>(c.results_dropped_stale),
      static_cast<unsigned long long>(c.submit_skipped_full),
      static_cast<unsigned long long>(worker.counters().processed));

  FP_CHECK_EQ(c.frames, static_cast<std::uint64_t>(kFrames));
  FP_CHECK_MSG(worker.counters().processed > 0, "worker processed nothing");
  FP_CHECK_MSG(c.results_ingested > 0, "no results reached the pacer");
  FP_CHECK_MSG(valid_estimates > kFrames / 2,
               "only " + std::to_string(valid_estimates) + " of " +
                   std::to_string(kFrames) + " frames had an estimate");
  // Every result the pacer accepted must have been strictly newer than the
  // last, so nothing should have been rejected as out of order.
  FP_CHECK_EQ(c.results_rejected_out_of_order, std::uint64_t{0});
}

FP_TEST(pipeline_survives_a_worker_slower_than_the_frame_rate) {
  // The engine's actual operating point: 8.33ms frames, 15-30ms inference. The
  // render loop must keep its cadence and keep producing finite estimates while
  // most submissions are refused.
  auto pacer = std::make_unique<Pacer>(Pacer::Config{});
  auto frame = std::make_unique<SourceFrame>();

  InferenceWorker::Config wcfg;  // default 15-30ms
  InferenceWorker worker(pacer->requests(), pacer->results(), wcfg);

  std::atomic<bool> stop{false};
  std::thread worker_thread([&] { worker.run(stop); });

  constexpr int kFrames = 150;
  Timestamp deadline = Clock::now() + kFrameBudget120Hz;
  const Timestamp start = Clock::now();
  for (int i = 0; i < kFrames; ++i) {
    fill_frame(*frame, static_cast<std::uint8_t>((i * 3) & 0xFF));
    const Estimate est =
        pacer->step(*frame, static_cast<std::uint64_t>(i), Clock::now(), deadline);
    for (std::uint32_t f = 0; f < est.feature_count; ++f) {
      FP_CHECK(std::isfinite(est.features[f]));
    }
    deadline = pacer->sleep_until_next_frame(deadline);
  }
  const double elapsed_ms = to_ms(Clock::now() - start);

  stop.store(true, std::memory_order_release);
  worker_thread.join();

  const PacerCounters& c = pacer->counters();
  const InferenceWorker::Counters& w = worker.counters();
  std::printf(
      "         %d frames in %.1fms; submitted=%llu refused=%llu "
      "skipped_stale=%llu processed=%llu\n",
      kFrames, elapsed_ms, static_cast<unsigned long long>(c.submitted),
      static_cast<unsigned long long>(c.submit_skipped_full),
      static_cast<unsigned long long>(w.requests_skipped_stale),
      static_cast<unsigned long long>(w.processed));

  // Load shedding must actually engage -- 150 frames in against ~50 inferences
  // out means ~100 frames have to go somewhere. If nothing is shed, the test
  // is not covering the saturated regime it claims to.
  //
  // Which counter absorbs it is a property of the consumption policy, not of
  // the test: under latest-wins the worker drains the queue on every pickup, so
  // the shedding shows up as `requests_skipped_stale` and the render thread is
  // never refused at all. A FIFO worker would instead let the queue fill and
  // show `submit_skipped_full`. Accept either, and require that the totals
  // account for every frame.
  const std::uint64_t shed = c.submit_skipped_full + w.requests_skipped_stale;
  FP_CHECK_MSG(shed > 0, "expected the pipeline to shed load");
  FP_CHECK_EQ(c.submitted + c.submit_skipped_full, static_cast<std::uint64_t>(kFrames));
  FP_CHECK_MSG(c.results_ingested > 0, "no results reached the pacer");

  // The loop held its cadence rather than being dragged to inference speed.
  FP_CHECK_MSG(elapsed_ms < to_ms(kFrameBudget120Hz) * kFrames * 1.60,
               "render loop was dragged down to inference speed: " +
                   std::to_string(elapsed_ms) + "ms");
}
