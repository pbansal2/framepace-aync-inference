// FramePace - benchmark harness.
//
// Four stages, cheapest and most isolated first:
//   1. downsample kernel throughput, scalar vs SIMD
//   2. SPSC queue round-trip latency across two cores
//   3. pacer estimate cost (the per-frame extrapolation)
//   4. the full 120Hz pipeline, end to end
//
// All recorders are allocated up front. The measured regions contain no
// allocation, no I/O and no locking; printing happens after each stage
// completes. See tests/test_no_alloc.cpp for the enforcement of that claim.

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>

#include "framepace/clock.hpp"
#include "framepace/downsample.hpp"
#include "framepace/frame.hpp"
#include "framepace/inference_worker.hpp"
#include "framepace/pacer.hpp"
#include "framepace/spsc_queue.hpp"
#include "framepace/stats.hpp"

using namespace framepace;

namespace {

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

void rule(char c = '-') {
  char buf[79];
  std::memset(buf, c, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';
  std::printf("%s\n", buf);
}

void heading(const char* title) {
  std::printf("\n");
  rule('=');
  std::printf("%s\n", title);
  rule('=');
}

template<std::size_t N>
void report_latency(const char* label, SampleRecorder<N>& rec,
                    const char* unit = "ms") {
  rec.finalize();
  // Samples are nanoseconds; scale to the requested display unit.
  const double div = std::strcmp(unit, "ms") == 0 ? 1e6 : 1e3;
  const auto scaled = [&rec, div](double p) {
    return static_cast<double>(rec.percentile(p)) / div;
  };
  std::printf(
      "  %-26s n=%-7zu  p50=%8.3f  p90=%8.3f  p99=%8.3f  "
      "p99.9=%8.3f  max=%8.3f  %s\n",
      label, rec.count(), scaled(50), scaled(90), scaled(99), scaled(99.9),
      static_cast<double>(rec.max()) / div, unit);
  if (rec.dropped() > 0) {
    std::printf("  %-26s WARNING: %llu samples dropped (recorder full)\n", "",
                static_cast<unsigned long long>(rec.dropped()));
  }
}

/// Optimisation sink: forces `value` to be materialised so the compiler cannot
/// delete the computation that produced it. The "m" constraint (rather than
/// "r,m") works for every type including floats, which cannot live in a
/// general-purpose register.
template<typename T>
FRAMEPACE_ALWAYS_INLINE void keep(const T& value) noexcept {
  asm volatile("" : : "m"(value) : "memory");
}

// ---------------------------------------------------------------------------
// Stage 1: downsample throughput
// ---------------------------------------------------------------------------

void bench_downsample() {
  char title[96];
  std::snprintf(title, sizeof(title),
                "1. Downsample: %zux%zu -> %zux%zu (box, factor %u)", kSourceWidth,
                kSourceHeight, kTensorWidth, kTensorHeight, 1U << kDownsampleLog2);
  heading(title);

  auto frame = std::make_unique<SourceFrame>();
  Xorshift64 rng(0xD05A3B1ECAFEULL);
  for (auto& px : frame->data) {
    px = static_cast<std::uint8_t>(rng.next() & 0xFF);
  }
  auto scratch = std::make_unique<
      std::array<std::uint8_t, pow2_scratch_bytes(kSourceWidth, kSourceHeight)>>();
  auto tensor = std::make_unique<std::array<std::uint8_t, kTensorPixels>>();

  constexpr int kWarmup = 20;
  constexpr int kIters = 300;
  // Each level reads a quarter of the bytes the previous one did, so the chain
  // reads source_bytes * (1 + 1/4 + 1/16 + ...) over kDownsampleLog2 levels.
  double level_share = 0.0;
  for (unsigned level = 0; level < kDownsampleLog2; ++level) {
    level_share += 1.0 / static_cast<double>(1U << (2U * level));
  }
  const double kBytesRead =
      static_cast<double>(kSourceWidth * kSourceHeight) * level_share;

  // --- single-level kernels, scalar vs SIMD -------------------------------
  auto scalar_rec = make_recorder<kIters>();
  auto simd_rec = make_recorder<kIters>();

  for (int i = 0; i < kWarmup; ++i) {
    downsample_halve_u8_scalar(frame->pixels(), kSourceWidth, kSourceHeight,
                               SourceFrame::kStride, scratch->data(), kSourceWidth / 2);
  }
  for (int i = 0; i < kIters; ++i) {
    const Timestamp t0 = Clock::now();
    downsample_halve_u8_scalar(frame->pixels(), kSourceWidth, kSourceHeight,
                               SourceFrame::kStride, scratch->data(), kSourceWidth / 2);
    scalar_rec->record(Clock::now() - t0);
    keep(scratch->data());
  }

  for (int i = 0; i < kWarmup; ++i) {
    downsample_halve_u8(frame->pixels(), kSourceWidth, kSourceHeight,
                        SourceFrame::kStride, scratch->data(), kSourceWidth / 2);
  }
  for (int i = 0; i < kIters; ++i) {
    const Timestamp t0 = Clock::now();
    downsample_halve_u8(frame->pixels(), kSourceWidth, kSourceHeight,
                        SourceFrame::kStride, scratch->data(), kSourceWidth / 2);
    simd_rec->record(Clock::now() - t0);
    keep(scratch->data());
  }

  // --- the full chain the pacer actually runs ------------------------------
  auto chain_rec = make_recorder<kIters>();
  for (int i = 0; i < kWarmup; ++i) {
    (void)downsample_pow2_u8(frame->pixels(), kSourceWidth, kSourceHeight,
                             SourceFrame::kStride, kDownsampleLog2, tensor->data(),
                             kTensorWidth, *scratch);
  }
  for (int i = 0; i < kIters; ++i) {
    const Timestamp t0 = Clock::now();
    (void)downsample_pow2_u8(frame->pixels(), kSourceWidth, kSourceHeight,
                             SourceFrame::kStride, kDownsampleLog2, tensor->data(),
                             kTensorWidth, *scratch);
    chain_rec->record(Clock::now() - t0);
    keep(tensor->data());
  }

  std::printf("  kernel in use: %s\n\n", kernel_name(active_kernel()));
  report_latency("halve (scalar)", *scalar_rec, "us");
  report_latency("halve (simd)", *simd_rec, "us");
  char chain_label[32];
  std::snprintf(chain_label, sizeof(chain_label), "full chain (%u levels)",
                kDownsampleLog2);
  report_latency(chain_label, *chain_rec, "us");

  scalar_rec->finalize();
  simd_rec->finalize();
  chain_rec->finalize();
  const double scalar_us = static_cast<double>(scalar_rec->percentile(50)) / 1e3;
  const double simd_us = static_cast<double>(simd_rec->percentile(50)) / 1e3;
  const double chain_us = static_cast<double>(chain_rec->percentile(50)) / 1e3;

  std::printf("\n  SIMD speedup (single level, p50): %.2fx\n",
              simd_us > 0.0 ? scalar_us / simd_us : 0.0);
  std::printf("  chain throughput:                 %.2f GB/s\n",
              chain_us > 0.0 ? kBytesRead / (chain_us * 1e3) : 0.0);
  std::printf("  share of an 8.333ms frame budget: %.2f%%\n",
              100.0 * chain_us / (to_ms(kFrameBudget120Hz) * 1e3));
}

// ---------------------------------------------------------------------------
// Stage 2: SPSC round-trip latency
// ---------------------------------------------------------------------------

void bench_spsc_roundtrip() {
  heading("2. SPSC queue: cross-core round-trip latency");

  struct Ping {
    std::int64_t sent_ns;
    std::uint64_t seq;
  };

  constexpr int kIters = 50'000;
  auto to_worker = std::make_unique<SpscQueue<Ping, 64>>();
  auto to_main = std::make_unique<SpscQueue<Ping, 64>>();
  auto rec = make_recorder<kIters>();

  std::atomic<bool> stop{false};
  std::thread echo([&] {
    Ping p{};
    while (!stop.load(std::memory_order_acquire)) {
      if (!to_worker->try_pop(p)) {
        cpu_relax();
        continue;
      }
      while (!to_main->try_push(p)) {
        if (stop.load(std::memory_order_acquire)) {
          return;
        }
        cpu_relax();
      }
    }
  });

  // Warm both threads onto their cores before measuring.
  for (int i = 0; i < 2000; ++i) {
    Ping p{to_ns(Clock::now()), static_cast<std::uint64_t>(i)};
    while (!to_worker->try_push(p)) {
      cpu_relax();
    }
    while (!to_main->try_pop(p)) {
      cpu_relax();
    }
  }

  for (int i = 0; i < kIters; ++i) {
    Ping out{to_ns(Clock::now()), static_cast<std::uint64_t>(i)};
    while (!to_worker->try_push(out)) {
      cpu_relax();
    }
    Ping back{};
    while (!to_main->try_pop(back)) {
      cpu_relax();
    }
    rec->record(to_ns(Clock::now()) - back.sent_ns);
  }

  stop.store(true, std::memory_order_release);
  echo.join();

  report_latency("push -> pop -> push -> pop", *rec, "us");
  rec->finalize();
  std::printf("\n  one-way hop (p50 / 2): %.3f us\n",
              static_cast<double>(rec->percentile(50)) / 2e3);
}

// ---------------------------------------------------------------------------
// Stage 3: per-frame estimate cost
// ---------------------------------------------------------------------------

void bench_estimate() {
  heading("3. Extrapolation: cost of one per-frame estimate");

  ResultTracker tracker;
  InferenceResult r{};
  r.feature_count = static_cast<std::uint32_t>(kMaxFeatures);
  for (std::uint64_t i = 1; i <= 2; ++i) {
    r.frame_id = i;
    r.capture_ns = static_cast<std::int64_t>(i) * 8'333'333;
    for (std::size_t f = 0; f < kMaxFeatures; ++f) {
      r.features[f] = static_cast<float>(i * f);
    }
    (void)tracker.ingest(r);
  }

  constexpr int kIters = 200'000;
  constexpr int kBlock = 100;
  auto rec = make_recorder<kIters / kBlock>();
  const Timestamp at = from_ns(r.capture_ns + 10'000'000);

  for (int i = 0; i < 10'000; ++i) {
    const Estimate e = tracker.estimate_at(at, 33ms);
    keep(e.features[0]);
  }
  // Timed in blocks: a single estimate is on the order of the clock's own read
  // latency, so per-call timing would measure clock_gettime instead.
  for (int i = 0; i < kIters / kBlock; ++i) {
    const Timestamp t0 = Clock::now();
    for (int j = 0; j < kBlock; ++j) {
      const Estimate e = tracker.estimate_at(at, 33ms);
      keep(e.features[0]);
    }
    rec->record((Clock::now() - t0).count() / kBlock);
  }

  rec->finalize();
  std::printf("  %d features, %d calls in blocks of %d\n",
              static_cast<int>(kMaxFeatures), kIters, kBlock);
  std::printf("  p50=%lld ns   p99=%lld ns   max=%lld ns\n",
              static_cast<long long>(rec->percentile(50)),
              static_cast<long long>(rec->percentile(99)),
              static_cast<long long>(rec->max()));
}

// ---------------------------------------------------------------------------
// Stage 4: the full pipeline
// ---------------------------------------------------------------------------

struct PipelineOptions {
  int frames = 1200;  // 10s at 120Hz
  Duration min_latency = 15ms;
  Duration max_latency = 30ms;
  unsigned submit_interval = 1;
  bool busy_spin = false;
};

void bench_pipeline(const PipelineOptions& opts) {
  heading("4. Full pipeline: 120Hz render loop + async inference");

  Pacer::Config pcfg;
  pcfg.submit_interval = opts.submit_interval;
  auto pacer = std::make_unique<Pacer>(pcfg);

  // A ring of distinct pre-built frames, standing in for a camera handing over
  // a fresh buffer each frame.
  //
  // The obvious alternative -- memset one buffer per iteration -- costs ~2MB of
  // stores inside the render loop and evicts L2 before the downsample reads it.
  // That is harness cost, not engine cost, and it dominated the jitter figures:
  // it inflated p99 interval jitter several-fold. Building the frames up front
  // measures the engine instead.
  //
  // The pattern is a cyclically shifted gradient, so block-mean features move
  // smoothly (and wrap without a discontinuity) and the extrapolator has real
  // velocity to work with rather than noise.
  // Four 2MB buffers (8MB). Enough to mimic real buffer rotation and to keep
  // the downsample reading genuinely cold data, but small enough not to thrash
  // the shared cache: a deeper ring measures the memory system more than the
  // engine and inflated per-frame CPU ~3x in testing.
  constexpr std::size_t kFrameRing = 4;
  auto frames = std::make_unique<std::array<SourceFrame, kFrameRing>>();
  for (std::size_t k = 0; k < kFrameRing; ++k) {
    SourceFrame& f = (*frames)[k];
    const auto shift = static_cast<std::uint8_t>(k * (256 / kFrameRing));
    for (std::size_t y = 0; y < kSourceHeight; ++y) {
      std::uint8_t* row = f.row(y);
      for (std::size_t x = 0; x < kSourceWidth; ++x) {
        row[x] = static_cast<std::uint8_t>((x + y + shift) & 0xFF);
      }
    }
  }

  InferenceWorker::Config wcfg;
  wcfg.min_latency = opts.min_latency;
  wcfg.max_latency = opts.max_latency;
  wcfg.busy_spin = opts.busy_spin;
  InferenceWorker worker(pacer->requests(), pacer->results(), wcfg);

  // Capacity sized for the configured run, allocated before the loop starts.
  constexpr std::size_t kMaxSamples = 20'000;
  auto cpu_rec = make_recorder<kMaxSamples>();       // step() cost, no sleep
  auto interval_rec = make_recorder<kMaxSamples>();  // presented frame interval
  auto jitter_rec = make_recorder<kMaxSamples>();    // |interval - budget|
  auto age_rec = make_recorder<kMaxSamples>();       // estimate source age
  auto e2e_rec = make_recorder<kMaxSamples>();       // capture -> result visible

  std::atomic<bool> stop{false};
  std::thread worker_thread([&] { worker.run(stop); });

  std::printf("  frames: %d   inference: %.1f-%.1fms %s   submit every %u frame(s)\n",
              opts.frames, to_ms(opts.min_latency), to_ms(opts.max_latency),
              opts.busy_spin ? "(busy-spin)" : "(sleep)", opts.submit_interval);
  std::printf("  warming up...\n");
  std::fflush(stdout);

  Timestamp deadline = Clock::now() + kFrameBudget120Hz;
  for (int i = 0; i < 120; ++i) {
    const SourceFrame& f = (*frames)[static_cast<std::size_t>(i) % kFrameRing];
    (void)pacer->step(f, static_cast<std::uint64_t>(i), Clock::now(), deadline);
    deadline = pacer->sleep_until_next_frame(deadline);
  }

  std::printf("  measuring...\n");
  std::fflush(stdout);

  std::uint64_t last_ingested = pacer->counters().results_ingested;
  Timestamp last_present = Clock::now();
  const Timestamp run_start = Clock::now();

  for (int i = 0; i < opts.frames; ++i) {
    const auto frame_id = static_cast<std::uint64_t>(120 + i);
    const SourceFrame& f = (*frames)[static_cast<std::size_t>(i) % kFrameRing];

    const Timestamp capture = Clock::now();
    const Estimate est = pacer->step(f, frame_id, capture, deadline);
    const Timestamp after_work = Clock::now();

    cpu_rec->record(after_work - capture);
    interval_rec->record(after_work - last_present);
    const auto delta = (after_work - last_present) - pcfg.frame_budget;
    jitter_rec->record(delta.count() < 0 ? -delta.count() : delta.count());
    last_present = after_work;

    if (est.valid) {
      age_rec->record(est.source_age);
    }
    // A newly ingested result means the renderer just became aware of an
    // inference. Measure capture -> now, not Estimate::source_age: the latter
    // is relative to the presentation deadline and so adds up to a full frame
    // of look-ahead, overstating pipeline latency by ~8ms.
    const std::uint64_t ingested = pacer->counters().results_ingested;
    if (ingested != last_ingested && est.valid) {
      e2e_rec->record(std::chrono::nanoseconds{to_ns(after_work) -
                                               pacer->tracker().newest_capture_ns()});
      last_ingested = ingested;
    }

    deadline = pacer->sleep_until_next_frame(deadline);
  }

  const double wall_ms = to_ms(Clock::now() - run_start);
  stop.store(true, std::memory_order_release);
  worker_thread.join();

  // --- results ------------------------------------------------------------
  const PacerCounters& c = pacer->counters();
  const InferenceWorker::Counters& w = worker.counters();

  std::printf("\n  Timing\n");
  rule();
  report_latency("render-loop CPU per frame", *cpu_rec, "us");
  report_latency("presented frame interval", *interval_rec, "ms");
  report_latency("interval jitter |err|", *jitter_rec, "us");
  report_latency("estimate source age", *age_rec, "ms");
  report_latency("capture -> visible (e2e)", *e2e_rec, "ms");

  const double target_ms = to_ms(pcfg.frame_budget) * opts.frames;
  std::printf("\n  Cadence\n");
  rule();
  std::printf("  %-34s %.1f ms\n", "wall clock", wall_ms);
  std::printf("  %-34s %.1f ms\n", "target", target_ms);
  std::printf("  %-34s %+.2f %%\n", "deviation",
              100.0 * (wall_ms - target_ms) / target_ms);
  std::printf("  %-34s %.2f fps\n", "effective rate",
              wall_ms > 0.0 ? 1000.0 * opts.frames / wall_ms : 0.0);

  cpu_rec->finalize();
  std::printf("  %-34s %.2f %% of 8.333ms\n", "render CPU budget used (p99)",
              100.0 * static_cast<double>(cpu_rec->percentile(99)) /
                  static_cast<double>(pcfg.frame_budget.count()));

  std::printf("\n  Pipeline counters\n");
  rule();
  std::printf("  %-34s %llu\n", "frames", static_cast<unsigned long long>(c.frames));
  std::printf("  %-34s %llu\n", "inference submitted",
              static_cast<unsigned long long>(c.submitted));
  std::printf("  %-34s %llu\n", "submits refused (queue full)",
              static_cast<unsigned long long>(c.submit_skipped_full));
  std::printf("  %-34s %llu\n", "submits throttled (cadence)",
              static_cast<unsigned long long>(c.submit_skipped_cadence));
  std::printf("  %-34s %llu\n", "requests skipped as stale",
              static_cast<unsigned long long>(w.requests_skipped_stale));
  std::printf("  %-34s %llu\n", "inferences completed",
              static_cast<unsigned long long>(w.processed));
  std::printf("  %-34s %llu\n", "results ingested",
              static_cast<unsigned long long>(c.results_ingested));
  std::printf("  %-34s %llu\n", "results dropped as stale",
              static_cast<unsigned long long>(c.results_dropped_stale));
  std::printf("  %-34s %llu\n", "results dropped (queue full)",
              static_cast<unsigned long long>(w.results_dropped_full));
  std::printf("  %-34s %llu\n", "frames with no estimate yet",
              static_cast<unsigned long long>(c.frames_without_estimate));
  const double clamp_pct =
      c.frames > 0 ? 100.0 * static_cast<double>(c.frames_extrapolation_clamped) /
                         static_cast<double>(c.frames)
                   : 0.0;
  std::printf("  %-34s %llu (%.1f%% of frames)\n", "frames at extrap. ceiling",
              static_cast<unsigned long long>(c.frames_extrapolation_clamped),
              clamp_pct);
  if (clamp_pct > 50.0) {
    std::printf(
        "\n  NOTE: the extrapolation ceiling is binding on most frames,\n"
        "        so latency hiding is saturated. Either Pacer::Config::\n"
        "        max_extrapolation is too low for this inference\n"
        "        latency, or the pipeline is slower than configured.\n");
  }

  const double frames_per_inference =
      w.processed > 0 ? static_cast<double>(c.frames) / static_cast<double>(w.processed)
                      : 0.0;
  std::printf("\n  %-34s %.2f\n", "frames presented per inference",
              frames_per_inference);
  std::printf(
      "  %-34s %.1f %%\n", "frames reusing a prior inference",
      c.frames > 0
          ? 100.0 *
                (1.0 - 1.0 / (frames_per_inference > 0.0 ? frames_per_inference : 1.0))
          : 0.0);
}

void print_env() {
  rule('=');
  std::printf("FramePace benchmark\n");
  rule('=');
  std::printf("  %-22s %s\n", "SIMD kernel", kernel_name(active_kernel()));
  std::printf("  %-22s %zu bytes\n", "cache line", kCacheLineSize);
  std::printf("  %-22s %u\n", "hardware threads", std::thread::hardware_concurrency());
  std::printf("  %-22s %.4f ms (120 FPS)\n", "frame budget", to_ms(kFrameBudget120Hz));
  std::printf("  %-22s %zux%zu -> %zux%zu\n", "geometry", kSourceWidth, kSourceHeight,
              kTensorWidth, kTensorHeight);
  std::printf("  %-22s %zu KB\n", "request queue",
              (sizeof(InferenceRequest) * kRequestQueueDepth) / 1024);
  std::printf("  %-22s %s\n", "ThreadSanitizer",
              kThreadSanitizerEnabled ? "ENABLED" : "off");
  std::printf("  %-22s %s\n", "AddressSanitizer",
              kAddressSanitizerEnabled ? "ENABLED" : "off");
  if (kAnySanitizerEnabled) {
    std::printf(
        "\n  NOTE: a sanitizer is active. Timings below are NOT\n"
        "        representative -- rebuild without sanitizers for\n"
        "        meaningful numbers.\n");
  }
}

void usage(const char* argv0) {
  std::printf(
      "usage: %s [options]\n"
      "  --frames N            frames to measure (default 1200, ~10s)\n"
      "  --min-latency MS      inference floor in ms (default 15)\n"
      "  --max-latency MS      inference ceiling in ms (default 30)\n"
      "  --submit-interval N   submit inference every Nth frame (default 1)\n"
      "  --busy-spin           worker spins instead of sleeping\n"
      "  --only STAGE          downsample | spsc | estimate | pipeline\n"
      "  --help\n",
      argv0);
}

}  // namespace

int main(int argc, char** argv) {
  PipelineOptions opts;
  const char* only = nullptr;

  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    const bool has_value = (i + 1) < argc;
    if (std::strcmp(a, "--help") == 0 || std::strcmp(a, "-h") == 0) {
      usage(argv[0]);
      return EXIT_SUCCESS;
    }
    if (std::strcmp(a, "--busy-spin") == 0) {
      opts.busy_spin = true;
    } else if (std::strcmp(a, "--frames") == 0 && has_value) {
      opts.frames = std::atoi(argv[++i]);
    } else if (std::strcmp(a, "--min-latency") == 0 && has_value) {
      opts.min_latency = std::chrono::microseconds{
          static_cast<std::int64_t>(std::atof(argv[++i]) * 1000.0)};
    } else if (std::strcmp(a, "--max-latency") == 0 && has_value) {
      opts.max_latency = std::chrono::microseconds{
          static_cast<std::int64_t>(std::atof(argv[++i]) * 1000.0)};
    } else if (std::strcmp(a, "--submit-interval") == 0 && has_value) {
      opts.submit_interval = static_cast<unsigned>(std::max(1, std::atoi(argv[++i])));
    } else if (std::strcmp(a, "--only") == 0 && has_value) {
      only = argv[++i];
    } else {
      std::fprintf(stderr, "unknown or incomplete option: %s\n\n", a);
      usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  if (opts.frames <= 0) {
    std::fprintf(stderr, "--frames must be positive\n");
    return EXIT_FAILURE;
  }
  if (opts.max_latency < opts.min_latency) {
    std::fprintf(stderr, "--max-latency must be >= --min-latency\n");
    return EXIT_FAILURE;
  }

  const auto wants = [only](const char* stage) {
    return only == nullptr || std::strcmp(only, stage) == 0;
  };

  print_env();
  if (wants("downsample")) {
    bench_downsample();
  }
  if (wants("spsc")) {
    bench_spsc_roundtrip();
  }
  if (wants("estimate")) {
    bench_estimate();
  }
  if (wants("pipeline")) {
    bench_pipeline(opts);
  }
  std::printf("\n");
  return EXIT_SUCCESS;
}
