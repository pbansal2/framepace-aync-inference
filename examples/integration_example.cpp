// FramePace - integration example: plug in your own model and your own engine.
//
//   ./build/integration_example [--frames N] [--latency-ms MS]
//
// Copy this file as the starting point for an integration. There are three
// places to edit, each marked INTEGRATION POINT:
//
//   1. capture_frame()  your engine or camera writes an 8-bit luma frame
//   2. run_model()      your model turns the tensor into a feature vector
//   3. use_estimate()   your renderer consumes the per-frame estimate
//
// Everything else is wiring to keep as it is: one worker thread owns the
// consumer end of the request queue and the producer end of the result queue,
// and the render loop calls Pacer::step() once per frame. Nothing here modifies
// the library -- this uses only the public API.
//
// The scene is a bright square moving on a circle, and the "model" finds its
// centroid. Because the true position is known analytically, the example
// reports how far off the renderer would be with and without FramePace's
// extrapolation, so it measures the latency hiding rather than just running it.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "framepace/clock.hpp"
#include "framepace/downsample.hpp"
#include "framepace/frame.hpp"
#include "framepace/pacer.hpp"

using namespace framepace;

namespace {

constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// The synthetic scene (stands in for your engine's world state)
// ---------------------------------------------------------------------------

/// True centre of the square at time `t_s`, normalised to [0, 1] on each axis.
/// One revolution every 2 s on a circle of radius 0.3 x frame height.
struct Point {
  double x;
  double y;
};

Point scene_position(double t_s) {
  const double radius_y = 0.3;
  const double radius_x =
      radius_y * static_cast<double>(kSourceHeight) / static_cast<double>(kSourceWidth);
  const double angle = 2.0 * kPi * t_s / 2.0;
  return {0.5 + radius_x * std::cos(angle), 0.5 + radius_y * std::sin(angle)};
}

// ---------------------------------------------------------------------------
// INTEGRATION POINT 1: produce the source frame
// ---------------------------------------------------------------------------
// FramePace takes one 8-bit luma plane of kSourceWidth x kSourceHeight (set via
// the FRAMEPACE_SOURCE_WIDTH / FRAMEPACE_SOURCE_HEIGHT CMake options). Write
// your camera image or render target into `frame` here. For RGB(A) input,
// convert to luma first, e.g. Y = (77*R + 150*G + 29*B) >> 8 per pixel.

void capture_frame(SourceFrame& frame, double t_s) {
  std::memset(frame.pixels(), 16, SourceFrame::kBytes);  // dark background

  const Point p = scene_position(t_s);
  const auto side = static_cast<long>(kSourceHeight / 8);
  const auto x0 = static_cast<long>(std::lround(p.x * kSourceWidth)) - side / 2;
  const auto y0 = static_cast<long>(std::lround(p.y * kSourceHeight)) - side / 2;
  for (long y = std::max(y0, 0L);
       y < std::min(y0 + side, static_cast<long>(kSourceHeight)); ++y) {
    const long xa = std::max(x0, 0L);
    const long xb = std::min(x0 + side, static_cast<long>(kSourceWidth));
    if (xb > xa) {
      std::memset(frame.row(static_cast<std::size_t>(y)) + xa, 255,
                  static_cast<std::size_t>(xb - xa));
    }
  }
}

// ---------------------------------------------------------------------------
// INTEGRATION POINT 2: run your model
// ---------------------------------------------------------------------------
// Input: the downsampled tensor, already normalised to float [0, 1], row-major,
// `width` x `height` (kTensorWidth x kTensorHeight). Output: up to kMaxFeatures
// floats in `out.features`, with `out.feature_count` set to how many you used.
//
// Replace the body with your inference call (ONNX Runtime, Core ML, TFLite,
// TensorRT, a hand-written network...). This function runs on the worker
// thread, never on the render thread, so it may take as long as it needs.
// Allocate model buffers once, before the loop, not here.

void run_model(const float* input, std::size_t width, std::size_t height,
               InferenceResult& out) {
  // Stand-in model: centroid of the bright pixels, in normalised coordinates.
  double sum = 0.0;
  double sx = 0.0;
  double sy = 0.0;
  for (std::size_t y = 0; y < height; ++y) {
    for (std::size_t x = 0; x < width; ++x) {
      const double v = static_cast<double>(input[y * width + x]);
      if (v > 0.5) {
        sum += v;
        sx += v * static_cast<double>(x);
        sy += v * static_cast<double>(y);
      }
    }
  }
  // +0.5: a tensor pixel's centre sits halfway across the source block it
  // averages, so this maps tensor indices back to the source coordinate frame.
  out.features[0] =
      sum > 0.0 ? static_cast<float>((sx / sum + 0.5) / static_cast<double>(width))
                : 0.0F;
  out.features[1] =
      sum > 0.0 ? static_cast<float>((sy / sum + 0.5) / static_cast<double>(height))
                : 0.0F;
  out.feature_count = 2;
}

// ---------------------------------------------------------------------------
// The model worker thread: keep this wiring as is
// ---------------------------------------------------------------------------

struct WorkerStats {
  std::uint64_t inferences = 0;
  std::uint64_t skipped_stale = 0;
  std::uint64_t results_dropped = 0;
};

/// The contract this loop follows, which any replacement must keep:
///   - Exactly one thread calls the request queue's consumer API and the result
///     queue's producer API. This is that thread.
///   - Every non-null read_latest_slot() is matched by exactly one release().
///   - The request pointer is invalid after release(); copy what you need first.
///   - Never block on a full result queue: drop and move on. The renderer only
///     ever wants the newest result.
void model_worker(RequestQueue& requests, ResultQueue& results,
                  const std::atomic<bool>& stop, Duration emulated_latency,
                  WorkerStats& stats) {
  // Model input buffer: allocated once, before the loop.
  auto input = std::make_unique<std::array<float, kTensorPixels>>();

  while (!stop.load(std::memory_order_acquire)) {
    // Latest-wins: skip straight to the newest queued frame. Older ones are
    // stale by the time this model would finish them.
    std::size_t skipped = 0;
    const InferenceRequest* request = requests.read_latest_slot(skipped);
    if (request == nullptr) {
      std::this_thread::sleep_for(200us);  // idle: nothing queued yet
      continue;
    }
    stats.skipped_stale += skipped;
    const Timestamp started = Clock::now();

    InferenceResult result{};
    result.frame_id = request->frame_id;
    result.capture_ns = request->capture_ns;  // the basis for extrapolation
    const std::size_t width = request->width;
    const std::size_t height = request->height;
    normalize_u8_to_f32(request->tensor, *input);  // u8 [0,255] -> f32 [0,1]

    // The tensor is copied out, so hand the slot back now rather than after a
    // long inference: the render thread can keep submitting meanwhile.
    requests.release();

    run_model(input->data(), width, height, result);

    // Emulate a heavier model so the example shows realistic latency hiding.
    // DELETE THIS when run_model() is your real model.
    std::this_thread::sleep_until(started + emulated_latency);

    result.complete_ns = to_ns(Clock::now());
    ++stats.inferences;
    if (!results.try_push(result)) {
      ++stats.results_dropped;
    }
  }
}

// ---------------------------------------------------------------------------
// INTEGRATION POINT 3: consume the estimate
// ---------------------------------------------------------------------------
// Called once per frame on the render thread with the features projected to
// the frame's presentation time. Draw with `est.features` here. Check
// `est.valid` first: there is no estimate until the first inference lands.
//
// In this example "rendering" means scoring the estimate against the true
// position, alongside the newest raw result, which is what you would draw
// without FramePace's extrapolation.

struct ErrorLog {
  std::vector<double> raw;        // newest result as-is
  std::vector<double> estimated;  // FramePace estimate
};

void use_estimate(const Estimate& est, const Estimate& raw, Point truth,
                  ErrorLog& log) {
  if (!est.valid || !raw.valid || est.feature_count < 2) {
    return;
  }
  const auto pixel_error = [truth](const Estimate& e) {
    const double dx = (static_cast<double>(e.features[0]) - truth.x) *
                      static_cast<double>(kSourceWidth);
    const double dy = (static_cast<double>(e.features[1]) - truth.y) *
                      static_cast<double>(kSourceHeight);
    return std::sqrt(dx * dx + dy * dy);
  };
  log.estimated.push_back(pixel_error(est));
  log.raw.push_back(pixel_error(raw));
}

double mean(const std::vector<double>& v) {
  double s = 0.0;
  for (double x : v) {
    s += x;
  }
  return v.empty() ? 0.0 : s / static_cast<double>(v.size());
}

double p95(std::vector<double> v) {
  if (v.empty()) {
    return 0.0;
  }
  std::sort(v.begin(), v.end());
  return v[static_cast<std::size_t>(0.95 * static_cast<double>(v.size() - 1))];
}

}  // namespace

int main(int argc, char** argv) {
  int frames = 600;  // 5 s at 120 Hz
  double latency_ms = 20.0;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
      frames = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--latency-ms") == 0 && i + 1 < argc) {
      latency_ms = std::atof(argv[++i]);
    } else {
      std::printf("usage: %s [--frames N] [--latency-ms MS]\n", argv[0]);
      return std::strcmp(argv[i], "--help") == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
  }
  if (frames <= 0 || latency_ms < 0.0) {
    std::fprintf(stderr, "--frames must be > 0 and --latency-ms >= 0\n");
    return EXIT_FAILURE;
  }

  // --- startup: every allocation happens here, before the loop ------------
  // Pacer carries ~1MB of inline scratch: heap-allocate it once, never per frame.
  auto pacer = std::make_unique<Pacer>(Pacer::Config{});
  auto frame = std::make_unique<SourceFrame>();
  ErrorLog log;
  log.raw.reserve(static_cast<std::size_t>(frames));
  log.estimated.reserve(static_cast<std::size_t>(frames));

  const auto latency = std::chrono::duration_cast<Duration>(
      std::chrono::duration<double, std::milli>{latency_ms});
  std::atomic<bool> stop{false};
  WorkerStats stats;
  std::thread worker(
      [&] { model_worker(pacer->requests(), pacer->results(), stop, latency, stats); });

  std::printf("FramePace integration example\n");
  std::printf("  geometry        %zux%zu -> %zux%zu luma, 2 of %zu features used\n",
              kSourceWidth, kSourceHeight, kTensorWidth, kTensorHeight, kMaxFeatures);
  std::printf("  model latency   %.1f ms (emulated)\n", latency_ms);
  std::printf("  running         %d frames at 120 Hz...\n\n", frames);
  std::fflush(stdout);

  // --- the render loop: one Pacer::step() per frame ------------------------
  const Timestamp t0 = Clock::now();
  Timestamp deadline = t0 + kFrameBudget120Hz;
  for (int i = 0; i < frames; ++i) {
    const Timestamp capture = Clock::now();
    capture_frame(*frame, std::chrono::duration<double>(capture - t0).count());

    // Ingests the newest result, submits this frame for inference if there is
    // room, and returns features projected to `deadline`. Never blocks.
    const Estimate est =
        pacer->step(*frame, static_cast<std::uint64_t>(i), capture, deadline);

    const Estimate raw = pacer->tracker().estimate_at(deadline, Duration::zero());
    const Point truth =
        scene_position(std::chrono::duration<double>(deadline - t0).count());
    use_estimate(est, raw, truth, log);

    deadline = pacer->sleep_until_next_frame(deadline);
  }
  const double wall_s = std::chrono::duration<double>(Clock::now() - t0).count();

  stop.store(true, std::memory_order_release);
  worker.join();

  // --- report --------------------------------------------------------------
  const PacerCounters& c = pacer->counters();
  std::printf("Render loop\n");
  std::printf("  effective rate         %.1f fps\n",
              static_cast<double>(frames) / wall_s);
  std::printf("  frames with estimate   %zu / %d\n", log.estimated.size(), frames);
  std::printf("  inferences completed   %llu (%.1f frames each)\n",
              static_cast<unsigned long long>(stats.inferences),
              stats.inferences > 0
                  ? static_cast<double>(frames) / static_cast<double>(stats.inferences)
                  : 0.0);
  std::printf("  stale frames skipped   %llu\n",
              static_cast<unsigned long long>(stats.skipped_stale));
  std::printf("  submits refused        %llu\n",
              static_cast<unsigned long long>(c.submit_skipped_full));

  const double raw_mean = mean(log.raw);
  const double est_mean = mean(log.estimated);
  std::printf(
      "\nPosition error of what the renderer draws (source pixels, lower is better)\n");
  std::printf("                           mean      p95\n");
  std::printf("  newest result as-is   %7.1f  %7.1f\n", raw_mean, p95(log.raw));
  std::printf("  FramePace estimate    %7.1f  %7.1f\n", est_mean, p95(log.estimated));
  if (raw_mean > 0.0) {
    std::printf("  -> extrapolation removes %.0f%% of the latency-induced error\n",
                100.0 * (1.0 - est_mean / raw_mean));
  }

  // Non-zero exit only if the pipeline itself failed, so this doubles as a
  // smoke test. The error figures are reported, not asserted: they depend on
  // scheduling and are meaningful on a quiet machine, not a shared CI runner.
  const bool pipeline_ok = stats.inferences > 0 &&
                           log.estimated.size() > static_cast<std::size_t>(frames) / 2;
  if (!pipeline_ok) {
    std::fprintf(stderr, "\nFAIL: the pipeline produced too few estimates\n");
  }
  return pipeline_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
