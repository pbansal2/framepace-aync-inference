#include "framepace/inference_worker.hpp"

#include <algorithm>
#include <thread>

namespace framepace {
namespace {

/// Reduce the tensor into `kMaxFeatures` block means.
///
/// Stands in for a real model's forward pass. Partitions the tensor into
/// `kMaxFeatures` contiguous spans and averages each. Cheap (one pass, ~32KB)
/// but data-dependent, and smooth with respect to the input, which is what the
/// pacer's extrapolation needs.
void extract_features(const InferenceRequest& request,
                      std::array<float, kMaxFeatures>& out) noexcept {
  const std::size_t pixels = static_cast<std::size_t>(request.width) * request.height;
  const std::size_t block = pixels / kMaxFeatures;

  if (block == 0) {
    out.fill(0.0F);
    return;
  }
  for (std::size_t f = 0; f < kMaxFeatures; ++f) {
    const std::uint8_t* p = request.tensor.data() + f * block;
    std::uint32_t sum = 0;
    for (std::size_t i = 0; i < block; ++i) {
      sum += p[i];
    }
    out[f] = static_cast<float>(sum) / static_cast<float>(block * 255U);
  }
}

}  // namespace

InferenceWorker::InferenceWorker(RequestQueue& requests, ResultQueue& results,
                                 const Config& config) noexcept
    : requests_(requests), results_(results), config_(config), rng_(config.seed) {}

bool InferenceWorker::step() noexcept {
  // Latest-wins, and read in place.
  //
  // In place because the tensor is 32KB: copying it out just to reduce it would
  // double the memory traffic of the whole inference stage. Latest-wins because
  // a FIFO read would always pick the oldest queued frame, adding one inference
  // time of staleness per queue slot -- with depth 4 and 22ms inference that is
  // ~90ms of latency for nothing. Inferring on a frame the renderer presented
  // 90ms ago has no value; the stale ones are dropped instead.
  std::size_t skipped = 0;
  const InferenceRequest* request = requests_.read_latest_slot(skipped);
  if (request == nullptr) {
    return false;
  }
  counters_.requests_skipped_stale += skipped;

  const Timestamp started = Clock::now();

  InferenceResult result{};
  result.frame_id = request->frame_id;
  result.capture_ns = request->capture_ns;
  result.feature_count = static_cast<std::uint32_t>(kMaxFeatures);
  result.reserved = 0;
  extract_features(*request, result.features);

  // Done with the ring slot; hand it back before we spend the latency budget so
  // the render thread regains a submission slot as early as possible.
  requests_.release();

  const auto min_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(config_.min_latency).count();
  const auto max_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(config_.max_latency).count();
  const Timestamp target =
      started + std::chrono::nanoseconds{rng_.range(min_ns, max_ns)};

  if (config_.busy_spin) {
    while (Clock::now() < target) {
      cpu_relax();
    }
  } else if (Clock::now() < target) {
    // 150us spin margin: tighter than the pacer's, because the worker's
    // deadline only shapes the simulated latency distribution and does not
    // need to hold a presentation cadence.
    precise_sleep_until(target, 150us);
  }

  result.complete_ns = to_ns(Clock::now());
  ++counters_.processed;

  if (!results_.try_push(result)) {
    ++counters_.results_dropped_full;
  }
  return true;
}

void InferenceWorker::run(const std::atomic<bool>& stop) noexcept {
  while (!stop.load(std::memory_order_acquire)) {
    if (!step()) {
      ++counters_.idle_polls;
      std::this_thread::sleep_for(config_.idle_backoff);
    }
  }
}

}  // namespace framepace
