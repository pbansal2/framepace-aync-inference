// FramePace - proof that the hot path performs zero heap allocations.
//
// Code review cannot establish this claim: a hidden allocation can come from a
// std::function, a container growth, a lambda capture promoted to the heap, a
// std::string in a log line, or an over-aligned type the compiler routes
// through aligned operator new. So this target replaces the global allocator
// with a counting one, arms the counter around the steady-state loop, and
// asserts the count is exactly zero.
//
// Every global allocation form is overridden below, including the C++17
// aligned variants -- the engine's types are `alignas(128)`, so a miss there
// would route straight past the counter and the test would pass vacuously.
//
// Startup allocations (the Pacer, the frame buffers, std::thread) are expected
// and deliberately excluded: they happen once, before the counter is armed.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <thread>

#if defined(_WIN32)
#include <malloc.h>  // _aligned_malloc / _aligned_free
#endif

#include "framepace/downsample.hpp"
#include "framepace/frame.hpp"
#include "framepace/inference_worker.hpp"
#include "framepace/pacer.hpp"
#include "framepace/spsc_queue.hpp"
#include "framepace/stats.hpp"
#include "test_util.hpp"

using namespace framepace;

// ---------------------------------------------------------------------------
// Counting allocator
// ---------------------------------------------------------------------------

// GCC's -Wmismatched-new-delete is a false positive for this whole section.
// It inlines a replacement `operator delete` into its caller, sees `free()`
// applied to a pointer from `operator new`, and reports a mismatch -- it cannot
// know that the replacement `operator new` below *is* malloc/posix_memalign,
// for which `free()` is exactly the right release. Scoped to the allocator
// replacement only; every other TU keeps the warning.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

namespace alloc_probe {

std::atomic<std::uint64_t> total{0};
std::atomic<std::uint64_t> armed_count{0};
std::atomic<std::uint64_t> armed_bytes{0};
std::atomic<bool> armed{false};

inline void note(std::size_t bytes) noexcept {
  total.fetch_add(1, std::memory_order_relaxed);
  if (armed.load(std::memory_order_relaxed)) {
    armed_count.fetch_add(1, std::memory_order_relaxed);
    armed_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
}

inline void* raw(std::size_t bytes) {
  void* p = std::malloc(bytes != 0 ? bytes : 1);
  return p;
}

inline void* raw_aligned(std::size_t bytes, std::size_t alignment) {
  // posix_memalign rather than std::aligned_alloc: the latter requires `bytes`
  // to be a multiple of `alignment`, which callers of operator new do not
  // guarantee, and glibc enforces it.
  if (alignment < sizeof(void*)) {
    alignment = sizeof(void*);
  }
#if defined(_WIN32)
  // Windows has no posix_memalign, and _aligned_malloc'd blocks must be
  // released with _aligned_free -- plain free() on one is undefined. The
  // aligned operator delete overloads below route here for that reason.
  return _aligned_malloc(bytes != 0 ? bytes : 1, alignment);
#else
  void* p = nullptr;
  if (posix_memalign(&p, alignment, bytes != 0 ? bytes : 1) != 0) {
    return nullptr;
  }
  return p;
#endif
}

/// Release a block from `raw_aligned`. Separate from plain `free` because the
/// two allocation families are not interchangeable on Windows.
inline void free_aligned(void* p) noexcept {
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

/// Makes `value` observable to the compiler.
///
/// Needed by the positive-control tests: C++14 onwards explicitly permits
/// eliding a new/delete pair whose result is never used, and clang at -O2 does
/// exactly that. Without a sink, the controls allocate nothing, report zero,
/// and look like a broken probe. (`volatile auto* p` does not help -- that
/// declares a pointer *to* volatile, leaving the pointer itself dead.)
template<typename T>
inline void sink(T& value) noexcept {
  asm volatile("" : : "m"(value) : "memory");
}

/// Arm/disarm as an RAII scope so an assertion failure inside the region
/// cannot leave the counter armed for the rest of the binary.
class Scope {
 public:
  Scope() noexcept {
    armed_count.store(0, std::memory_order_relaxed);
    armed_bytes.store(0, std::memory_order_relaxed);
    armed.store(true, std::memory_order_release);
  }
  ~Scope() { armed.store(false, std::memory_order_release); }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

  [[nodiscard]] static std::uint64_t count() noexcept {
    return armed_count.load(std::memory_order_acquire);
  }
  [[nodiscard]] static std::uint64_t bytes() noexcept {
    return armed_bytes.load(std::memory_order_acquire);
  }
};

}  // namespace alloc_probe

// --- throwing forms --------------------------------------------------------
void* operator new(std::size_t n) {
  alloc_probe::note(n);
  void* p = alloc_probe::raw(n);
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}
void* operator new[](std::size_t n) {
  return ::operator new(n);
}

void* operator new(std::size_t n, std::align_val_t a) {
  alloc_probe::note(n);
  void* p = alloc_probe::raw_aligned(n, static_cast<std::size_t>(a));
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}
void* operator new[](std::size_t n, std::align_val_t a) {
  return ::operator new(n, a);
}

// --- nothrow forms ---------------------------------------------------------
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  alloc_probe::note(n);
  return alloc_probe::raw(n);
}
void* operator new[](std::size_t n, const std::nothrow_t& t) noexcept {
  return ::operator new(n, t);
}
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
  alloc_probe::note(n);
  return alloc_probe::raw_aligned(n, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a,
                     const std::nothrow_t& t) noexcept {
  return ::operator new(n, a, t);
}

// --- deletes ---------------------------------------------------------------
// All of these must free with the same primitive family the news used
// (malloc/posix_memalign are both free'd with free).
void operator delete(void* p) noexcept {
  std::free(p);
}
void operator delete[](void* p) noexcept {
  std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
  std::free(p);
}
void operator delete[](void* p, std::size_t) noexcept {
  std::free(p);
}
void operator delete(void* p, const std::nothrow_t&) noexcept {
  std::free(p);
}
void operator delete[](void* p, const std::nothrow_t&) noexcept {
  std::free(p);
}
void operator delete(void* p, std::align_val_t) noexcept {
  alloc_probe::free_aligned(p);
}
void operator delete[](void* p, std::align_val_t) noexcept {
  alloc_probe::free_aligned(p);
}
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
  alloc_probe::free_aligned(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
  alloc_probe::free_aligned(p);
}
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept {
  alloc_probe::free_aligned(p);
}
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept {
  alloc_probe::free_aligned(p);
}

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11
#pragma GCC diagnostic pop
#endif

// ---------------------------------------------------------------------------
// Sanity: the probe itself must work
// ---------------------------------------------------------------------------

FP_TEST(probe_detects_an_allocation) {
  // A positive control. Without this, a probe that silently failed to
  // interpose would make every test below pass for the wrong reason.
  std::uint64_t observed = 0;
  {
    alloc_probe::Scope scope;
    int* leak = new int[256];
    alloc_probe::sink(leak);
    observed = alloc_probe::Scope::count();
    delete[] leak;
  }
  FP_CHECK_MSG(observed >= 1,
               "the counting allocator is not interposing; every zero-alloc "
               "result in this binary would be meaningless");
}

FP_TEST(probe_detects_an_aligned_allocation) {
  // Separate control for the aligned path, which the engine's alignas(128)
  // types actually use. A miss here would let an over-aligned allocation slip
  // past the counter and make every test below pass vacuously.
  struct alignas(256) Over {
    char payload[256];
  };
  std::uint64_t observed = 0;
  {
    alloc_probe::Scope scope;
    Over* p = new Over{};
    alloc_probe::sink(p);
    observed = alloc_probe::Scope::count();
    delete p;
  }
  FP_CHECK_MSG(observed >= 1, "aligned operator new is not interposed");
}

// ---------------------------------------------------------------------------
// Per-primitive
// ---------------------------------------------------------------------------
// Isolated checks first, so a failure in the full-pipeline test below can be
// attributed instead of merely observed.

FP_TEST(spsc_queue_operations_do_not_allocate) {
  auto queue = std::make_unique<SpscQueue<InferenceRequest, 8>>();
  InferenceRequest out{};

  alloc_probe::Scope scope;
  for (int i = 0; i < 1000; ++i) {
    InferenceRequest* slot = queue->write_slot();
    if (slot != nullptr) {
      slot->frame_id = static_cast<std::uint64_t>(i);
      queue->commit();
    }
    (void)queue->try_pop(out);
    (void)queue->size();
    (void)queue->space();
    bool had = false;
    (void)queue->pop_latest(out, had);
  }
  FP_CHECK_EQ(alloc_probe::Scope::count(), std::uint64_t{0});
}

FP_TEST(downsample_does_not_allocate) {
  auto frame = std::make_unique<SourceFrame>();
  frame->data.fill(90);
  auto scratch = std::make_unique<
      std::array<std::uint8_t, pow2_scratch_bytes(kSourceWidth, kSourceHeight)>>();
  auto tensor = std::make_unique<std::array<std::uint8_t, kTensorPixels>>();
  auto floats = std::make_unique<std::array<float, kTensorPixels>>();

  alloc_probe::Scope scope;
  for (int i = 0; i < 20; ++i) {
    const bool ok = downsample_pow2_u8(frame->pixels(), kSourceWidth, kSourceHeight,
                                       SourceFrame::kStride, kDownsampleLog2,
                                       tensor->data(), kTensorWidth, *scratch);
    FP_CHECK(ok);
    normalize_u8_to_f32(*tensor, *floats);
  }
  FP_CHECK_EQ(alloc_probe::Scope::count(), std::uint64_t{0});
}

FP_TEST(result_tracker_does_not_allocate) {
  ResultTracker tracker;
  InferenceResult r{};
  r.feature_count = static_cast<std::uint32_t>(kMaxFeatures);

  alloc_probe::Scope scope;
  for (std::uint64_t i = 1; i <= 2000; ++i) {
    r.frame_id = i;
    r.capture_ns = static_cast<std::int64_t>(i) * 8'333'333;
    r.features[0] = static_cast<float>(i);
    (void)tracker.ingest(r);
    const Estimate est = tracker.estimate_at(from_ns(r.capture_ns + 5'000'000), 33ms);
    (void)est;
  }
  FP_CHECK_EQ(alloc_probe::Scope::count(), std::uint64_t{0});
}

FP_TEST(sample_recorder_does_not_allocate) {
  auto recorder = make_recorder<4096>();

  alloc_probe::Scope scope;
  for (int i = 0; i < 4096; ++i) {
    recorder->record(static_cast<std::int64_t>(i));
  }
  recorder->finalize();
  (void)recorder->percentile(99.9);
  (void)recorder->mean();
  (void)recorder->stddev();
  FP_CHECK_EQ(alloc_probe::Scope::count(), std::uint64_t{0});
}

FP_TEST(worker_step_does_not_allocate) {
  auto pacer = std::make_unique<Pacer>(Pacer::Config{});
  auto frame = std::make_unique<SourceFrame>();
  frame->data.fill(60);

  InferenceWorker::Config wcfg;
  wcfg.min_latency = 0ms;
  wcfg.max_latency = 0ms;  // measure the work, not the simulated latency
  InferenceWorker worker(pacer->requests(), pacer->results(), wcfg);

  alloc_probe::Scope scope;
  for (int i = 0; i < 50; ++i) {
    (void)pacer->submit(*frame, static_cast<std::uint64_t>(i), Clock::now());
    (void)worker.step();
    pacer->ingest_results();
    (void)pacer->estimate_at(Clock::now());
  }
  FP_CHECK_EQ(alloc_probe::Scope::count(), std::uint64_t{0});
}

// ---------------------------------------------------------------------------
// The real thing
// ---------------------------------------------------------------------------

FP_TEST(steady_state_pipeline_does_not_allocate) {
  // Two live threads, the real 120Hz cadence, and the real 15-30ms inference
  // envelope. This is the claim that actually matters.
  auto pacer = std::make_unique<Pacer>(Pacer::Config{});
  auto frame = std::make_unique<SourceFrame>();
  auto recorder = make_recorder<2048>();

  InferenceWorker::Config wcfg;
  wcfg.min_latency = 15ms;
  wcfg.max_latency = 30ms;
  InferenceWorker worker(pacer->requests(), pacer->results(), wcfg);

  std::atomic<bool> stop{false};
  std::thread worker_thread([&] { worker.run(stop); });

  // Warm up *before* arming. Thread startup, the first stdio flush, lazy
  // locale init and any one-time runtime setup all allocate legitimately; none
  // of it is the hot path. Counting it would be a false positive.
  Timestamp deadline = Clock::now() + kFrameBudget120Hz;
  for (int i = 0; i < 40; ++i) {
    frame->data.fill(static_cast<std::uint8_t>(i));
    (void)pacer->step(*frame, static_cast<std::uint64_t>(i), Clock::now(), deadline);
    deadline = pacer->sleep_until_next_frame(deadline);
  }

  const std::uint64_t startup_allocs =
      alloc_probe::total.load(std::memory_order_relaxed);

  constexpr int kFrames = 240;  // ~2s at 120Hz
  std::uint64_t hot_allocs = 0;
  std::uint64_t hot_bytes = 0;
  {
    alloc_probe::Scope scope;
    for (int i = 40; i < 40 + kFrames; ++i) {
      frame->data.fill(static_cast<std::uint8_t>(i & 0xFF));
      const Timestamp capture = Clock::now();
      const Estimate est =
          pacer->step(*frame, static_cast<std::uint64_t>(i), capture, deadline);
      (void)est;
      recorder->record(Clock::now() - capture);
      deadline = pacer->sleep_until_next_frame(deadline);
    }
    hot_allocs = alloc_probe::Scope::count();
    hot_bytes = alloc_probe::Scope::bytes();
  }

  stop.store(true, std::memory_order_release);
  worker_thread.join();

  std::printf(
      "         startup allocations: %llu | hot-path allocations: %llu "
      "(%llu bytes) over %d frames x 2 threads\n",
      static_cast<unsigned long long>(startup_allocs),
      static_cast<unsigned long long>(hot_allocs),
      static_cast<unsigned long long>(hot_bytes), kFrames);

  FP_CHECK_MSG(hot_allocs == 0,
               "hot path allocated " + std::to_string(hot_allocs) + " time(s)");
  FP_CHECK_MSG(pacer->counters().frames >= static_cast<std::uint64_t>(kFrames),
               "the loop did not actually run");
  FP_CHECK_MSG(worker.counters().processed > 0,
               "the worker thread was idle, so its hot path was not covered");
}
