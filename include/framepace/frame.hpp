// FramePace - POD message types that cross the render/inference boundary.
//
// Every type here is trivially copyable and fixed-size. That is a hard
// requirement, not a style preference: these objects live inside the SPSC
// ring's inline storage, so a std::vector or std::string member would put a
// heap allocation directly in the 8.33ms hot path.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "framepace/config.hpp"

namespace framepace {

// ---------------------------------------------------------------------------
// Pipeline geometry
// ---------------------------------------------------------------------------
// Default: 1920x1080 luma in, downsampled by 8 to 240x135 for the model. Factor
// 8 is three exact 2x halvings and divides both axes evenly, so the chain needs
// no padding, no cropping and no odd-size edge case at any intermediate step
// (1920x1080 -> 960x540 -> 480x270 -> 240x135).
//
// Geometry is compile-time, because every buffer on the hot path is a
// fixed-size inline array -- that is what makes the pipeline allocation-free.
// To change it, set the CMake cache variables FRAMEPACE_SOURCE_WIDTH,
// FRAMEPACE_SOURCE_HEIGHT, FRAMEPACE_DOWNSAMPLE_LOG2 and FRAMEPACE_MAX_FEATURES.
// CMake attaches them to the framepace target as PUBLIC definitions, so the
// library and every consumer see the same values. Do NOT define these macros
// by hand in one translation unit: a library and a consumer that disagree on
// the size of InferenceRequest is an ODR violation that corrupts memory
// silently.
#ifndef FRAMEPACE_SOURCE_WIDTH
#define FRAMEPACE_SOURCE_WIDTH 1920
#endif
#ifndef FRAMEPACE_SOURCE_HEIGHT
#define FRAMEPACE_SOURCE_HEIGHT 1080
#endif
#ifndef FRAMEPACE_DOWNSAMPLE_LOG2
#define FRAMEPACE_DOWNSAMPLE_LOG2 3
#endif
#ifndef FRAMEPACE_MAX_FEATURES
#define FRAMEPACE_MAX_FEATURES 32
#endif

inline constexpr std::size_t kSourceWidth = FRAMEPACE_SOURCE_WIDTH;
inline constexpr std::size_t kSourceHeight = FRAMEPACE_SOURCE_HEIGHT;
inline constexpr unsigned kDownsampleLog2 = FRAMEPACE_DOWNSAMPLE_LOG2;

inline constexpr std::size_t kTensorWidth = kSourceWidth >> kDownsampleLog2;
inline constexpr std::size_t kTensorHeight = kSourceHeight >> kDownsampleLog2;
inline constexpr std::size_t kTensorPixels = kTensorWidth * kTensorHeight;

static_assert(kDownsampleLog2 <= 6, "downsample factor is capped at 64");
static_assert(
    (kSourceWidth % (std::size_t{1} << kDownsampleLog2)) == 0,
    "FRAMEPACE_SOURCE_WIDTH must be divisible by 2^FRAMEPACE_DOWNSAMPLE_LOG2");
static_assert(
    (kSourceHeight % (std::size_t{1} << kDownsampleLog2)) == 0,
    "FRAMEPACE_SOURCE_HEIGHT must be divisible by 2^FRAMEPACE_DOWNSAMPLE_LOG2");
static_assert(kTensorWidth >= 1 && kTensorHeight >= 1, "tensor must be non-empty");

/// Floats the model emits per inference (e.g. pose keypoints, flow vectors,
/// exposure weights). Fixed so `InferenceResult` stays a POD.
inline constexpr std::size_t kMaxFeatures = FRAMEPACE_MAX_FEATURES;
static_assert(kMaxFeatures >= 1, "FRAMEPACE_MAX_FEATURES must be at least 1");

// ---------------------------------------------------------------------------
// Image plane
// ---------------------------------------------------------------------------

/// Fixed-size 8-bit single-channel image plane, cache-line aligned so SIMD
/// kernels can use aligned loads on row 0 and so no row shares a line with an
/// unrelated object.
template<std::size_t W, std::size_t H>
struct alignas(kCacheLineSize) Plane8 {
  static constexpr std::size_t kWidth = W;
  static constexpr std::size_t kHeight = H;
  static constexpr std::size_t kStride = W;
  static constexpr std::size_t kBytes = W * H;

  std::array<std::uint8_t, kBytes> data;

  [[nodiscard]] std::uint8_t* row(std::size_t y) noexcept {
    return data.data() + y * kStride;
  }
  [[nodiscard]] const std::uint8_t* row(std::size_t y) const noexcept {
    return data.data() + y * kStride;
  }
  [[nodiscard]] std::uint8_t* pixels() noexcept { return data.data(); }
  [[nodiscard]] const std::uint8_t* pixels() const noexcept { return data.data(); }
};

using SourceFrame = Plane8<kSourceWidth, kSourceHeight>;

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/// Render thread -> inference thread. Carries the already-downsampled tensor,
/// so the expensive SIMD reduction happens on the render thread (bounded, ~50us)
/// rather than handing 2MB of full-resolution luma across the queue.
struct alignas(kCacheLineSize) InferenceRequest {
  std::uint64_t frame_id;
  /// Monotonic ns at which the source frame was captured. The basis for all
  /// latency accounting and for extrapolation.
  std::int64_t capture_ns;
  std::uint32_t width;
  std::uint32_t height;
  std::array<std::uint8_t, kTensorPixels> tensor;
};

/// Inference thread -> render thread.
struct alignas(kCacheLineSize) InferenceResult {
  std::uint64_t frame_id;
  std::int64_t capture_ns;   // copied from the request
  std::int64_t complete_ns;  // when inference finished
  std::uint32_t feature_count;
  std::uint32_t reserved;  // explicit, so padding is never implementation-defined
  std::array<float, kMaxFeatures> features;

  [[nodiscard]] Span<const float> valid_features() const noexcept {
    return Span<const float>{features.data(), feature_count};
  }
};

static_assert(std::is_trivially_copyable_v<InferenceRequest>);
static_assert(std::is_trivially_copyable_v<InferenceResult>);
static_assert(std::is_trivially_copyable_v<SourceFrame>);
static_assert(alignof(InferenceRequest) >= kCacheLineSize);
static_assert(alignof(InferenceResult) >= kCacheLineSize);

// ---------------------------------------------------------------------------
// Queue depths
// ---------------------------------------------------------------------------
// Both are power-of-two and deliberately shallow. Inference runs 15-30ms while
// frames arrive every 8.33ms, so the producer is ~2-4x faster than the consumer
// and a deep queue would only let staleness accumulate: a depth-64 request
// queue means the worker spends half a second chewing through frames the
// renderer has long since presented. Shallow queues drop early instead, which
// is the correct behaviour for a latency pipeline.
inline constexpr std::size_t kRequestQueueDepth = 4;
inline constexpr std::size_t kResultQueueDepth = 8;

}  // namespace framepace
