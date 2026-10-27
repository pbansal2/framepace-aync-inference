// FramePace - SIMD box downsampling kernels.
#pragma once

#include <cstddef>
#include <cstdint>

#include "framepace/config.hpp"

namespace framepace {

/// Which halving kernel this build compiled to.
enum class SimdKernel : std::uint8_t { kScalar, kNeon, kSsse3 };

[[nodiscard]] SimdKernel active_kernel() noexcept;
[[nodiscard]] const char* kernel_name(SimdKernel k) noexcept;

// ---------------------------------------------------------------------------
// 2x2 box halving
// ---------------------------------------------------------------------------
// dst[x,y] = (src[2x,2y] + src[2x+1,2y] + src[2x,2y+1] + src[2x+1,2y+1] + 2) / 4
//
// The `+ 2` is round-half-up. It matters that scalar and SIMD agree *bit for
// bit* -- otherwise a build without SIMD produces a different model input and
// the engine's output silently depends on the compiler flags. NEON gets this
// for free via `vrshrn_n_u16` (rounding narrowing shift); the scalar and SSSE3
// paths add the bias explicitly. `test_downsample` asserts the equivalence.
//
// Requires: `src_w` and `src_h` even, no overlap between src and dst.

/// Portable reference implementation. Always available, also the cross-check
/// oracle for the SIMD paths.
void downsample_halve_u8_scalar(const std::uint8_t* src, std::size_t src_w,
                                std::size_t src_h, std::size_t src_stride,
                                std::uint8_t* dst, std::size_t dst_stride) noexcept;

/// Best kernel for this target. Falls back to the scalar body verbatim when no
/// SIMD ISA is available, so callers never need to branch.
void downsample_halve_u8(const std::uint8_t* src, std::size_t src_w, std::size_t src_h,
                         std::size_t src_stride, std::uint8_t* dst,
                         std::size_t dst_stride) noexcept;

// ---------------------------------------------------------------------------
// Power-of-two chain
// ---------------------------------------------------------------------------

/// Scratch bytes `downsample_pow2_u8` needs for a `src_w` x `src_h` input.
///
/// Two ping-pong buffers of one quarter the source area. SIMD kernels must not
/// alias their input and output, so a single in-place buffer is not an option;
/// two is the minimum and sizes shrink monotonically down the chain, so the
/// first level's dimensions bound every later one.
[[nodiscard]] constexpr std::size_t pow2_scratch_bytes(std::size_t src_w,
                                                       std::size_t src_h) noexcept {
  return 2 * (src_w / 2) * (src_h / 2);
}

/// Downsample by `1 << log2_factor` via repeated 2x2 halving.
///
/// Chained halving rather than one wide box filter: a single 8x8 pass would do
/// ~25% less arithmetic (1 + 1/4 + 1/16 vs 1 unit of work), but repeated
/// halving reuses one tiny verified kernel, keeps every access pattern
/// contiguous, and costs well under the frame budget. The two differ slightly
/// in rounding -- chained halving is a true box average only when each level's
/// rounding error cancels -- so `test_downsample` bounds the deviation from an
/// exact area average rather than asserting equality.
///
/// Returns false (and writes nothing) if the arguments are inconsistent:
/// scratch too small, dimensions not divisible by the factor, or an
/// intermediate level with an odd dimension.
///
/// Allocation-free: all working memory comes from `scratch`.
[[nodiscard]] bool downsample_pow2_u8(const std::uint8_t* src, std::size_t src_w,
                                      std::size_t src_h, std::size_t src_stride,
                                      unsigned log2_factor, std::uint8_t* dst,
                                      std::size_t dst_stride,
                                      Span<std::uint8_t> scratch) noexcept;

// ---------------------------------------------------------------------------
// Tensor normalisation
// ---------------------------------------------------------------------------

/// `dst[i] = src[i] * scale + bias`, vectorised. Typical model preprocessing is
/// scale = 1/255, bias = 0. Sizes must match.
void normalize_u8_to_f32(Span<const std::uint8_t> src, Span<float> dst,
                         float scale = 1.0F / 255.0F, float bias = 0.0F) noexcept;

}  // namespace framepace
