#include "framepace/downsample.hpp"

#include <cstring>

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
#define FRAMEPACE_USE_NEON 1
#include <arm_neon.h>
#elif defined(__SSSE3__)
#define FRAMEPACE_USE_SSSE3 1
#include <tmmintrin.h>
#endif

namespace framepace {
namespace {

/// Round-half-up average of four bytes. Max sum 1020, so u16 arithmetic is
/// exact and no clamping is needed.
[[nodiscard]] FRAMEPACE_ALWAYS_INLINE std::uint8_t avg4(unsigned a, unsigned b,
                                                        unsigned c,
                                                        unsigned d) noexcept {
  return static_cast<std::uint8_t>((a + b + c + d + 2u) >> 2);
}

/// Scalar remainder for the columns a SIMD body could not fill.
void halve_row_tail(const std::uint8_t* r0, const std::uint8_t* r1, std::uint8_t* out,
                    std::size_t x_begin, std::size_t dst_w) noexcept {
  for (std::size_t x = x_begin; x < dst_w; ++x) {
    out[x] = avg4(r0[2 * x], r0[2 * x + 1], r1[2 * x], r1[2 * x + 1]);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Scalar reference
// ---------------------------------------------------------------------------

void downsample_halve_u8_scalar(const std::uint8_t* src, std::size_t src_w,
                                std::size_t src_h, std::size_t src_stride,
                                std::uint8_t* dst, std::size_t dst_stride) noexcept {
  const std::size_t dst_w = src_w / 2;
  const std::size_t dst_h = src_h / 2;
  for (std::size_t y = 0; y < dst_h; ++y) {
    const std::uint8_t* r0 = src + (2 * y) * src_stride;
    const std::uint8_t* r1 = r0 + src_stride;
    std::uint8_t* out = dst + y * dst_stride;

    // Vectorisation is disabled here on purpose, and this loop is spelled out
    // rather than delegating to halve_row_tail (which the SIMD paths use for
    // their remainders and which *should* stay optimisable).
    //
    // This function has two jobs: be the obviously-correct oracle that
    // test_downsample checks the SIMD kernels against, and be the scalar
    // baseline frame_bench reports a speedup over. At -O3 clang happily
    // auto-vectorises this pattern, which quietly defeats the second job --
    // the measured "SIMD vs scalar" speedup collapses to ~1.2x because both
    // sides are running vector code. Pinning it to scalar makes the reported
    // number mean what it says.
#if defined(__clang__)
#pragma clang loop vectorize(disable) interleave(disable) unroll(disable)
#elif defined(__GNUC__) && __GNUC__ >= 14
// `novector` is GCC 14+. Older GCC rejects it under -Wall -Werror
// (-Wunknown-pragmas), and has no equivalent per-loop control.
#pragma GCC novector
#endif
    for (std::size_t x = 0; x < dst_w; ++x) {
      out[x] = avg4(r0[2 * x], r0[2 * x + 1], r1[2 * x], r1[2 * x + 1]);
    }
  }
}

// ---------------------------------------------------------------------------
// Vectorised halving
// ---------------------------------------------------------------------------

void downsample_halve_u8(const std::uint8_t* src, std::size_t src_w, std::size_t src_h,
                         std::size_t src_stride, std::uint8_t* dst,
                         std::size_t dst_stride) noexcept {
  const std::size_t dst_w = src_w / 2;
  const std::size_t dst_h = src_h / 2;

#if defined(FRAMEPACE_USE_NEON)
  // 16 destination pixels per iteration from 32 source pixels per row.
  //
  // vpaddlq_u8 is the whole trick: it sums *adjacent* byte pairs into 16-bit
  // lanes, which is exactly the horizontal half of a 2x2 box filter, with no
  // shuffle or de-interleave needed. Add the two rows' pair-sums for the
  // vertical half, then vrshrn_n_u16(_, 2) rounds and narrows back to bytes in
  // one instruction.
  const std::size_t vec_w = dst_w & ~std::size_t{15};
  for (std::size_t y = 0; y < dst_h; ++y) {
    const std::uint8_t* r0 = src + (2 * y) * src_stride;
    const std::uint8_t* r1 = r0 + src_stride;
    std::uint8_t* out = dst + y * dst_stride;

    for (std::size_t x = 0; x < vec_w; x += 16) {
      const std::size_t sx = 2 * x;
      const uint16x8_t lo =
          vaddq_u16(vpaddlq_u8(vld1q_u8(r0 + sx)), vpaddlq_u8(vld1q_u8(r1 + sx)));
      const uint16x8_t hi = vaddq_u16(vpaddlq_u8(vld1q_u8(r0 + sx + 16)),
                                      vpaddlq_u8(vld1q_u8(r1 + sx + 16)));
      vst1q_u8(out + x, vcombine_u8(vrshrn_n_u16(lo, 2), vrshrn_n_u16(hi, 2)));
    }
    halve_row_tail(r0, r1, out, vec_w, dst_w);
  }

#elif defined(FRAMEPACE_USE_SSSE3)
  // _mm_maddubs_epi16(v, ones) multiplies unsigned bytes by 1 and adds
  // adjacent pairs into signed 16-bit lanes -- the x86 equivalent of
  // vpaddlq_u8. Max pair sum is 510, well inside int16, so the saturating
  // behaviour never engages. Rounding bias and shift are explicit here because
  // SSE has no rounding-narrow instruction.
  const __m128i ones = _mm_set1_epi8(1);
  const __m128i bias = _mm_set1_epi16(2);
  const std::size_t vec_w = dst_w & ~std::size_t{15};
  for (std::size_t y = 0; y < dst_h; ++y) {
    const std::uint8_t* r0 = src + (2 * y) * src_stride;
    const std::uint8_t* r1 = r0 + src_stride;
    std::uint8_t* out = dst + y * dst_stride;

    for (std::size_t x = 0; x < vec_w; x += 16) {
      const std::size_t sx = 2 * x;
      const __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(r0 + sx));
      const __m128i a1 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(r0 + sx + 16));
      const __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(r1 + sx));
      const __m128i b1 =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(r1 + sx + 16));

      __m128i lo =
          _mm_add_epi16(_mm_maddubs_epi16(a0, ones), _mm_maddubs_epi16(b0, ones));
      __m128i hi =
          _mm_add_epi16(_mm_maddubs_epi16(a1, ones), _mm_maddubs_epi16(b1, ones));
      lo = _mm_srli_epi16(_mm_add_epi16(lo, bias), 2);
      hi = _mm_srli_epi16(_mm_add_epi16(hi, bias), 2);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(out + x), _mm_packus_epi16(lo, hi));
    }
    halve_row_tail(r0, r1, out, vec_w, dst_w);
  }

#else
  downsample_halve_u8_scalar(src, src_w, src_h, src_stride, dst, dst_stride);
#endif
}

// ---------------------------------------------------------------------------
// Dispatch introspection
// ---------------------------------------------------------------------------

SimdKernel active_kernel() noexcept {
#if defined(FRAMEPACE_USE_NEON)
  return SimdKernel::kNeon;
#elif defined(FRAMEPACE_USE_SSSE3)
  return SimdKernel::kSsse3;
#else
  return SimdKernel::kScalar;
#endif
}

const char* kernel_name(SimdKernel k) noexcept {
  switch (k) {
    case SimdKernel::kNeon:
      return "NEON (vpaddlq_u8 + vrshrn_n_u16)";
    case SimdKernel::kSsse3:
      return "SSSE3 (maddubs + packus)";
    case SimdKernel::kScalar:
      break;
  }
  return "scalar";
}

// ---------------------------------------------------------------------------
// Power-of-two chain
// ---------------------------------------------------------------------------

bool downsample_pow2_u8(const std::uint8_t* src, std::size_t src_w, std::size_t src_h,
                        std::size_t src_stride, unsigned log2_factor, std::uint8_t* dst,
                        std::size_t dst_stride, Span<std::uint8_t> scratch) noexcept {
  if (src == nullptr || dst == nullptr) {
    return false;
  }
  if (log2_factor == 0) {
    // Straight copy, honouring both strides.
    for (std::size_t y = 0; y < src_h; ++y) {
      std::memcpy(dst + y * dst_stride, src + y * src_stride, src_w);
    }
    return true;
  }
  const std::size_t factor = std::size_t{1} << log2_factor;
  if ((src_w % factor) != 0 || (src_h % factor) != 0) {
    return false;
  }
  if (log2_factor > 1 && scratch.size() < pow2_scratch_bytes(src_w, src_h)) {
    return false;
  }

  // Ping-pong halves of the scratch span. Each is sized for level 1
  // (src_w/2 x src_h/2), which bounds every later level.
  const std::size_t half = log2_factor > 1 ? scratch.size() / 2 : 0;
  std::uint8_t* buffers[2] = {
      log2_factor > 1 ? scratch.data() : nullptr,
      log2_factor > 1 ? scratch.data() + half : nullptr,
  };

  const std::uint8_t* cur = src;
  std::size_t cur_w = src_w;
  std::size_t cur_h = src_h;
  std::size_t cur_stride = src_stride;

  for (unsigned level = 0; level < log2_factor; ++level) {
    if ((cur_w % 2) != 0 || (cur_h % 2) != 0) {
      return false;  // unreachable given the divisibility check above
    }
    const std::size_t next_w = cur_w / 2;
    const std::size_t next_h = cur_h / 2;
    const bool last = (level + 1 == log2_factor);

    // The final level writes straight into the caller's buffer, so the
    // tensor is never copied an extra time.
    std::uint8_t* out = last ? dst : buffers[level % 2];
    const std::size_t out_stride = last ? dst_stride : next_w;

    if (out == nullptr) {
      return false;
    }
    if (!last && half < next_w * next_h) {
      return false;
    }

    downsample_halve_u8(cur, cur_w, cur_h, cur_stride, out, out_stride);

    cur = out;
    cur_w = next_w;
    cur_h = next_h;
    cur_stride = out_stride;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Normalisation
// ---------------------------------------------------------------------------

void normalize_u8_to_f32(Span<const std::uint8_t> src, Span<float> dst, float scale,
                         float bias) noexcept {
  const std::size_t n = src.size() < dst.size() ? src.size() : dst.size();
  std::size_t i = 0;

#if defined(FRAMEPACE_USE_NEON)
  const float32x4_t vscale = vdupq_n_f32(scale);
  const float32x4_t vbias = vdupq_n_f32(bias);
  for (; i + 16 <= n; i += 16) {
    const uint8x16_t raw = vld1q_u8(src.data() + i);
    const uint16x8_t w0 = vmovl_u8(vget_low_u8(raw));
    const uint16x8_t w1 = vmovl_u8(vget_high_u8(raw));
    const uint32x4_t q[4] = {
        vmovl_u16(vget_low_u16(w0)),
        vmovl_u16(vget_high_u16(w0)),
        vmovl_u16(vget_low_u16(w1)),
        vmovl_u16(vget_high_u16(w1)),
    };
    for (std::size_t k = 0; k < 4; ++k) {
      vst1q_f32(dst.data() + i + 4 * k, vfmaq_f32(vbias, vcvtq_f32_u32(q[k]), vscale));
    }
  }
#endif

  for (; i < n; ++i) {
    dst[i] = static_cast<float>(src[i]) * scale + bias;
  }
}

}  // namespace framepace
