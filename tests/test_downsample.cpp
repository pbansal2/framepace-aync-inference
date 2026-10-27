// FramePace - SIMD downsampling correctness.
//
// The central claim is bit-exact equivalence between the SIMD kernel and the
// scalar reference. Without it, the model receives different input depending on
// which ISA the build targeted, and a bug reproducible on x86 would vanish on
// arm64.

#include "framepace/downsample.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <vector>

#include "framepace/frame.hpp"
#include "framepace/inference_worker.hpp"
#include "test_util.hpp"

using namespace framepace;

namespace {

std::vector<std::uint8_t> random_image(std::size_t w, std::size_t h,
                                       std::uint64_t seed) {
  Xorshift64 rng(seed);
  std::vector<std::uint8_t> img(w * h);
  for (auto& px : img) {
    px = static_cast<std::uint8_t>(rng.next() & 0xFF);
  }
  return img;
}

/// Exact area average over a factor x factor block, in integer arithmetic with
/// round-half-up. The ground truth for the deviation bound below.
std::vector<std::uint8_t> exact_box_downsample(const std::vector<std::uint8_t>& src,
                                               std::size_t w, std::size_t h,
                                               std::size_t factor) {
  const std::size_t dw = w / factor;
  const std::size_t dh = h / factor;
  const std::size_t area = factor * factor;
  std::vector<std::uint8_t> dst(dw * dh);
  for (std::size_t y = 0; y < dh; ++y) {
    for (std::size_t x = 0; x < dw; ++x) {
      std::uint32_t sum = 0;
      for (std::size_t by = 0; by < factor; ++by) {
        for (std::size_t bx = 0; bx < factor; ++bx) {
          sum += src[(y * factor + by) * w + (x * factor + bx)];
        }
      }
      dst[y * dw + x] = static_cast<std::uint8_t>((sum + area / 2) / area);
    }
  }
  return dst;
}

}  // namespace

// ---------------------------------------------------------------------------
// Known values
// ---------------------------------------------------------------------------

FP_TEST(halve_computes_rounded_box_average) {
  // 4x2 source -> 2x1 destination.
  //   block 0: 10 20 / 30 40 -> (100 + 2) / 4 = 25  (exact 25.0)
  //   block 1: 50 60 / 70 81 -> (261 + 2) / 4 = 65  (exact 65.25 -> 65)
  const std::uint8_t src[8] = {10, 20, 50, 60, 30, 40, 70, 81};
  std::uint8_t dst[2] = {0, 0};

  downsample_halve_u8_scalar(src, 4, 2, 4, dst, 2);
  FP_CHECK_EQ(int{dst[0]}, 25);
  FP_CHECK_EQ(int{dst[1]}, 65);

  std::uint8_t simd_dst[2] = {0, 0};
  downsample_halve_u8(src, 4, 2, 4, simd_dst, 2);
  FP_CHECK_EQ(int{simd_dst[0]}, 25);
  FP_CHECK_EQ(int{simd_dst[1]}, 65);
}

FP_TEST(halve_rounds_half_up) {
  // Sum 2 -> (2 + 2) / 4 = 1, i.e. 0.5 rounds up rather than truncating to 0.
  const std::uint8_t src[4] = {1, 1, 0, 0};
  std::uint8_t dst = 0;
  downsample_halve_u8_scalar(src, 2, 2, 2, &dst, 1);
  FP_CHECK_EQ(int{dst}, 1);
}

FP_TEST(halve_saturates_at_max_without_overflow) {
  // Sum 1020 is the worst case for the u16 intermediate; must give 255, not
  // wrap or clamp low.
  std::vector<std::uint8_t> src(4, 255);
  std::uint8_t dst = 0;
  downsample_halve_u8(src.data(), 2, 2, 2, &dst, 1);
  FP_CHECK_EQ(int{dst}, 255);
}

// ---------------------------------------------------------------------------
// SIMD / scalar equivalence
// ---------------------------------------------------------------------------

FP_TEST(simd_matches_scalar_bit_exactly) {
  // Widths chosen around the 16-pixel vector body: below it (pure tail),
  // exactly on it, just past it (body + tail), and the real pipeline sizes.
  const std::size_t widths[] = {2,  4,  8,   16,  30,  32,  34,  48,
                                62, 64, 100, 240, 480, 960, 1920};
  const std::size_t heights[] = {2, 4, 6, 16, 34, 135 * 2, 270, 540, 1080};

  std::size_t cases = 0;
  for (std::size_t w : widths) {
    for (std::size_t h : heights) {
      if (w * h > 1920 * 1080) {
        continue;
      }
      const auto src = random_image(w, h, 0x5EED'0000ULL + w * 131 + h);
      std::vector<std::uint8_t> a((w / 2) * (h / 2), 0xAA);
      std::vector<std::uint8_t> b((w / 2) * (h / 2), 0x55);

      downsample_halve_u8_scalar(src.data(), w, h, w, a.data(), w / 2);
      downsample_halve_u8(src.data(), w, h, w, b.data(), w / 2);

      const bool equal = (a == b);
      FP_CHECK_MSG(equal, "scalar/SIMD mismatch at " + std::to_string(w) + "x" +
                              std::to_string(h));
      if (!equal) {
        for (std::size_t i = 0; i < a.size(); ++i) {
          if (a[i] != b[i]) {
            std::fprintf(stderr, "      first diff at %zu: scalar=%d simd=%d\n", i,
                         int{a[i]}, int{b[i]});
            break;
          }
        }
      }
      ++cases;
    }
  }
  std::printf("         compared %zu geometries against kernel: %s\n", cases,
              kernel_name(active_kernel()));
}

FP_TEST(simd_matches_scalar_with_padded_stride) {
  // Stride > width: rows are not contiguous, so the kernel must honour the
  // stride rather than walking straight off the end of row 0.
  constexpr std::size_t w = 34;
  constexpr std::size_t h = 10;
  constexpr std::size_t stride = 64;
  const auto src = random_image(stride, h, 0xABCDEF);

  std::vector<std::uint8_t> a((w / 2) * (h / 2), 0);
  std::vector<std::uint8_t> b((w / 2) * (h / 2), 0);
  downsample_halve_u8_scalar(src.data(), w, h, stride, a.data(), w / 2);
  downsample_halve_u8(src.data(), w, h, stride, b.data(), w / 2);
  FP_CHECK(a == b);

  // And a destination stride wider than the destination width.
  constexpr std::size_t dst_stride = 40;
  std::vector<std::uint8_t> c(dst_stride * (h / 2), 0x11);
  std::vector<std::uint8_t> d(dst_stride * (h / 2), 0x11);
  downsample_halve_u8_scalar(src.data(), w, h, stride, c.data(), dst_stride);
  downsample_halve_u8(src.data(), w, h, stride, d.data(), dst_stride);
  FP_CHECK(c == d);
}

// ---------------------------------------------------------------------------
// Power-of-two chain
// ---------------------------------------------------------------------------

FP_TEST(pow2_chain_matches_manual_halving) {
  constexpr std::size_t w = 240;
  constexpr std::size_t h = 80;
  const auto src = random_image(w, h, 0x1234'5678);

  // Reference: three explicit halvings.
  std::vector<std::uint8_t> l1((w / 2) * (h / 2));
  std::vector<std::uint8_t> l2((w / 4) * (h / 4));
  std::vector<std::uint8_t> l3((w / 8) * (h / 8));
  downsample_halve_u8_scalar(src.data(), w, h, w, l1.data(), w / 2);
  downsample_halve_u8_scalar(l1.data(), w / 2, h / 2, w / 2, l2.data(), w / 4);
  downsample_halve_u8_scalar(l2.data(), w / 4, h / 4, w / 4, l3.data(), w / 8);

  std::vector<std::uint8_t> scratch(pow2_scratch_bytes(w, h));
  std::vector<std::uint8_t> out((w / 8) * (h / 8), 0);
  FP_REQUIRE(downsample_pow2_u8(src.data(), w, h, w, 3, out.data(), w / 8, scratch));
  FP_CHECK(out == l3);
}

FP_TEST(pow2_factor_one_is_identity_copy) {
  constexpr std::size_t w = 16;
  constexpr std::size_t h = 4;
  const auto src = random_image(w, h, 7);
  std::vector<std::uint8_t> out(w * h, 0);
  FP_REQUIRE(downsample_pow2_u8(src.data(), w, h, w, 0, out.data(), w, {}));
  FP_CHECK(out == src);
}

FP_TEST(pow2_factor_two_needs_no_scratch) {
  // One level writes straight to the destination, so an empty scratch span
  // must still succeed.
  constexpr std::size_t w = 32;
  constexpr std::size_t h = 8;
  const auto src = random_image(w, h, 11);
  std::vector<std::uint8_t> out((w / 2) * (h / 2), 0);
  FP_REQUIRE(downsample_pow2_u8(src.data(), w, h, w, 1, out.data(), w / 2, {}));

  std::vector<std::uint8_t> expect((w / 2) * (h / 2));
  downsample_halve_u8_scalar(src.data(), w, h, w, expect.data(), w / 2);
  FP_CHECK(out == expect);
}

FP_TEST(pow2_rejects_bad_arguments) {
  constexpr std::size_t w = 240;
  constexpr std::size_t h = 80;
  const auto src = random_image(w, h, 3);
  std::vector<std::uint8_t> out(w * h, 0);
  std::vector<std::uint8_t> scratch(pow2_scratch_bytes(w, h));

  // Scratch too small for a 3-level chain.
  std::vector<std::uint8_t> tiny(8);
  FP_CHECK(!downsample_pow2_u8(src.data(), w, h, w, 3, out.data(), w / 8, tiny));
  // Height 80 is not divisible by 32.
  FP_CHECK(!downsample_pow2_u8(src.data(), w, h, w, 5, out.data(), w / 32, scratch));
  // Null pointers.
  FP_CHECK(!downsample_pow2_u8(nullptr, w, h, w, 3, out.data(), w / 8, scratch));
  FP_CHECK(!downsample_pow2_u8(src.data(), w, h, w, 3, nullptr, w / 8, scratch));
}

FP_TEST(pow2_chain_tracks_exact_area_average) {
  // Chained halving is not identical to one wide box filter: each level's
  // round-half-up introduces at most 1/2 LSB, and those errors compound through
  // the chain. Three levels can therefore differ from an exact 8x8 area average
  // by a small amount. Bound it rather than asserting equality -- and bound it
  // tightly enough that a real kernel bug cannot hide inside the tolerance.
  // Sized locally, deliberately not from kTensorWidth/kTensorPixels: this tests
  // the kernel math at a fixed 1080p input, independent of the configured
  // pipeline geometry. Using the pipeline constants here wrote past the end of
  // `chained` whenever FRAMEPACE_SOURCE_WIDTH/HEIGHT were set to anything else.
  constexpr std::size_t w = 1920;
  constexpr std::size_t h = 1080;
  constexpr std::size_t out_w = w / 8;
  constexpr std::size_t out_h = h / 8;
  const auto src = random_image(w, h, 0xFEED'FACE);
  const auto exact = exact_box_downsample(src, w, h, 8);

  std::vector<std::uint8_t> scratch(pow2_scratch_bytes(w, h));
  std::vector<std::uint8_t> chained(out_w * out_h, 0);
  FP_REQUIRE(
      downsample_pow2_u8(src.data(), w, h, w, 3, chained.data(), out_w, scratch));
  FP_REQUIRE(exact.size() == chained.size());

  int worst = 0;
  double total = 0.0;
  for (std::size_t i = 0; i < exact.size(); ++i) {
    const int diff = std::abs(int{exact[i]} - int{chained[i]});
    worst = std::max(worst, diff);
    total += diff;
  }
  const double mean_abs = total / static_cast<double>(exact.size());
  std::printf("         chained vs exact 8x8 box: max |diff| = %d, mean = %.4f\n",
              worst, mean_abs);
  FP_CHECK_MSG(worst <= 2, "max deviation " + std::to_string(worst));
  FP_CHECK_MSG(mean_abs < 0.5, "mean deviation " + std::to_string(mean_abs));
}

FP_TEST(pipeline_geometry_downsamples_cleanly) {
  auto frame = std::make_unique<SourceFrame>();
  Xorshift64 rng(99);
  for (auto& px : frame->data) {
    px = static_cast<std::uint8_t>(rng.next() & 0xFF);
  }
  std::vector<std::uint8_t> scratch(pow2_scratch_bytes(kSourceWidth, kSourceHeight));
  std::vector<std::uint8_t> tensor(kTensorPixels, 0);

  FP_REQUIRE(downsample_pow2_u8(frame->pixels(), kSourceWidth, kSourceHeight,
                                SourceFrame::kStride, kDownsampleLog2, tensor.data(),
                                kTensorWidth, scratch));
  // A uniform-random source averages to ~127.5; every output must be a real
  // average, so nothing should sit at an untouched sentinel.
  const auto [lo, hi] = std::minmax_element(tensor.begin(), tensor.end());
  FP_CHECK(*lo > 0);
  FP_CHECK(*hi < 255);
}

// ---------------------------------------------------------------------------
// Normalisation
// ---------------------------------------------------------------------------

FP_TEST(normalize_scales_and_biases) {
  std::vector<std::uint8_t> src(64);
  for (std::size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<std::uint8_t>(i * 4);
  }
  std::vector<float> dst(src.size(), -1.0F);
  normalize_u8_to_f32(src, dst);
  for (std::size_t i = 0; i < src.size(); ++i) {
    FP_CHECK_NEAR(dst[i], static_cast<double>(src[i]) / 255.0, 1e-6);
  }

  // Non-default scale/bias, and a size that is not a multiple of the 16-wide
  // vector body so the scalar tail runs too.
  std::vector<std::uint8_t> odd(37, 128);
  std::vector<float> odd_dst(odd.size(), 0.0F);
  normalize_u8_to_f32(odd, odd_dst, 2.0F, -1.0F);
  for (float v : odd_dst) {
    FP_CHECK_NEAR(v, 128.0 * 2.0 - 1.0, 1e-4);
  }
}

FP_TEST(normalize_respects_shorter_span) {
  std::vector<std::uint8_t> src(100, 255);
  std::vector<float> dst(10, 0.0F);
  normalize_u8_to_f32(src, dst);
  for (float v : dst) {
    FP_CHECK_NEAR(v, 1.0, 1e-6);
  }
}
