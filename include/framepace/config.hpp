// FramePace - compile-time configuration and platform shims.
//
// Nothing in this header allocates, throws, or has a non-trivial dynamic
// initializer: it is safe to include from any hot-path translation unit.
#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

namespace framepace {

// ---------------------------------------------------------------------------
// Cache geometry
// ---------------------------------------------------------------------------
// Architecture detection comes *first*, deliberately. libc++ reports
// std::hardware_destructive_interference_size == 64 on arm64, but Apple silicon
// uses 128-byte cache lines (confirm with `sysctl -n hw.cachelinesize`). Using
// the library value there would leave two "padded" atomics sharing one line and
// silently reintroduce the false sharing this engine exists to avoid.
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
inline constexpr std::size_t kCacheLineSize = 128;
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
// 64-byte lines, but the adjacent-line prefetcher pulls pairs; 128 is the
// safe stride for "never share a line with the other thread".
inline constexpr std::size_t kCacheLineSize = 128;
#elif defined(__cpp_lib_hardware_interference_size)
inline constexpr std::size_t kCacheLineSize =
    std::hardware_destructive_interference_size;
#else
inline constexpr std::size_t kCacheLineSize = 64;
#endif

static_assert((kCacheLineSize & (kCacheLineSize - 1)) == 0,
              "cache line size must be a power of two");

// ---------------------------------------------------------------------------
// Sanitizer / build-mode introspection
// ---------------------------------------------------------------------------
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define FRAMEPACE_THREAD_SANITIZER 1
#endif
#if __has_feature(address_sanitizer)
#define FRAMEPACE_ADDRESS_SANITIZER 1
#endif
#endif
#if defined(__SANITIZE_THREAD__)
#define FRAMEPACE_THREAD_SANITIZER 1
#endif
#if defined(__SANITIZE_ADDRESS__)
#define FRAMEPACE_ADDRESS_SANITIZER 1
#endif

#if defined(FRAMEPACE_THREAD_SANITIZER)
inline constexpr bool kThreadSanitizerEnabled = true;
#else
inline constexpr bool kThreadSanitizerEnabled = false;
#endif

#if defined(FRAMEPACE_ADDRESS_SANITIZER)
inline constexpr bool kAddressSanitizerEnabled = true;
#else
inline constexpr bool kAddressSanitizerEnabled = false;
#endif

inline constexpr bool kAnySanitizerEnabled =
    kThreadSanitizerEnabled || kAddressSanitizerEnabled;

// ---------------------------------------------------------------------------
// Inlining and branch hints
// ---------------------------------------------------------------------------
#if defined(_MSC_VER)
#define FRAMEPACE_ALWAYS_INLINE __forceinline
#else
#define FRAMEPACE_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

// The C++20 [[likely]] / [[unlikely]] attributes are not available in C++17,
// and an unknown attribute is a hard error under -Werror -Wpedantic. These
// macros carry the same intent to the optimiser on every compiler we target.
#if defined(__GNUC__) || defined(__clang__)
#define FRAMEPACE_LIKELY(x) (__builtin_expect(!!(x), 1))
#define FRAMEPACE_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#define FRAMEPACE_LIKELY(x) (x)
#define FRAMEPACE_UNLIKELY(x) (x)
#endif

// ---------------------------------------------------------------------------
// Span
// ---------------------------------------------------------------------------

/// Non-owning view over contiguous memory: a minimal stand-in for `std::span`,
/// which is C++20 while this engine targets C++17.
///
/// Carries only what the kernels use -- `data()`, `size()`, and indexing -- so
/// there is no iterator, subspan, or static-extent machinery to get wrong. It
/// holds a pointer and a length and never allocates, which is what lets the
/// downsample chain take its scratch buffer as one argument while still being
/// able to reject a buffer that is too small.
template<typename T>
class Span {
 public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;

  constexpr Span() noexcept = default;
  constexpr Span(T* ptr, std::size_t count) noexcept : data_(ptr), size_(count) {}

  /// Bind to any contiguous container (`std::array`, `std::vector`, another
  /// `Span`). Constrained on `data()` being pointer-convertible, so unrelated
  /// types do not silently match. Takes an lvalue reference specifically so a
  /// temporary container cannot bind and leave the view dangling.
  template<typename Container,
           typename = std::enable_if_t<
               !std::is_same_v<std::remove_cv_t<Container>, Span> &&
               std::is_convertible_v<decltype(std::declval<Container&>().data()), T*>>>
  constexpr Span(Container& c) noexcept  // NOLINT(google-explicit-constructor)
      : data_(c.data()), size_(c.size()) {}

  [[nodiscard]] constexpr T* data() const noexcept { return data_; }
  [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
  [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] constexpr T& operator[](std::size_t i) const noexcept {
    return data_[i];
  }
  [[nodiscard]] constexpr T* begin() const noexcept { return data_; }
  [[nodiscard]] constexpr T* end() const noexcept { return data_ + size_; }

 private:
  T* data_{nullptr};
  std::size_t size_{0};
};

}  // namespace framepace
