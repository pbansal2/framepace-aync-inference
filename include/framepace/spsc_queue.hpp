// FramePace - cache-aligned, wait-free single-producer / single-consumer queue.
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "framepace/config.hpp"

namespace framepace {

/// Bounded wait-free SPSC ring buffer.
///
/// Exactly one thread may call the producer half (`try_push`, `write_slot`,
/// `commit`) and exactly one *other* thread the consumer half (`try_pop`,
/// `read_slot`, `release`). Both halves are wait-free: every operation
/// completes in a bounded number of steps with no locks, no CAS retry loop,
/// and no syscall. A full queue fails the push rather than blocking the
/// producer, which is the property the 120Hz render thread actually needs --
/// it must never stall behind a 15-30ms inference.
///
/// Storage is an inline `std::array`, so the queue performs **no heap
/// allocation at any point**, construction included. Large instances should be
/// created once at startup (`std::make_unique<Queue>()`) rather than on a
/// thread stack; that single startup allocation is outside the hot path.
///
/// Design notes
/// ------------
/// * Monotonically increasing 64-bit head/tail counters, masked only on index.
///   Unlike the "sacrifice one slot" variant this uses all `Capacity` slots and
///   makes `size()` a single subtraction. At 120 FPS a uint64 counter wraps
///   after ~4.9 billion years; unsigned wrap is well-defined regardless.
/// * Four separate cache lines: producer-written state, consumer-written state,
///   the storage array, and trailing padding. Without this split, every push
///   invalidates the line the consumer is polling and the ring degenerates into
///   a cache-line ping-pong that costs more than a mutex.
/// * Each side caches its view of the other's counter and only re-reads the
///   shared atomic when its cached value says the queue is full/empty. In
///   steady state with a non-empty, non-full queue, neither thread touches the
///   other's cache line at all.
///
/// Correctness rests on one release/acquire pair per direction: the producer's
/// `tail_.store(release)` publishes the slot write, and the consumer's
/// `tail_.load(acquire)` consumes it. The payload itself is deliberately a
/// plain (non-atomic) object -- the ordering edge through the counters is what
/// makes that race-free, and ThreadSanitizer models exactly this.
template<typename T, std::size_t Capacity>
class alignas(kCacheLineSize) SpscQueue {
 public:
  using value_type = T;

  static_assert(Capacity >= 2, "capacity must be at least 2");
  static_assert((Capacity & (Capacity - 1)) == 0,
                "capacity must be a power of two so index masking is a single AND");
  static_assert(std::is_trivially_copyable_v<T>,
                "SpscQueue carries POD frame/tensor messages: a trivially "
                "copyable payload keeps slot reuse free of lifetime bookkeeping "
                "and keeps the in-place write_slot()/commit() API sound");
  static_assert(std::is_default_constructible_v<T>,
                "slots are value-initialised once at construction");

  static constexpr std::size_t kCapacity = Capacity;

  SpscQueue() = default;

  SpscQueue(const SpscQueue&) = delete;
  SpscQueue& operator=(const SpscQueue&) = delete;
  SpscQueue(SpscQueue&&) = delete;
  SpscQueue& operator=(SpscQueue&&) = delete;

  // -------------------------------------------------------------------------
  // Producer side -- one thread only
  // -------------------------------------------------------------------------

  /// Copy `value` into the ring. Returns false if full; never blocks.
  [[nodiscard]] bool try_push(const T& value) noexcept {
    const std::uint64_t tail = tail_.load(std::memory_order_relaxed);
    if (FRAMEPACE_UNLIKELY(!producer_has_space(tail))) {
      return false;
    }
    slots_[tail & kMask] = value;
    tail_.store(tail + 1, std::memory_order_release);
    return true;
  }

  /// Construct the payload in place from `args`, avoiding a staging copy.
  template<typename... Args>
  [[nodiscard]] bool try_emplace(Args&&... args) noexcept {
    const std::uint64_t tail = tail_.load(std::memory_order_relaxed);
    if (FRAMEPACE_UNLIKELY(!producer_has_space(tail))) {
      return false;
    }
    slots_[tail & kMask] = T{std::forward<Args>(args)...};
    tail_.store(tail + 1, std::memory_order_release);
    return true;
  }

  /// Zero-copy production, half one: borrow the next writable slot, or nullptr
  /// if the queue is full.
  ///
  /// Lets the producer build a large message (a 32KB tensor, say) directly in
  /// the ring instead of filling a stack temporary and memcpy'ing it in. The
  /// returned reference stays valid until `commit()`, and the slot is not
  /// visible to the consumer until then.
  [[nodiscard]] T* write_slot() noexcept {
    const std::uint64_t tail = tail_.load(std::memory_order_relaxed);
    if (FRAMEPACE_UNLIKELY(!producer_has_space(tail))) {
      return nullptr;
    }
    return &slots_[tail & kMask];
  }

  /// Zero-copy production, half two: publish the slot returned by
  /// `write_slot()`. Must be called exactly once per successful `write_slot()`,
  /// and only after the payload is fully written.
  void commit() noexcept {
    const std::uint64_t tail = tail_.load(std::memory_order_relaxed);
    tail_.store(tail + 1, std::memory_order_release);
  }

  /// Slots currently writable without blocking. Producer-side estimate: a
  /// concurrent pop can only make this larger, never smaller.
  [[nodiscard]] std::size_t space() noexcept {
    const std::uint64_t tail = tail_.load(std::memory_order_relaxed);
    cached_head_ = head_.load(std::memory_order_acquire);
    return Capacity - static_cast<std::size_t>(tail - cached_head_);
  }

  // -------------------------------------------------------------------------
  // Consumer side -- one thread only
  // -------------------------------------------------------------------------

  /// Move the oldest element into `out`. Returns false if empty; never blocks.
  [[nodiscard]] bool try_pop(T& out) noexcept {
    const std::uint64_t head = head_.load(std::memory_order_relaxed);
    if (FRAMEPACE_UNLIKELY(!consumer_has_data(head))) {
      return false;
    }
    out = slots_[head & kMask];
    head_.store(head + 1, std::memory_order_release);
    return true;
  }

  /// Zero-copy consumption, half one: borrow the oldest element in place, or
  /// nullptr if empty. Valid until `release()`.
  [[nodiscard]] const T* read_slot() noexcept {
    const std::uint64_t head = head_.load(std::memory_order_relaxed);
    if (FRAMEPACE_UNLIKELY(!consumer_has_data(head))) {
      return nullptr;
    }
    return &slots_[head & kMask];
  }

  /// Zero-copy consumption, half two: return the borrowed slot to the
  /// producer. Exactly once per successful `read_slot()`.
  void release() noexcept {
    const std::uint64_t head = head_.load(std::memory_order_relaxed);
    head_.store(head + 1, std::memory_order_release);
  }

  /// Discard all but the newest element, writing it to `out`.
  ///
  /// This is the frame pacer's ingest primitive. When inference falls behind,
  /// stale results are worse than useless -- the renderer wants the freshest
  /// estimate, not a backlog it must replay. Returns the number of elements
  /// dropped, and false in `had_value` if the queue was empty.
  [[nodiscard]] std::size_t pop_latest(T& out, bool& had_value) noexcept {
    std::size_t dropped = 0;
    had_value = false;
    while (try_pop(out)) {
      dropped += had_value ? 1 : 0;
      had_value = true;
    }
    return dropped;
  }

  /// Zero-copy latest-wins consumption: skip to the newest committed element
  /// and borrow it in place, reporting how many stale ones were discarded.
  /// Returns nullptr if empty. Pair with `release()` exactly as `read_slot()`.
  ///
  /// The in-place counterpart to `pop_latest`, and the primitive the inference
  /// worker needs. A FIFO read would make the worker always pick the *oldest*
  /// queued frame, so a depth-N queue would add N inference times of staleness
  /// to every result -- the opposite of what a latency pipeline wants. Skipping
  /// to the newest keeps end-to-end latency at roughly one inference time
  /// regardless of queue depth.
  ///
  /// Advancing `head_` before the payload read is safe: it frees only the
  /// slots *before* the one being returned, so the producer can refill the
  /// stale slots while we work, and the borrowed slot stays ours until
  /// `release()`.
  [[nodiscard]] const T* read_latest_slot(std::size_t& dropped) noexcept {
    dropped = 0;
    const std::uint64_t head = head_.load(std::memory_order_relaxed);
    // Load the producer's counter directly rather than going through
    // consumer_has_data(): that helper keeps a stale cache when it already
    // knows data exists, and here we specifically need the freshest tail.
    cached_tail_ = tail_.load(std::memory_order_acquire);
    if (FRAMEPACE_UNLIKELY(head == cached_tail_)) {
      return nullptr;
    }
    const std::uint64_t newest = cached_tail_ - 1;
    dropped = static_cast<std::size_t>(newest - head);
    if (dropped > 0) {
      head_.store(newest, std::memory_order_release);
    }
    return &slots_[newest & kMask];
  }

  // -------------------------------------------------------------------------
  // Either side (approximate while the other side runs)
  // -------------------------------------------------------------------------

  [[nodiscard]] std::size_t size() const noexcept {
    const std::uint64_t tail = tail_.load(std::memory_order_acquire);
    const std::uint64_t head = head_.load(std::memory_order_acquire);
    return static_cast<std::size_t>(tail - head);
  }

  [[nodiscard]] bool empty() const noexcept { return size() == 0; }
  [[nodiscard]] bool full() const noexcept { return size() >= Capacity; }
  [[nodiscard]] static constexpr std::size_t capacity() noexcept { return Capacity; }

  /// Total elements ever pushed / popped. Monotonic; useful for counters.
  [[nodiscard]] std::uint64_t push_count() const noexcept {
    return tail_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::uint64_t pop_count() const noexcept {
    return head_.load(std::memory_order_acquire);
  }

 private:
  static constexpr std::uint64_t kMask = Capacity - 1;

  [[nodiscard]] FRAMEPACE_ALWAYS_INLINE bool producer_has_space(
      std::uint64_t tail) noexcept {
    // Fast path: the cached head already proves there is room, so we never
    // touch the consumer's cache line.
    if (tail - cached_head_ < Capacity) {
      return true;
    }
    cached_head_ = head_.load(std::memory_order_acquire);
    return tail - cached_head_ < Capacity;
  }

  [[nodiscard]] FRAMEPACE_ALWAYS_INLINE bool consumer_has_data(
      std::uint64_t head) noexcept {
    if (head != cached_tail_) {
      return true;
    }
    cached_tail_ = tail_.load(std::memory_order_acquire);
    return head != cached_tail_;
  }

  // Line 1: written by the producer, read by the consumer.
  alignas(kCacheLineSize) std::atomic<std::uint64_t> tail_{0};
  // Producer-private cache of `head_`. Shares the producer's line on purpose.
  std::uint64_t cached_head_{0};

  // Line 2: written by the consumer, read by the producer.
  alignas(kCacheLineSize) std::atomic<std::uint64_t> head_{0};
  // Consumer-private cache of `tail_`.
  std::uint64_t cached_tail_{0};

  // Line 3+: the payload storage.
  alignas(kCacheLineSize) std::array<T, Capacity> slots_{};

  // Trailing pad so a neighbouring object cannot share our last line.
  char pad_[kCacheLineSize]{};
};

}  // namespace framepace
