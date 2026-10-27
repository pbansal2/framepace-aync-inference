// FramePace - SPSC queue correctness and the ThreadSanitizer target.
//
// Run under TSan:
//   cmake -B build-tsan -DENABLE_TSAN=ON && cmake --build build-tsan
//   ./build-tsan/test_spsc
//
// The concurrent cases below are the ones that matter to TSan. Every payload
// write in the ring is a plain non-atomic store read by another thread, so the
// release/acquire edge through the head/tail counters is the *only* thing
// making them race-free. If that ordering were weakened to relaxed, or the
// index cached incorrectly, TSan reports it here.

#include "framepace/spsc_queue.hpp"

#include <atomic>
#include <cstdint>
#include <thread>

#include "framepace/clock.hpp"
#include "framepace/frame.hpp"
#include "test_util.hpp"

using namespace framepace;

namespace {

/// Self-describing payload. `checksum` is derived from `seq`, and `fill` is
/// large enough to span multiple cache lines, so a torn or reordered slot
/// read is detectable rather than merely likely to look odd.
struct Payload {
  std::uint64_t seq;
  std::uint64_t checksum;
  std::array<std::uint32_t, 14> fill;

  static constexpr std::uint64_t expected_checksum(std::uint64_t s) noexcept {
    return s * 0x9E3779B97F4A7C15ULL ^ (s << 17);
  }

  static Payload make(std::uint64_t s) noexcept {
    Payload p{};
    p.seq = s;
    p.checksum = expected_checksum(s);
    for (std::size_t i = 0; i < p.fill.size(); ++i) {
      p.fill[i] = static_cast<std::uint32_t>(s + i);
    }
    return p;
  }

  [[nodiscard]] bool intact() const noexcept {
    if (checksum != expected_checksum(seq)) {
      return false;
    }
    for (std::size_t i = 0; i < fill.size(); ++i) {
      if (fill[i] != static_cast<std::uint32_t>(seq + i)) {
        return false;
      }
    }
    return true;
  }
};

/// Keep the TSan run to a sane wall-clock time. TSan costs roughly 5-15x.
constexpr std::uint64_t kStressItems = kThreadSanitizerEnabled ? 200'000 : 2'000'000;
constexpr std::uint64_t kLargeItems = kThreadSanitizerEnabled ? 3'000 : 20'000;

}  // namespace

// ---------------------------------------------------------------------------
// Single-threaded structure
// ---------------------------------------------------------------------------

FP_TEST(empty_queue_has_no_data) {
  SpscQueue<int, 4> q;
  int value = -1;
  FP_CHECK(q.empty());
  FP_CHECK(!q.full());
  FP_CHECK_EQ(q.size(), std::size_t{0});
  FP_CHECK_EQ(q.capacity(), std::size_t{4});
  FP_CHECK(!q.try_pop(value));
  FP_CHECK_EQ(value, -1);  // untouched on failure
  FP_CHECK(q.read_slot() == nullptr);
}

FP_TEST(uses_every_slot_not_capacity_minus_one) {
  // The monotonic-counter design exists so all N slots are usable. A
  // "sacrifice one slot" ring would report full at 3 here.
  SpscQueue<int, 4> q;
  for (int i = 0; i < 4; ++i) {
    FP_CHECK_MSG(q.try_push(i), "push " + std::to_string(i) + " of capacity 4");
  }
  FP_CHECK(q.full());
  FP_CHECK_EQ(q.size(), std::size_t{4});
  FP_CHECK(!q.try_push(99));
  FP_CHECK(q.write_slot() == nullptr);
}

FP_TEST(preserves_fifo_order) {
  SpscQueue<int, 8> q;
  for (int i = 0; i < 8; ++i) {
    FP_CHECK(q.try_push(i * 7));
  }
  for (int i = 0; i < 8; ++i) {
    int value = -1;
    FP_REQUIRE(q.try_pop(value));
    FP_CHECK_EQ(value, i * 7);
  }
  FP_CHECK(q.empty());
}

FP_TEST(wraps_around_many_times) {
  // Drive the index far past Capacity to prove masking and the cached-counter
  // fast paths stay consistent across wraps.
  SpscQueue<std::uint64_t, 4> q;
  std::uint64_t expected = 0;
  for (std::uint64_t round = 0; round < 10'000; ++round) {
    const std::size_t batch = (round % 4) + 1;
    for (std::size_t i = 0; i < batch; ++i) {
      FP_REQUIRE(q.try_push(expected + i));
    }
    for (std::size_t i = 0; i < batch; ++i) {
      std::uint64_t value = 0;
      FP_REQUIRE(q.try_pop(value));
      FP_CHECK_EQ(value, expected + i);
    }
    expected += batch;
  }
  FP_CHECK(q.empty());
  FP_CHECK_EQ(q.push_count(), expected);
  FP_CHECK_EQ(q.pop_count(), expected);
}

FP_TEST(space_reflects_occupancy) {
  SpscQueue<int, 8> q;
  FP_CHECK_EQ(q.space(), std::size_t{8});
  FP_CHECK(q.try_push(1));
  FP_CHECK(q.try_push(2));
  FP_CHECK_EQ(q.space(), std::size_t{6});
  int v = 0;
  FP_CHECK(q.try_pop(v));
  FP_CHECK_EQ(q.space(), std::size_t{7});
}

FP_TEST(in_place_write_and_read) {
  SpscQueue<Payload, 4> q;

  Payload* slot = q.write_slot();
  FP_REQUIRE(slot != nullptr);
  // Not visible until commit: the producer still owns the slot.
  FP_CHECK(q.empty());
  *slot = Payload::make(42);
  q.commit();
  FP_CHECK_EQ(q.size(), std::size_t{1});

  const Payload* read = q.read_slot();
  FP_REQUIRE(read != nullptr);
  FP_CHECK_EQ(read->seq, std::uint64_t{42});
  FP_CHECK(read->intact());
  // Still occupied until release.
  FP_CHECK_EQ(q.size(), std::size_t{1});
  q.release();
  FP_CHECK(q.empty());
}

FP_TEST(uncommitted_slot_is_reusable) {
  // Abandoning a write_slot() must leave the queue exactly as it was -- the
  // pacer relies on this to bail out of a failed downsample.
  SpscQueue<int, 4> q;
  int* slot = q.write_slot();
  FP_REQUIRE(slot != nullptr);
  *slot = 123;
  // no commit
  FP_CHECK(q.empty());
  FP_CHECK(q.try_push(7));
  int v = 0;
  FP_REQUIRE(q.try_pop(v));
  FP_CHECK_EQ(v, 7);
}

FP_TEST(pop_latest_keeps_newest_and_counts_drops) {
  SpscQueue<int, 8> q;
  int value = -1;
  bool had_value = true;

  // Empty case.
  FP_CHECK_EQ(q.pop_latest(value, had_value), std::size_t{0});
  FP_CHECK(!had_value);

  // Single element: kept, nothing dropped.
  FP_CHECK(q.try_push(5));
  FP_CHECK_EQ(q.pop_latest(value, had_value), std::size_t{0});
  FP_CHECK(had_value);
  FP_CHECK_EQ(value, 5);

  // Four elements: newest kept, three dropped.
  for (int i = 10; i < 14; ++i) {
    FP_CHECK(q.try_push(i));
  }
  FP_CHECK_EQ(q.pop_latest(value, had_value), std::size_t{3});
  FP_CHECK(had_value);
  FP_CHECK_EQ(value, 13);
  FP_CHECK(q.empty());
}

FP_TEST(read_latest_slot_skips_to_the_newest) {
  SpscQueue<int, 8> q;
  std::size_t dropped = 99;

  // Empty.
  FP_CHECK(q.read_latest_slot(dropped) == nullptr);
  FP_CHECK_EQ(dropped, std::size_t{0});

  // Single element: returned, nothing skipped.
  FP_CHECK(q.try_push(5));
  const int* slot = q.read_latest_slot(dropped);
  FP_REQUIRE(slot != nullptr);
  FP_CHECK_EQ(*slot, 5);
  FP_CHECK_EQ(dropped, std::size_t{0});
  q.release();
  FP_CHECK(q.empty());

  // Four elements: newest returned, three skipped, queue fully drained after
  // release.
  for (int i = 10; i < 14; ++i) {
    FP_CHECK(q.try_push(i));
  }
  slot = q.read_latest_slot(dropped);
  FP_REQUIRE(slot != nullptr);
  FP_CHECK_EQ(*slot, 13);
  FP_CHECK_EQ(dropped, std::size_t{3});
  // The three stale slots are already returned to the producer; only the
  // borrowed one is still held.
  FP_CHECK_EQ(q.size(), std::size_t{1});
  q.release();
  FP_CHECK(q.empty());
}

FP_TEST(read_latest_slot_frees_stale_slots_immediately) {
  // The skipped slots must become writable before release(), otherwise a
  // depth-N queue would still stall the producer for a whole inference.
  SpscQueue<int, 4> q;
  for (int i = 0; i < 4; ++i) {
    FP_CHECK(q.try_push(i));
  }
  FP_CHECK(q.full());

  std::size_t dropped = 0;
  const int* slot = q.read_latest_slot(dropped);
  FP_REQUIRE(slot != nullptr);
  FP_CHECK_EQ(*slot, 3);
  FP_CHECK_EQ(dropped, std::size_t{3});

  // Three slots are free while we still hold the newest.
  FP_CHECK_EQ(q.space(), std::size_t{3});
  FP_CHECK(q.try_push(100));
  FP_CHECK(q.try_push(101));
  // The borrowed slot's contents must survive the producer refilling around it.
  FP_CHECK_EQ(*slot, 3);
  q.release();
}

// ---------------------------------------------------------------------------
// Concurrent -- the ThreadSanitizer cases
// ---------------------------------------------------------------------------

FP_TEST(concurrent_sequence_is_lossless_and_ordered) {
  // Producer never drops (it retries), so the consumer must observe exactly
  // 0..N-1 in order. Any reordering, duplication or loss fails; any missing
  // synchronisation edge is a TSan report.
  auto queue = std::make_unique<SpscQueue<Payload, 64>>();
  std::atomic<std::uint64_t> received{0};
  std::atomic<std::uint64_t> corrupt{0};
  std::atomic<std::uint64_t> misordered{0};

  std::thread producer([&] {
    for (std::uint64_t i = 0; i < kStressItems; ++i) {
      const Payload p = Payload::make(i);
      while (!queue->try_push(p)) {
        cpu_relax();
      }
    }
  });

  std::thread consumer([&] {
    std::uint64_t next = 0;
    Payload p{};
    while (next < kStressItems) {
      if (!queue->try_pop(p)) {
        cpu_relax();
        continue;
      }
      if (!p.intact()) {
        corrupt.fetch_add(1, std::memory_order_relaxed);
      }
      if (p.seq != next) {
        misordered.fetch_add(1, std::memory_order_relaxed);
      }
      ++next;
      received.fetch_add(1, std::memory_order_relaxed);
    }
  });

  producer.join();
  consumer.join();

  FP_CHECK_EQ(received.load(), kStressItems);
  FP_CHECK_EQ(corrupt.load(), std::uint64_t{0});
  FP_CHECK_EQ(misordered.load(), std::uint64_t{0});
  FP_CHECK(queue->empty());
  FP_CHECK_EQ(queue->push_count(), kStressItems);
  FP_CHECK_EQ(queue->pop_count(), kStressItems);
}

FP_TEST(concurrent_in_place_api_is_race_free) {
  // Same traffic through write_slot()/commit() and read_slot()/release(),
  // because that is the path the engine actually uses for its 32KB tensors and
  // it publishes the payload through a different code path than try_push.
  auto queue = std::make_unique<SpscQueue<Payload, 32>>();
  std::atomic<std::uint64_t> corrupt{0};
  std::atomic<std::uint64_t> misordered{0};

  std::thread producer([&] {
    for (std::uint64_t i = 0; i < kStressItems; ++i) {
      Payload* slot = nullptr;
      while ((slot = queue->write_slot()) == nullptr) {
        cpu_relax();
      }
      *slot = Payload::make(i);
      queue->commit();
    }
  });

  std::thread consumer([&] {
    std::uint64_t next = 0;
    while (next < kStressItems) {
      const Payload* slot = queue->read_slot();
      if (slot == nullptr) {
        cpu_relax();
        continue;
      }
      if (!slot->intact()) {
        corrupt.fetch_add(1, std::memory_order_relaxed);
      }
      if (slot->seq != next) {
        misordered.fetch_add(1, std::memory_order_relaxed);
      }
      queue->release();
      ++next;
    }
  });

  producer.join();
  consumer.join();

  FP_CHECK_EQ(corrupt.load(), std::uint64_t{0});
  FP_CHECK_EQ(misordered.load(), std::uint64_t{0});
  FP_CHECK(queue->empty());
}

FP_TEST(concurrent_large_payload_is_not_torn) {
  // 32KB+ messages, the real InferenceRequest. A single slot spans ~256 cache
  // lines, so a missing release edge tears visibly rather than subtly.
  auto queue = std::make_unique<SpscQueue<InferenceRequest, kRequestQueueDepth>>();
  std::atomic<std::uint64_t> consumed{0};
  std::atomic<std::uint64_t> bad{0};

  std::thread producer([&] {
    for (std::uint64_t i = 0; i < kLargeItems; ++i) {
      InferenceRequest* slot = nullptr;
      while ((slot = queue->write_slot()) == nullptr) {
        cpu_relax();
      }
      slot->frame_id = i;
      slot->capture_ns = static_cast<std::int64_t>(i) * 8'333'333;
      slot->width = static_cast<std::uint32_t>(kTensorWidth);
      slot->height = static_cast<std::uint32_t>(kTensorHeight);
      // Fill the whole tensor with a frame-derived byte so any stale or torn
      // region shows up as a mismatch.
      const auto byte = static_cast<std::uint8_t>(i & 0xFF);
      slot->tensor.fill(byte);
      queue->commit();
    }
  });

  std::thread consumer([&] {
    std::uint64_t next = 0;
    while (next < kLargeItems) {
      const InferenceRequest* slot = queue->read_slot();
      if (slot == nullptr) {
        cpu_relax();
        continue;
      }
      const auto expect = static_cast<std::uint8_t>(slot->frame_id & 0xFF);
      bool ok = slot->frame_id == next && slot->width == kTensorWidth &&
                slot->height == kTensorHeight;
      for (std::size_t i = 0; ok && i < slot->tensor.size(); ++i) {
        ok = slot->tensor[i] == expect;
      }
      if (!ok) {
        bad.fetch_add(1, std::memory_order_relaxed);
      }
      queue->release();
      ++next;
      consumed.fetch_add(1, std::memory_order_relaxed);
    }
  });

  producer.join();
  consumer.join();

  FP_CHECK_EQ(consumed.load(), kLargeItems);
  FP_CHECK_EQ(bad.load(), std::uint64_t{0});
}

FP_TEST(concurrent_lossy_producer_never_corrupts) {
  // The engine's real regime: the producer outruns the consumer and pushes
  // fail. Drops are expected; what must hold is that everything the consumer
  // *does* see is intact and strictly increasing.
  auto queue = std::make_unique<SpscQueue<Payload, 4>>();
  std::atomic<bool> done{false};
  std::atomic<std::uint64_t> seen{0};
  std::atomic<std::uint64_t> dropped{0};
  std::atomic<std::uint64_t> corrupt{0};
  std::atomic<std::uint64_t> regressions{0};

  std::thread producer([&] {
    for (std::uint64_t i = 0; i < kStressItems; ++i) {
      if (!queue->try_push(Payload::make(i))) {
        dropped.fetch_add(1, std::memory_order_relaxed);
      }
    }
    done.store(true, std::memory_order_release);
  });

  std::thread consumer([&] {
    std::uint64_t last = 0;
    bool have_last = false;
    Payload p{};
    while (true) {
      if (queue->try_pop(p)) {
        if (!p.intact()) {
          corrupt.fetch_add(1, std::memory_order_relaxed);
        }
        if (have_last && p.seq <= last) {
          regressions.fetch_add(1, std::memory_order_relaxed);
        }
        last = p.seq;
        have_last = true;
        seen.fetch_add(1, std::memory_order_relaxed);
      } else if (done.load(std::memory_order_acquire)) {
        // Drain whatever landed between the empty check and the flag.
        if (!queue->try_pop(p)) {
          break;
        }
        if (!p.intact()) {
          corrupt.fetch_add(1, std::memory_order_relaxed);
        }
        seen.fetch_add(1, std::memory_order_relaxed);
      }
    }
  });

  producer.join();
  consumer.join();

  FP_CHECK_EQ(corrupt.load(), std::uint64_t{0});
  FP_CHECK_EQ(regressions.load(), std::uint64_t{0});
  FP_CHECK_EQ(seen.load() + dropped.load(), kStressItems);
  FP_CHECK_MSG(seen.load() > 0, "consumer saw nothing at all");
}

FP_TEST(concurrent_read_latest_slot_is_race_free) {
  // The inference worker's real consumption path: a fast producer, a slow
  // consumer, and latest-wins reads. This one needs TSan coverage specifically
  // because read_latest_slot() advances head_ *before* reading the payload, so
  // the producer may be refilling the skipped slots while the consumer reads
  // the one it kept. If that boundary were off by one, the producer would
  // overwrite the borrowed slot and TSan would report the overlap.
  auto queue = std::make_unique<SpscQueue<Payload, 8>>();
  std::atomic<bool> done{false};
  std::atomic<std::uint64_t> corrupt{0};
  std::atomic<std::uint64_t> regressions{0};
  std::atomic<std::uint64_t> kept{0};
  std::atomic<std::uint64_t> skipped{0};

  std::thread producer([&] {
    for (std::uint64_t i = 0; i < kStressItems; ++i) {
      while (!queue->try_push(Payload::make(i))) {
        cpu_relax();
      }
    }
    done.store(true, std::memory_order_release);
  });

  std::thread consumer([&] {
    std::uint64_t last = 0;
    bool have_last = false;
    while (true) {
      std::size_t dropped = 0;
      const Payload* slot = queue->read_latest_slot(dropped);
      if (slot == nullptr) {
        if (done.load(std::memory_order_acquire)) {
          break;
        }
        cpu_relax();
        continue;
      }
      // Read the payload twice with work in between: if the producer could
      // reclaim this slot early, the two reads would disagree.
      const std::uint64_t seq = slot->seq;
      const bool ok_first = slot->intact();
      cpu_relax();
      if (!ok_first || slot->seq != seq || !slot->intact()) {
        corrupt.fetch_add(1, std::memory_order_relaxed);
      }
      if (have_last && seq <= last) {
        regressions.fetch_add(1, std::memory_order_relaxed);
      }
      last = seq;
      have_last = true;
      queue->release();
      kept.fetch_add(1, std::memory_order_relaxed);
      skipped.fetch_add(dropped, std::memory_order_relaxed);
    }
  });

  producer.join();
  consumer.join();

  FP_CHECK_EQ(corrupt.load(), std::uint64_t{0});
  FP_CHECK_EQ(regressions.load(), std::uint64_t{0});
  // Every item is either kept or skipped; none may vanish.
  FP_CHECK_EQ(kept.load() + skipped.load(), kStressItems);
  FP_CHECK_MSG(skipped.load() > 0,
               "the consumer never fell behind, so latest-wins was not "
               "actually exercised");
}

FP_TEST(concurrent_bidirectional_request_response) {
  // The engine's exact topology: two queues, opposite directions, one thread
  // on each end. Exercises the case where a single thread is simultaneously a
  // producer on one ring and a consumer on the other -- the pattern most prone
  // to an accidental shared-state bug between the two queue objects.
  auto requests = std::make_unique<SpscQueue<Payload, 8>>();
  auto responses = std::make_unique<SpscQueue<Payload, 8>>();
  constexpr std::uint64_t kRounds = kThreadSanitizerEnabled ? 20'000 : 200'000;

  std::atomic<std::uint64_t> bad_response{0};
  std::atomic<bool> stop{false};

  std::thread worker([&] {
    Payload in{};
    while (!stop.load(std::memory_order_acquire)) {
      if (!requests->try_pop(in)) {
        cpu_relax();
        continue;
      }
      // Echo with a transform so the response is verifiably derived from the
      // request we actually received.
      const Payload out = Payload::make(in.seq + 1);
      while (!responses->try_push(out)) {
        if (stop.load(std::memory_order_acquire)) {
          return;
        }
        cpu_relax();
      }
    }
  });

  for (std::uint64_t i = 0; i < kRounds; ++i) {
    const Payload req = Payload::make(i);
    while (!requests->try_push(req)) {
      cpu_relax();
    }
    Payload resp{};
    while (!responses->try_pop(resp)) {
      cpu_relax();
    }
    if (!resp.intact() || resp.seq != i + 1) {
      bad_response.fetch_add(1, std::memory_order_relaxed);
    }
  }

  stop.store(true, std::memory_order_release);
  worker.join();

  FP_CHECK_EQ(bad_response.load(), std::uint64_t{0});
}

FP_TEST(reports_sanitizer_mode) {
  // Informational: makes the TSan run self-identifying in CI logs, so a green
  // "test_spsc passed" cannot be mistaken for a sanitised run when it was not.
  std::printf(
      "         ThreadSanitizer: %s | cache line: %zu bytes | "
      "stress items: %llu\n",
      kThreadSanitizerEnabled ? "ENABLED" : "disabled", kCacheLineSize,
      static_cast<unsigned long long>(kStressItems));
  FP_CHECK(true);
}
