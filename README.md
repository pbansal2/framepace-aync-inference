# FramePace

A low-latency C++17 library that keeps a **120 FPS (8.333 ms) render loop** running
at full rate while an on-device ML model takes **15–30 ms** per inference. The model
runs on its own thread; its results are projected forward to each frame's
presentation time, so the renderer never waits and the output doesn't visibly lag.

The render thread never blocks, never waits on inference, and performs **zero heap
allocations** per frame. Verified with ThreadSanitizer, AddressSanitizer and a
counting allocator.

> **What this is, and what it isn't**
>
> - **It is** the pipeline *around* a model: frame downsampling (CPU SIMD — NEON or
>   SSSE3), two lock-free queues, a worker-thread contract, and per-frame
>   extrapolation and pacing.
> - **It ships no ML model and no GPU code.** The benchmark's "inference" is a
>   simulated 15–30 ms sleep, and the integration example uses a tiny stand-in
>   model (a centroid finder). You bring the model — ONNX Runtime, Core ML,
>   TFLite, TensorRT, anything callable from C++ — and your own frame source.

## Contents

- [How it works](#how-it-works)
- [Quick start](#quick-start) — build, test, and see it work in about five minutes
- [Integrate your model and engine](#integrate-your-model-and-engine) — step by step
- [Configuration reference](#configuration-reference)
- [Platform support](#platform-support)
- [Troubleshooting](#troubleshooting)
- [Benchmark results](#benchmark-results)
- [Verification](#verification) · [Gotchas](#gotchas) · [Repository layout](#repository-layout)

---

## How it works

At 120 FPS you have 8.333 ms per frame. Inference takes 15–30 ms — two to four
frames. Something has to give:

| Approach | Consequence |
|---|---|
| Block the render loop on inference | Frame rate collapses to 33–66 FPS |
| Present each result when it lands | Output lags input by 15–30 ms, visibly |
| **Decouple, then extrapolate forward** | Full frame rate, lag largely hidden |

FramePace takes the third:

```
 render thread (120 Hz)                              model thread (your code)
 ──────────────────────                              ────────────────────────
 your engine writes a frame
          │
 Pacer::step() ── downsample (SIMD) ──► request queue ──► newest frame only
          │                                                   │
          │                                              your model runs
          │                                                   │
          ◄──────────── newest result only ◄─ result queue ◄──┘
          │
 extrapolate to this frame's presentation time
          │
 your renderer draws with the estimate
```

Both queues are **latest-wins**: stale frames and stale results are dropped rather
than replayed, because in a latency pipeline a 90 ms-old frame has no value. The
model always works on the freshest frame, and the renderer always uses the freshest
result — projected forward by however old it is.

---

## Quick start

### 1. Prerequisites

You need **CMake ≥ 3.20** and a **C++17 compiler**. There are no other
dependencies — no package manager, no submodules, no test framework.

| Platform | Install |
|---|---|
| macOS | `xcode-select --install` (Apple clang), then `brew install cmake` |
| Ubuntu / Debian | `sudo apt install build-essential cmake` — add `clang` for ThreadSanitizer |
| Windows | Not tested yet — see [Platform support](#platform-support) |

Check: `cmake --version` should print 3.20 or newer. (Ubuntu 20.04 ships 3.16; use
`pip install cmake` there.)

### 2. Build and test

```bash
git clone https://github.com/pbansal2/frame_pacing_engine.git
cd frame_pacing_engine

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Success looks like this (older CMake versions word it slightly differently):

```
100% tests passed out of 4
```

(4 test programs, 57 test cases.)

### 3. See it work

```bash
./build/integration_example
```

This runs the full pipeline for 5 seconds with a stand-in model that takes 20 ms,
then scores what the renderer would have drawn against the true answer:

```
FramePace integration example
  geometry        1920x1080 -> 240x135 luma, 2 of 32 features used
  model latency   20.0 ms (emulated)
  running         600 frames at 120 Hz...

Render loop
  effective rate         120.0 fps
  frames with estimate   597 / 600
  inferences completed   209 (2.9 frames each)
  stale frames skipped   389
  submits refused        0

Position error of what the renderer draws (source pixels, lower is better)
                           mean      p95
  newest result as-is      48.2     59.4
  FramePace estimate        7.7     15.5
  -> extrapolation removes 84% of the latency-induced error
```

How to read it: the model finished 209 times, but 600 frames were drawn at a full
120 fps. Drawing the newest result as-is would put the object **48 px** behind where
it really is; FramePace's projected estimate is **7.7 px** off. Your numbers will
differ a little from run to run. Try `--latency-ms 5` or `--latency-ms 30` to
see how model speed changes the picture.

### 4. Optional: benchmark and concurrency check

```bash
./build/frame_bench                        # ~15 s; results explained below

cmake -B build-tsan -DENABLE_TSAN=ON       # ThreadSanitizer build
cmake --build build-tsan -j
./build-tsan/test_spsc                     # expect: 17 test(s) run, 0 failed
```

Don't benchmark a sanitizer build — they run 5–15x slower, and `frame_bench` warns
if you try.

---

## Integrate your model and engine

[`examples/integration_example.cpp`](examples/integration_example.cpp) is the
template. It uses only the public API, compiles in this repo, and has exactly three
places to edit, each marked `INTEGRATION POINT`. **Copy it and replace those three
functions** — the rest is wiring you keep. The steps below explain each part.

### Step 1 — Add FramePace to your build

Link `framepace::framepace`. The same target name works however you pull it in.

**FetchContent** (simplest):

```cmake
include(FetchContent)
FetchContent_Declare(framepace
  GIT_REPOSITORY https://github.com/pbansal2/frame_pacing_engine.git
  GIT_TAG        main)
FetchContent_MakeAvailable(framepace)

target_link_libraries(your_app PRIVATE framepace::framepace)
```

**`add_subdirectory`** (vendored copy): `add_subdirectory(third_party/framepace)`,
then the same `target_link_libraries` line.

**Installed** (`find_package`):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/your/prefix
cmake --build build && cmake --install build
```

```cmake
find_package(framepace REQUIRED)        # pass -DCMAKE_PREFIX_PATH=/your/prefix
target_link_libraries(your_app PRIVATE framepace::framepace)
```

When embedded, FramePace builds only the library: no tests, benchmarks, or examples
appear in your build or your `ctest`, and its strict warning flags apply only to its
own sources, never to yours.

### Step 2 — Match the frame geometry to your engine

FramePace takes **one 8-bit luma (grayscale) plane** per frame and downsamples it by
`2^N` on each axis for the model. Sizes are compile-time — every hot-path buffer is
a fixed-size array, which is what makes the pipeline allocation-free — so set them
when you configure:

```bash
cmake -B build -DFRAMEPACE_SOURCE_WIDTH=1280 -DFRAMEPACE_SOURCE_HEIGHT=720 \
               -DFRAMEPACE_DOWNSAMPLE_LOG2=3 -DFRAMEPACE_MAX_FEATURES=64
# -> model input is 160x90; results carry up to 64 floats
```

With FetchContent, `set()` these before `FetchContent_MakeAvailable`.

| Option | Default | Meaning |
|---|---|---|
| `FRAMEPACE_SOURCE_WIDTH` × `HEIGHT` | 1920 × 1080 | Size of the luma frame you hand to `Pacer::step()` |
| `FRAMEPACE_DOWNSAMPLE_LOG2` | 3 | Model input = source ÷ 2^N per axis (0–6). Default gives 240 × 135 |
| `FRAMEPACE_MAX_FEATURES` | 32 | Most floats one inference may return |

- Width and height must both divide by `2^N`; CMake rejects anything else at
  configure time with a clear message (e.g. 1366 × 768 at N = 3).
- **Set these only through CMake**, never with a manual `#define`. CMake applies
  them to the library and every consumer alike; a mismatch silently corrupts memory.
- **RGB or RGBA input** must be converted to luma first, for example
  `Y = (77*R + 150*G + 29*B) >> 8` per pixel. Multi-channel tensors aren't supported.
- The model receives the downsampled tensor as `uint8` (0–255).
  `framepace::normalize_u8_to_f32()` converts it to `float` with any scale and bias
  (default `/255` → 0–1), vectorised.

### Step 3 — Write the model worker

Run your model on one dedicated thread. That thread takes frames from
`pacer->requests()` and posts results to `pacer->results()`:

```cpp
void model_worker(framepace::RequestQueue& requests,
                  framepace::ResultQueue& results,
                  const std::atomic<bool>& stop) {
  using namespace framepace;
  auto input = std::make_unique<std::array<float, kTensorPixels>>();  // once, up front

  while (!stop.load(std::memory_order_acquire)) {
    std::size_t skipped = 0;
    const InferenceRequest* req = requests.read_latest_slot(skipped);  // newest frame
    if (req == nullptr) {
      std::this_thread::sleep_for(std::chrono::microseconds{200});     // nothing queued
      continue;
    }

    InferenceResult result{};
    result.frame_id   = req->frame_id;
    result.capture_ns = req->capture_ns;            // required: drives extrapolation
    normalize_u8_to_f32(req->tensor, *input);       // u8 -> float [0, 1]
    requests.release();                             // req is invalid from here on

    run_model(input->data(), kTensorWidth, kTensorHeight, result);  // YOUR MODEL:
                                                    // fills features[], feature_count
    result.complete_ns = to_ns(Clock::now());
    (void)results.try_push(result);                 // full? drop it -- never wait
  }
}
```

The contract this loop follows — any version of it must keep these:

1. **One thread only** uses `requests()`'s consumer side and `results()`'s producer
   side. The queues are single-producer/single-consumer; a second thread corrupts
   them silently rather than failing.
2. **Every non-null `read_latest_slot()` is matched by exactly one `release()`.**
   Copy what you need first — the request pointer is invalid after `release()`.
   Copying out and releasing *before* inference (as above) lets the render thread
   keep submitting while your model runs.
3. **Copy `frame_id` and `capture_ns` into the result.** Extrapolation is computed
   from capture time, and results must arrive with increasing `frame_id`.
4. **Never block on a full result queue** — drop the result. The renderer only
   ever wants the newest one.

**Async or GPU runtimes** (Core ML, ONNX Runtime async, TensorRT/CUDA streams): wait
for completion *on this worker thread* — e.g. block on the future, fence, or
callback-signalled event — and post the result from here. Don't call
`results.try_push()` from a runtime callback thread; that makes it a second producer.

### Step 4 — Drive it from your render loop

```cpp
// Pacer holds ~1 MB of scratch: allocate once at startup, never per frame or on the stack.
auto pacer = std::make_unique<framepace::Pacer>(framepace::Pacer::Config{});
std::atomic<bool> stop{false};
std::thread worker([&] { model_worker(pacer->requests(), pacer->results(), stop); });

auto frame = std::make_unique<framepace::SourceFrame>();   // 8-bit luma plane
auto deadline = framepace::Clock::now() + framepace::kFrameBudget120Hz;

for (std::uint64_t id = 0; running; ++id) {
  const auto capture = framepace::Clock::now();
  your_engine_write_luma(frame->pixels());                // YOUR ENGINE: fill the frame

  // Takes the newest result, submits this frame if there's room, and returns
  // features projected to `deadline`. Never blocks on the model.
  const framepace::Estimate est = pacer->step(*frame, id, capture, deadline);

  if (est.valid) {                                        // false until the first result
    your_renderer_draw(est.features.data(), est.feature_count);   // YOUR RENDERER
  }
  deadline = pacer->sleep_until_next_frame(deadline);     // holds 120 Hz without drift
}

stop.store(true, std::memory_order_release);
worker.join();
```

`step()` and `sleep_until_next_frame()` must always be called from the same
thread. If your engine already paces frames itself (vsync, a game loop), skip
`sleep_until_next_frame()` and pass your own presentation time as `deadline`.

### Step 5 — Check that it's healthy

`pacer->counters()` (render side) and your worker's own tallies tell you whether the
pipeline is working. The example prints them; in your app, log them occasionally.

| Signal | Healthy | If not |
|---|---|---|
| Frame rate | ≈ your target (e.g. 119.9 fps) | Your per-frame work exceeds the budget — measure it on its own |
| `frames_without_estimate` | A few, at startup only | Results aren't arriving — check the worker calls `try_push()` and copies `frame_id` |
| `results_rejected_out_of_order` | 0 | You're not copying `frame_id` from the request, or posting from more than one thread |
| `frames_extrapolation_clamped` | Under ~50% of frames | Latency hiding is capped — raise `max_extrapolation` (see Tuning) |
| `submit_skipped_full` | Small | Your worker isn't releasing slots promptly — `release()` before inference |
| Stale frames skipped (worker) | Large — that's normal | It counts frames the model correctly skipped because a newer one was waiting |

### What extrapolation can and can't do

The estimate projects **each feature linearly** from the two newest results. That's
right for quantities that move smoothly: positions, keypoints, offsets, flow,
brightness. It is **wrong** for:

- **Discrete outputs** — class IDs, counts, flags. Use the newest raw result:
  `pacer->tracker().estimate_at(t, framepace::Duration::zero())`, or set
  `max_extrapolation` to 0 if *all* your features are discrete.
- **Wrapping values** — angles that jump from +180° to −180°. Output `sin` and `cos`
  as two features instead, and recover the angle with `atan2`.

There's no per-feature switch: every feature in a result uses the same horizon.

### Tuning

Runtime settings live in `framepace::Pacer::Config`:

| Field | Default | When to change it |
|---|---|---|
| `frame_budget` | `kFrameBudget120Hz` | Other refresh rates (`kFrameBudget60Hz`, or any `Duration`) |
| `max_extrapolation` | 50 ms | Raise if `frames_extrapolation_clamped` is over ~50%. Size it as *end-to-end latency + half the inference period*, not as a frame count |
| `submit_interval` | 1 | Set 2–3 to downsample only every Nth frame when your model is much slower than the frame rate |
| `spin_margin` | 300 µs | How long `sleep_until_next_frame()` spins before the deadline. Lower saves CPU; higher reduces jitter |

FramePace doesn't change thread priorities. For a shipping app, raise the render
thread's priority yourself (QoS on macOS/iOS, `SCHED_FIFO` on Linux) — on a
general-purpose scheduler, preemption dominates frame jitter.

---

## Configuration reference

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release   # the normal build
cmake -B build -DENABLE_TSAN=ON             # ThreadSanitizer
cmake -B build -DENABLE_ASAN=ON             # AddressSanitizer + LeakSanitizer
cmake -B build -DENABLE_UBSAN=ON            # UndefinedBehaviorSanitizer
cmake -B build -DFRAMEPACE_WERROR=ON        # warnings are errors (CI uses this)
cmake -B build -DFRAMEPACE_NATIVE_ARCH=ON   # -mtune=native
```

| Option | Default (standalone) | Default (embedded) | Purpose |
|---|---|---|---|
| `FRAMEPACE_SOURCE_WIDTH` / `_HEIGHT` | 1920 / 1080 | same | Frame size — see [Step 2](#step-2--match-the-frame-geometry-to-your-engine) |
| `FRAMEPACE_DOWNSAMPLE_LOG2` | 3 | same | Model input = source ÷ 2^N |
| `FRAMEPACE_MAX_FEATURES` | 32 | same | Floats per result |
| `FRAMEPACE_BUILD_TESTS` | ON | OFF | The four test programs |
| `FRAMEPACE_BUILD_BENCHMARKS` | ON | OFF | `frame_bench` |
| `FRAMEPACE_BUILD_EXAMPLES` | ON | OFF | `integration_example` |
| `FRAMEPACE_INSTALL` | ON | OFF | `cmake --install` and `find_package` support |

TSan and ASan can't be combined (both replace the allocator); configure two build
directories instead. The bare `ENABLE_*` names are a standalone convenience — when
embedding, use `FRAMEPACE_ENABLE_TSAN` / `_ASAN` / `_UBSAN`, which can't collide
with your own project's options.

`frame_bench` options:

```bash
./build/frame_bench --only downsample|spsc|estimate|pipeline   # one stage
./build/frame_bench --frames 600                               # shorter run
./build/frame_bench --min-latency 5 --max-latency 8            # a faster model
./build/frame_bench --submit-interval 2                        # infer every 2nd frame
./build/frame_bench --busy-spin                                # worker pins a core
```

---

## Platform support

| Platform | Compiler | Status |
|---|---|---|
| macOS arm64 (NEON) | Apple clang 21 | Tested: Release, TSan, ASan + UBSan, 1080p / 720p / 4K geometries |
| macOS arm64 (NEON) | GCC 16 | Tested: Release, all tests |
| x86_64 (SSSE3) | clang | Tested under Rosetta: SIMD kernel bit-exact on 135 geometries, queue tests |
| Linux x86_64 | GCC, clang | Covered by CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)); not yet run on Linux hardware |
| Windows | MSVC | **Untested.** MSVC code paths exist but have never been compiled |

Any other architecture falls back to the portable scalar kernel, which produces
bit-identical output.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cmake: command not found`, or "CMake 3.20 or higher is required" | Install or upgrade CMake ([Prerequisites](#1-prerequisites)) |
| `test_pacer` fails, but only on a busy machine | It asserts real-time cadence. Re-run on an idle machine; the other tests are timing-independent |
| `Source WxH is not divisible by 8` at configure time | Pick a resolution divisible by 2^`FRAMEPACE_DOWNSAMPLE_LOG2`, or lower that option |
| Linux TSan: `FATAL: ThreadSanitizer: unexpected memory mapping` | `sudo sysctl vm.mmap_rnd_bits=28` — a kernel ASLR setting newer than some TSan versions |
| `est.valid` stays false | No results are arriving: check the worker calls `try_push()`, and copies `frame_id` and `capture_ns` |
| Estimates jump or overshoot | A feature is discrete or wraps around — see [What extrapolation can and can't do](#what-extrapolation-can-and-cant-do) |
| Crash (stack overflow) creating a `Pacer` | It's over 1 MB at 1080p and larger at higher resolutions, more than many thread stacks hold. Always use `std::make_unique<Pacer>(...)` |


---

## Benchmark results

10-second run, 1200 measured frames. Apple M-series arm64, 10 cores, 128-byte
cache lines, Apple clang 21, `-O3`.

```
  SIMD kernel            NEON (vpaddlq_u8 + vrshrn_n_u16)
  cache line             128 bytes
  hardware threads       10
  frame budget           8.3333 ms (120 FPS)
  geometry               1920x1080 -> 240x135
  request queue          127 KB
  ThreadSanitizer        off
```

`request queue 127 KB` is 4 × 32KB tensors held inline — the queue allocates
nothing, ever, construction included. The cache line figure is detected from the
architecture, not from libc++ (see [Gotchas](#gotchas)).

### 1. Downsample — 1920×1080 → 240×135 (box, factor 8)

| Kernel | p50 | p90 | p99 | p99.9 |
|---|---|---|---|---|
| halve (scalar) | 319.75 µs | 405.92 µs | 466.29 µs | 545.50 µs |
| halve (SIMD) | **26.75 µs** | 27.88 µs | 29.79 µs | 39.63 µs |
| full chain (3 levels) | **35.13 µs** | 35.58 µs | 43.63 µs | 128.75 µs |

```
SIMD speedup (single level, p50): 11.95x
chain throughput:                 77.48 GB/s
share of an 8.333ms frame budget:  0.42%
```

**11.95x** comes from two NEON instructions doing the whole box filter:
`vpaddlq_u8` sums *adjacent byte pairs* into 16-bit lanes — exactly the
horizontal half of a 2×2 reduction, with no shuffle or de-interleave — and
`vrshrn_n_u16(_, 2)` rounds and narrows back to bytes in one instruction. The
x86 path mirrors this with `_mm_maddubs_epi16`.

Factor 8 is three exact halvings (1920/8 = 240, 1080/8 = 135), so no level needs
padding, cropping, or an odd-dimension case. The full chain costs **0.42% of the
frame budget** — downsampling is not a bottleneck and was not worth optimising
further.

> The scalar row is a *genuine* scalar baseline. It carries
> `#pragma clang loop vectorize(disable)` because clang auto-vectorises the
> pattern at `-O3`; without it both rows run vector code and the speedup reads a
> meaningless **1.17x**.

### 2. SPSC queue — cross-core round-trip

| | p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| push→pop→push→pop | 0.125 µs | 0.500 µs | 0.583 µs | 0.750 µs | 47.04 µs |

```
one-way hop (p50 / 2): 0.062 us
```

**62ns** to move a message between cores. The queue is wait-free in both
directions: no locks, no CAS retry, no syscall. Head and tail live on separate
128-byte cache lines and each side caches its view of the other's counter, so in
steady state **neither thread touches the other's cache line at all**.

A full queue fails the push and returns instead of waiting — the property the
render thread actually needs. The 47µs max is a scheduler preemption on one of
50,000 round trips, not queue behaviour.

### 3. Extrapolation — per-frame estimate

```
32 features, 200000 calls in blocks of 100
p50=5 ns   p99=6 ns   max=74 ns
```

**5ns** to project 32 features forward in time — about 0.00006% of the frame
budget. The latency hiding is effectively free; all the engineering is in the
correctness of the degenerate cases (single sample, zero sample span,
presentation before capture, out-of-order results), each of which has a test.

Timed in blocks of 100 because a single call is faster than reading the clock.

### 4. Full pipeline — 120Hz loop + async inference

```
frames: 1200   inference: 15.0-30.0ms (sleep)   submit every 1 frame(s)
```

| Metric | p50 | p90 | p99 | p99.9 |
|---|---|---|---|---|
| render-loop CPU per frame | 420.92 µs | 1301.29 µs | 3079.79 µs | 13335.46 µs |
| presented frame interval | 8.330 ms | 10.181 ms | 12.854 ms | 22.376 ms |
| interval jitter \|err\| | 804.79 µs | 2579.00 µs | 6649.88 µs | 14042.75 µs |
| estimate source age | 49.675 ms | 65.501 ms | 74.300 ms | 75.000 ms |
| capture → visible (e2e) | 34.366 ms | 42.904 ms | 50.094 ms | 51.200 ms |

**Cadence**

```
wall clock                         10016.3 ms
target                             10000.0 ms
deviation                          +0.16 %
effective rate                     119.80 fps
render CPU budget used (p99)       36.96 % of 8.333ms
```

**119.80 FPS against a 120 FPS target, +0.16% over ten seconds.** Deadlines
advance on an absolute schedule (`deadline + budget`, never `now + budget`), so
per-frame overshoot cannot accumulate into phase drift.

`presented frame interval` p50 of **8.330ms** against a 8.333ms budget is the
headline: the loop holds its cadence while inference runs 2–4x slower.

**Pipeline counters**

```
frames                             1320
inference submitted                1315
submits refused (queue full)       5
submits throttled (cadence)        0
requests skipped as stale          886
inferences completed               429
results ingested                   427
results dropped as stale           0
results dropped (queue full)       0
frames with no estimate yet        3
frames at extrap. ceiling          555 (42.0% of frames)

frames presented per inference     3.08
frames reusing a prior inference   67.5 %
```

These account for every frame exactly, which is how you know nothing vanished:

```
submitted 1315  +  refused 5                =  1320 frames
skipped_stale 886  +  completed 429         =  1315 submitted
```

**The last two lines are the engine's entire purpose.** Inference completes 429
times while 1320 frames are presented — 3.08 frames per inference. **67.5% of
frames are presented from an extrapolated prior result.** Without that, this
workload would run at 39 FPS or sit 30ms behind the input.

`requests skipped as stale: 886` is load shedding working, not an error. The
worker always takes the newest queued frame, so 886 frames were correctly
declined rather than inferred on too late. Note that `submits refused` is only
**5** — the render thread is essentially never blocked by backpressure, because
the worker drains the queue on every pickup.

`frames at extrap. ceiling: 42.0%` means the extrapolation horizon binds on two
frames in five. That is tuned, not accidental — see below.

---

## How to read these numbers critically

**`estimate source age` (49.7ms) is not pipeline latency.** It is measured to the
*presentation deadline*, so it exceeds true latency by up to one frame budget by
construction. `capture → visible` (34.4ms) is the real figure: capture → the
moment the renderer can first use the result. The benchmark reports both
separately so they cannot be conflated.

**34.4ms e2e for a 15–30ms model is about right** — roughly 22ms inference, plus
queue wait, plus up to a frame of ingest delay. It is not reducible without
reducing inference time.

**Interval jitter is scheduler-bound, not engine-bound.** p99 of 6.65ms looks bad
until you notice the engine's own per-frame CPU is 37% of budget at p99. The
benchmark runs at normal priority on a general-purpose scheduler alongside
everything else on the machine. A shipping integration would elevate thread QoS
(macOS) or use `SCHED_FIFO` (Linux, privileged); **FramePace does not do this for
you.** Judge the engine by `render-loop CPU per frame`.

**Roughly 0.1% of frames exceed the budget.** `render-loop CPU` p99.9 of 13.3ms
and max of 15.8ms are above 8.333ms — one or two frames out of 1200, correlating
with the `presented frame interval` p99.9 of 22.4ms. These are preemption and
cold-memory outliers, not steady-state behaviour (p50 is 421µs, p99 is 3.08ms).
On a non-realtime scheduler this is the floor.

**`frames at extrap. ceiling` is the tuning dial for latency hiding.** 42% is
healthy. The obvious horizon — "4 frames at 120Hz" = 33ms — clamped on **94% of
frames**, meaning the latency hiding was silently saturated. The horizon has to
cover the steady-state *age of the newest result*, which is
`e2e_latency + inference_period/2` ≈ 36ms, not a frame count. The default is now
50ms. `frame_bench` prints an explicit warning above 50%.

**Single runs vary** with thermal and scheduling state. Treat them as indicative.

---

## Verification

| Check | Result |
|---|---|
| Tests (Release, clang and GCC) | 57/57 pass |
| Tests (ThreadSanitizer) | 57/57 pass, **0 data races** |
| Tests (AddressSanitizer + UBSan, non-default 1280×720 geometry) | 57/57 pass, 0 reports |
| Hot-path heap allocations | **0** over 240 frames × 2 live threads |
| Compiler diagnostics (`-Werror`, strict warning set, clang and GCC) | 0 |
| SIMD/scalar bit-exact equivalence | 135 geometries, NEON and SSSE3 |

Both headline claims are **proved, not asserted**:

- **Zero allocations.** Code review cannot establish this — a hidden allocation
  can come from a `std::function`, container growth, a lambda capture promoted to
  the heap, or an over-aligned type routed through aligned `operator new`. So
  [test_no_alloc.cpp](tests/test_no_alloc.cpp) replaces all 20 global allocation
  forms with a counting allocator, arms the counter around the steady-state loop,
  and asserts exactly zero. Two positive controls confirm the probe actually
  interposes — without them a silently-broken probe would make every zero result
  vacuous.

- **No data races.** A clean sanitizer run only means something if the tests
  exercise the synchronisation. Weakening the queue's `release` store to
  `relaxed` produces dozens of race reports (27–36 observed; the count varies
  with thread interleaving) in exactly these tests; reverting returns to clean.
  CI runs that negative control on every push — the exact commands are in the
  `tsan-negative-control` job of [ci.yml](.github/workflows/ci.yml).

Startup allocations (about 20, depending on the standard library) are expected and
excluded by design: `Pacer` owns its downsample scratch inline, so it is constructed
once via `make_unique` at startup. That single allocation is explicitly outside the
hot path.

---

## Gotchas

Three things that cost real debugging time and are easy to repeat:

1. **libc++ reports `hardware_destructive_interference_size == 64` on arm64, but
   Apple silicon uses 128-byte lines** (`sysctl -n hw.cachelinesize`). Trusting
   the library value leaves two "padded" atomics sharing a line and silently
   reintroduces the false sharing the queue exists to avoid.
   [config.hpp](include/framepace/config.hpp) does architecture detection first.

2. **FIFO consumption cost 4x in latency.** The original request queue was FIFO,
   so the worker always dequeued the *oldest* frame — a depth-4 queue baked in
   ~90ms of staleness. Measured e2e was **139ms** for a 15–30ms model. Latest-wins
   brought it to 34.4ms, now independent of queue depth.

3. **clang elides a `new`/`delete` pair whose result is unused**, which made both
   allocation-probe controls report zero and look like a broken probe.
   `volatile auto* p` does not help — that declares a pointer *to* volatile,
   leaving the pointer itself dead. An `asm` sink does.

---

---

## Repository layout

```
include/framepace/
  pacer.hpp             render-thread side: step(), ingest, submit, extrapolate, pace
  spsc_queue.hpp        wait-free single-producer/single-consumer ring
  frame.hpp             frame and message types, compile-time geometry
  downsample.hpp        SIMD downsampling and u8 -> float normalisation
  inference_worker.hpp  the simulated model used by tests and the benchmark
  clock.hpp             monotonic clock, spin hints, sub-millisecond sleep
  config.hpp            cache-line size, sanitizer detection, Span, branch hints
  stats.hpp             fixed-capacity percentile recorder
src/                    downsample.cpp, pacer.cpp, inference_worker.cpp
examples/               integration_example.cpp -- the template to copy
tests/                  test_spsc, test_downsample, test_pacer, test_no_alloc
benchmarks/             frame_bench.cpp
```

---

## License

MIT — see [LICENSE](LICENSE).
