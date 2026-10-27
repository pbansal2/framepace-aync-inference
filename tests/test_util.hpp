// FramePace - minimal self-contained test harness.
//
// No GoogleTest/Catch2 dependency on purpose: the TSan target must be
// buildable from a bare checkout with nothing but a compiler and CMake, and a
// third-party framework would add its own allocations and thread-local state to
// every ThreadSanitizer report.
//
// Include this header in exactly one translation unit per test binary -- it
// defines `main()`.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace framepace::testing {

struct TestCase {
  const char* name;
  void (*fn)();
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> cases;
  return cases;
}

inline int& failure_count() {
  static int failures = 0;
  return failures;
}

inline const char*& current_test() {
  static const char* name = "<none>";
  return name;
}

struct Registrar {
  Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

inline void report_failure(const char* file, int line, const std::string& what) {
  ++failure_count();
  std::fprintf(stderr, "  FAIL %s:%d\n    in [%s]\n    %s\n", file, line,
               current_test(), what.c_str());
}

template<typename T>
std::string to_text(const T& value) {
  if constexpr (std::is_same_v<T, bool>) {
    return value ? "true" : "false";
  } else if constexpr (std::is_floating_point_v<T>) {
    return std::to_string(value);
  } else if constexpr (std::is_integral_v<T>) {
    return std::to_string(value);
  } else {
    return "<value>";
  }
}

}  // namespace framepace::testing

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

#define FP_TEST(name)                                                 \
  static void name();                                                 \
  static ::framepace::testing::Registrar fp_reg_##name{#name, &name}; \
  static void name()

// ---------------------------------------------------------------------------
// Assertions
// ---------------------------------------------------------------------------
// CHECK_* records the failure and continues; REQUIRE_* returns from the test,
// for when continuing would read uninitialised state or hang.

#define FP_CHECK(cond)                                                              \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      ::framepace::testing::report_failure(__FILE__, __LINE__, "expected: " #cond); \
    }                                                                               \
  } while (false)

#define FP_CHECK_MSG(cond, msg)                                                  \
  do {                                                                           \
    if (!(cond)) {                                                               \
      ::framepace::testing::report_failure(                                      \
          __FILE__, __LINE__, std::string("expected: " #cond "\n    ") + (msg)); \
    }                                                                            \
  } while (false)

#define FP_CHECK_EQ(a, b)                                              \
  do {                                                                 \
    const auto fp_a = (a);                                             \
    const auto fp_b = (b);                                             \
    if (!(fp_a == fp_b)) {                                             \
      ::framepace::testing::report_failure(                            \
          __FILE__, __LINE__,                                          \
          std::string("expected " #a " == " #b "\n      lhs = ") +     \
              ::framepace::testing::to_text(fp_a) +                    \
              "\n      rhs = " + ::framepace::testing::to_text(fp_b)); \
    }                                                                  \
  } while (false)

#define FP_CHECK_NEAR(a, b, tol)                                                  \
  do {                                                                            \
    const double fp_a = static_cast<double>(a);                                   \
    const double fp_b = static_cast<double>(b);                                   \
    if (!(std::fabs(fp_a - fp_b) <= static_cast<double>(tol))) {                  \
      ::framepace::testing::report_failure(                                       \
          __FILE__, __LINE__,                                                     \
          std::string("expected " #a " ~= " #b " (tol " #tol ")\n      lhs = ") + \
              std::to_string(fp_a) + "\n      rhs = " + std::to_string(fp_b));    \
    }                                                                             \
  } while (false)

#define FP_REQUIRE(cond)                                                            \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      ::framepace::testing::report_failure(__FILE__, __LINE__, "required: " #cond); \
      return;                                                                       \
    }                                                                               \
  } while (false)

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  using namespace framepace::testing;

  const char* filter = argc > 1 ? argv[1] : nullptr;
  int ran = 0;
  int failed_tests = 0;

  std::printf("running %zu test(s)%s%s\n", registry().size(),
              filter ? " matching " : "", filter ? filter : "");

  for (const TestCase& tc : registry()) {
    if (filter != nullptr && std::strstr(tc.name, filter) == nullptr) {
      continue;
    }
    current_test() = tc.name;
    const int before = failure_count();
    std::printf("[ RUN  ] %s\n", tc.name);
    std::fflush(stdout);
    tc.fn();
    const bool ok = failure_count() == before;
    failed_tests += ok ? 0 : 1;
    std::printf("[ %s ] %s\n", ok ? " OK " : "FAIL", tc.name);
    std::fflush(stdout);
    ++ran;
  }

  std::printf("\n%d test(s) run, %d failed, %d assertion failure(s)\n", ran,
              failed_tests, failure_count());
  return failure_count() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
