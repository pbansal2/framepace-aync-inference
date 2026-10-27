cmake_minimum_required(VERSION 3.20)

project(framepace
  VERSION 0.1.0
  DESCRIPTION "Low-latency frame pacing engine: 120Hz render loop decoupled from async ML inference"
  LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Top-level vs embedded
# ---------------------------------------------------------------------------
# Everything that touches global state (CMAKE_CXX_STANDARD, output directories,
# the `format` targets, test registration) is gated on this. A project that
# pulls framepace in via add_subdirectory/FetchContent must not have its own
# build settings, target names, or `ctest` list silently rewritten by ours.
# PROJECT_IS_TOP_LEVEL is CMake 3.21; this supports the declared 3.20 minimum.
if(DEFINED PROJECT_IS_TOP_LEVEL)
  set(FRAMEPACE_IS_TOP_LEVEL ${PROJECT_IS_TOP_LEVEL})
elseif(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(FRAMEPACE_IS_TOP_LEVEL ON)
else()
  set(FRAMEPACE_IS_TOP_LEVEL OFF)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
# FRAMEPACE_-prefixed names are canonical: they are what a consumer should set,
# because unprefixed ENABLE_* would collide with an identically-named option in
# a parent project. The unprefixed spellings are kept as a top-level
# convenience, since the README documents them.
option(FRAMEPACE_ENABLE_TSAN  "Build with ThreadSanitizer"            OFF)
option(FRAMEPACE_ENABLE_ASAN  "Build with AddressSanitizer + LSan"    OFF)
option(FRAMEPACE_ENABLE_UBSAN "Build with UndefinedBehaviorSanitizer" OFF)
option(FRAMEPACE_WERROR       "Treat warnings as errors"              OFF)
option(FRAMEPACE_NATIVE_ARCH  "Tune for the build host (-mtune=native)" OFF)
option(FRAMEPACE_INSTALL      "Generate install and export rules"
       ${FRAMEPACE_IS_TOP_LEVEL})

# Tests and benchmarks default ON standalone, OFF when embedded: a consumer
# wants the library, not four extra executables and four entries in its ctest.
option(FRAMEPACE_BUILD_TESTS      "Build the test targets"
       ${FRAMEPACE_IS_TOP_LEVEL})
option(FRAMEPACE_BUILD_BENCHMARKS "Build the benchmark harness"
       ${FRAMEPACE_IS_TOP_LEVEL})
option(FRAMEPACE_BUILD_EXAMPLES   "Build the integration example"
       ${FRAMEPACE_IS_TOP_LEVEL})

# ---------------------------------------------------------------------------
# Pipeline geometry
# ---------------------------------------------------------------------------
# Compile-time, because every hot-path buffer is a fixed-size inline array.
# Attached to the framepace target as PUBLIC definitions so the library and
# every consumer agree on the size of each message -- see frame.hpp.
#
# Each cache entry is created only if the caller has not already set the
# variable. A parent project doing `set(FRAMEPACE_SOURCE_WIDTH 1280)` before
# add_subdirectory/FetchContent_MakeAvailable would otherwise be overridden on
# the FIRST configure only: under policy CMP0126's old behaviour (the default
# for our 3.20 minimum) creating a cache entry deletes a same-named normal
# variable, so the first configure used 1920 and the second 1280.
macro(_framepace_geometry_option name default doc)
  if(NOT DEFINED ${name})
    set(${name} ${default} CACHE STRING "${doc}")
  endif()
endmacro()
_framepace_geometry_option(FRAMEPACE_SOURCE_WIDTH    1920 "Source frame width in pixels (8-bit luma)")
_framepace_geometry_option(FRAMEPACE_SOURCE_HEIGHT   1080 "Source frame height in pixels (8-bit luma)")
_framepace_geometry_option(FRAMEPACE_DOWNSAMPLE_LOG2 3    "Model input = source / 2^N on each axis (0-6)")
_framepace_geometry_option(FRAMEPACE_MAX_FEATURES    32   "Floats per inference result")

foreach(_v FRAMEPACE_SOURCE_WIDTH FRAMEPACE_SOURCE_HEIGHT
           FRAMEPACE_DOWNSAMPLE_LOG2 FRAMEPACE_MAX_FEATURES)
  if(NOT ${_v} MATCHES "^[0-9]+$")
    message(FATAL_ERROR "${_v} must be a non-negative integer, got '${${_v}}'")
  endif()
endforeach()
if(FRAMEPACE_DOWNSAMPLE_LOG2 GREATER 6)
  message(FATAL_ERROR "FRAMEPACE_DOWNSAMPLE_LOG2 must be 0-6, got ${FRAMEPACE_DOWNSAMPLE_LOG2}")
endif()
if(FRAMEPACE_MAX_FEATURES LESS 1)
  message(FATAL_ERROR "FRAMEPACE_MAX_FEATURES must be at least 1")
endif()
math(EXPR _fp_factor "1 << ${FRAMEPACE_DOWNSAMPLE_LOG2}")
math(EXPR _fp_wrem "${FRAMEPACE_SOURCE_WIDTH} % ${_fp_factor}")
math(EXPR _fp_hrem "${FRAMEPACE_SOURCE_HEIGHT} % ${_fp_factor}")
if(FRAMEPACE_SOURCE_WIDTH LESS _fp_factor OR FRAMEPACE_SOURCE_HEIGHT LESS _fp_factor
   OR NOT _fp_wrem EQUAL 0 OR NOT _fp_hrem EQUAL 0)
  message(FATAL_ERROR
    "Source ${FRAMEPACE_SOURCE_WIDTH}x${FRAMEPACE_SOURCE_HEIGHT} is not divisible by "
    "${_fp_factor} (2^FRAMEPACE_DOWNSAMPLE_LOG2) on both axes. Pick a resolution "
    "that is, or lower FRAMEPACE_DOWNSAMPLE_LOG2.")
endif()
math(EXPR _fp_tensor_w "${FRAMEPACE_SOURCE_WIDTH} / ${_fp_factor}")
math(EXPR _fp_tensor_h "${FRAMEPACE_SOURCE_HEIGHT} / ${_fp_factor}")

if(FRAMEPACE_IS_TOP_LEVEL)
  option(ENABLE_TSAN  "Build with ThreadSanitizer (alias)"            OFF)
  option(ENABLE_ASAN  "Build with AddressSanitizer + LSan (alias)"    OFF)
  option(ENABLE_UBSAN "Build with UndefinedBehaviorSanitizer (alias)" OFF)
  if(ENABLE_TSAN)
    set(FRAMEPACE_ENABLE_TSAN ON)
  endif()
  if(ENABLE_ASAN)
    set(FRAMEPACE_ENABLE_ASAN ON)
  endif()
  if(ENABLE_UBSAN)
    set(FRAMEPACE_ENABLE_UBSAN ON)
  endif()
endif()

if(FRAMEPACE_ENABLE_TSAN AND FRAMEPACE_ENABLE_ASAN)
  message(FATAL_ERROR
    "ENABLE_TSAN and ENABLE_ASAN are mutually exclusive -- ThreadSanitizer and "
    "AddressSanitizer both replace the allocator and cannot be combined. "
    "Configure two build directories instead.")
endif()

# ---------------------------------------------------------------------------
# Build type
# ---------------------------------------------------------------------------
# `cmake -B build-tsan -DENABLE_TSAN=ON` (as documented in the README) passes no
# build type. An unset CMAKE_BUILD_TYPE means *no* optimisation flags at all,
# which for a sanitizer build also means unusable stack traces. Pick a sane
# default rather than silently producing a -O0 build.
# These all mutate project-wide state, so they are applied only when framepace
# is the project being built. An embedded build inherits the parent's standard,
# build type, and output layout instead -- the library only requires C++17,
# which it declares per-target via target_compile_features.
if(FRAMEPACE_IS_TOP_LEVEL)
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    if(FRAMEPACE_ENABLE_TSAN OR FRAMEPACE_ENABLE_ASAN OR FRAMEPACE_ENABLE_UBSAN)
      set(_default_build_type RelWithDebInfo)
    else()
      set(_default_build_type Release)
    endif()
    message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to ${_default_build_type}")
    set(CMAKE_BUILD_TYPE "${_default_build_type}" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
      STRINGS Debug Release RelWithDebInfo MinSizeRel)
  endif()

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)

  # Keeps clangd / IDE diagnostics in sync with the real include paths.
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

  if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    # The README documents `./build-tsan/test_spsc` and `./build/frame_bench`,
    # i.e. binaries directly in the build root, not in a subdirectory.
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  endif()
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Shared compile/link settings
# ---------------------------------------------------------------------------
# Two flag sets, split by who they are allowed to reach:
#
#   framepace_flags       warnings, -Werror, optimisation level, -mtune. Applied
#                         to framepace's OWN targets only (library, tests,
#                         benchmark). Never propagated: a consumer linking
#                         framepace::framepace must not have our -Wold-style-cast
#                         -Werror imposed on its own code.
#
#   framepace_sanitizers  -fsanitize compile and link flags. Propagated PUBLIC,
#                         because a sanitizer-instrumented static library fails
#                         to link unless the final executable links the runtime
#                         too. Empty in a normal build.
add_library(framepace_flags INTERFACE)
add_library(framepace_sanitizers INTERFACE)

if(MSVC)
  target_compile_options(framepace_flags INTERFACE /W4 /permissive- /Zc:preprocessor)
  if(FRAMEPACE_WERROR)
    target_compile_options(framepace_flags INTERFACE /WX)
  endif()
else()
  target_compile_options(framepace_flags INTERFACE
    -Wall -Wextra -Wpedantic
    -Wshadow
    -Wconversion -Wsign-conversion
    -Wcast-qual
    -Wold-style-cast
    -Wnon-virtual-dtor
    -Wdouble-promotion
    # The hot path must not hide an unchecked [[nodiscard]] -- try_push and
    # write_slot both signal backpressure through their return value.
    -Wunused-result)
  if(FRAMEPACE_WERROR)
    target_compile_options(framepace_flags INTERFACE -Werror)
  endif()
endif()

# Release tuning. No -ffast-math: the extrapolation relies on well-defined
# behaviour for the degenerate cases (zero span, non-finite guards) that
# -ffast-math explicitly licenses the compiler to break.
if(NOT MSVC)
  target_compile_options(framepace_flags INTERFACE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:RelWithDebInfo>:-O2>
    $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-fno-omit-frame-pointer>)
endif()

if(FRAMEPACE_NATIVE_ARCH AND NOT MSVC)
  # -march=native is unsupported by Apple clang on arm64; -mtune is portable
  # across both and does not change the baseline ISA (so NEON/SSSE3 selection
  # in downsample.cpp stays predictable).
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mtune=native" FRAMEPACE_HAS_MTUNE_NATIVE)
  if(FRAMEPACE_HAS_MTUNE_NATIVE)
    target_compile_options(framepace_flags INTERFACE -mtune=native)
  endif()
endif()

# --- sanitizers ------------------------------------------------------------
set(_sanitizers "")
if(FRAMEPACE_ENABLE_TSAN)
  list(APPEND _sanitizers thread)
endif()
if(FRAMEPACE_ENABLE_ASAN)
  list(APPEND _sanitizers address)
endif()
if(FRAMEPACE_ENABLE_UBSAN)
  list(APPEND _sanitizers undefined)
endif()

if(_sanitizers)
  if(MSVC)
    message(FATAL_ERROR "Sanitizer builds are not wired up for MSVC")
  endif()
  list(JOIN _sanitizers "," _sanitizer_list)
  message(STATUS "Sanitizers enabled: ${_sanitizer_list}")
  target_compile_options(framepace_sanitizers INTERFACE
    -fsanitize=${_sanitizer_list}
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls  # keeps sanitizer stack traces readable
    -g)
  # Must be on the link line too, or the runtime is never pulled in.
  target_link_options(framepace_sanitizers INTERFACE -fsanitize=${_sanitizer_list})
endif()

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
add_library(framepace STATIC
  src/downsample.cpp
  src/pacer.cpp
  src/inference_worker.cpp)

target_include_directories(framepace PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# C++17 is a genuine requirement of the public headers, so it propagates.
target_compile_features(framepace PUBLIC cxx_std_17)

target_compile_definitions(framepace PUBLIC
  FRAMEPACE_SOURCE_WIDTH=${FRAMEPACE_SOURCE_WIDTH}
  FRAMEPACE_SOURCE_HEIGHT=${FRAMEPACE_SOURCE_HEIGHT}
  FRAMEPACE_DOWNSAMPLE_LOG2=${FRAMEPACE_DOWNSAMPLE_LOG2}
  FRAMEPACE_MAX_FEATURES=${FRAMEPACE_MAX_FEATURES})

# BUILD_INTERFACE keeps framepace_flags out of the install export entirely: a
# static library's PRIVATE dependencies are otherwise exported as LINK_ONLY
# entries, which would require exporting -- and so propagating -- the flags.
target_link_libraries(framepace
  PUBLIC  Threads::Threads framepace_sanitizers
  PRIVATE $<BUILD_INTERFACE:framepace_flags>)

set_target_properties(framepace PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  EXPORT_NAME framepace)

# Namespaced alias, so `framepace::framepace` links identically whether the
# project was embedded via add_subdirectory or found via find_package. Without
# it, consumers have to special-case the two.
add_library(framepace::framepace ALIAS framepace)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(FRAMEPACE_BUILD_TESTS)
  enable_testing()

  set(_framepace_tests
    test_spsc        # the ThreadSanitizer target
    test_downsample
    test_pacer
    test_no_alloc)

  foreach(_test IN LISTS _framepace_tests)
    add_executable(${_test} tests/${_test}.cpp)
    target_link_libraries(${_test} PRIVATE framepace framepace_flags)
    target_include_directories(${_test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_test(NAME ${_test} COMMAND ${_test})
  endforeach()

  # test_no_alloc replaces global operator new to count allocations.
  # AddressSanitizer installs its own allocator hooks, and ASan also reports
  # the intentional allocation in the probe's positive-control test as a leak.
  if(FRAMEPACE_ENABLE_ASAN)
    set_tests_properties(test_no_alloc PROPERTIES
      ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
  endif()

  # Concurrency tests under TSan are ~5-15x slower than native.
  set_tests_properties(test_spsc test_pacer PROPERTIES TIMEOUT 600)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(FRAMEPACE_BUILD_BENCHMARKS)
  add_executable(frame_bench benchmarks/frame_bench.cpp)
  target_link_libraries(frame_bench PRIVATE framepace framepace_flags)
endif()

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(FRAMEPACE_BUILD_EXAMPLES)
  add_executable(integration_example examples/integration_example.cpp)
  target_link_libraries(integration_example PRIVATE framepace framepace_flags)
endif()

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
# Top-level only. `format` and `format-check` are deliberately unprefixed
# because they are the conventional names, but those are exactly the names a parent
# project is likely to have already -- defining them while embedded is a hard
# configure error, not a warning.
if(FRAMEPACE_IS_TOP_LEVEL)
  find_program(CLANG_FORMAT_EXE NAMES clang-format)
  if(CLANG_FORMAT_EXE)
    file(GLOB_RECURSE _framepace_sources
      ${CMAKE_CURRENT_SOURCE_DIR}/include/framepace/*.hpp
      ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hpp
      ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp)

    add_custom_target(format
      COMMAND ${CLANG_FORMAT_EXE} -i ${_framepace_sources}
      COMMENT "Formatting sources with clang-format"
      VERBATIM)

    add_custom_target(format-check
      COMMAND ${CLANG_FORMAT_EXE} --dry-run -Werror ${_framepace_sources}
      COMMENT "Checking formatting"
      VERBATIM)
  endif()
endif()

# ---------------------------------------------------------------------------
# Install / export
# ---------------------------------------------------------------------------
# Makes `find_package(framepace)` work after `cmake --install`, which is the
# other half of being consumable: add_subdirectory covers vendoring, this
# covers a system or staged install.
if(FRAMEPACE_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  # framepace_sanitizers is a PUBLIC dependency of the exported target, so it
  # is exported with it. framepace_flags is deliberately not: see above.
  install(TARGETS framepace framepace_sanitizers
    EXPORT framepaceTargets
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  install(DIRECTORY include/framepace
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp")

  install(EXPORT framepaceTargets
    FILE framepaceTargets.cmake
    NAMESPACE framepace::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/framepace)

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/framepaceConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

  # Generated rather than shipped as a file: it needs the Threads dependency
  # re-found on the consumer's side, since the exported target references it.
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/framepaceConfig.cmake"
"include(CMakeFindDependencyMacro)
find_dependency(Threads)
include(\"\${CMAKE_CURRENT_LIST_DIR}/framepaceTargets.cmake\")
")

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/framepaceConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/framepaceConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/framepace)
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if(FRAMEPACE_IS_TOP_LEVEL)
  message(STATUS "")
  message(STATUS "framepace ${PROJECT_VERSION}")
  message(STATUS "  build type   : ${CMAKE_BUILD_TYPE}")
  message(STATUS "  compiler     : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
  message(STATUS "  target arch  : ${CMAKE_SYSTEM_PROCESSOR}")
  message(STATUS "  sanitizers   : ${_sanitizer_list}")
  message(STATUS "  geometry     : ${FRAMEPACE_SOURCE_WIDTH}x${FRAMEPACE_SOURCE_HEIGHT} -> ${_fp_tensor_w}x${_fp_tensor_h}, ${FRAMEPACE_MAX_FEATURES} features")
  message(STATUS "  tests        : ${FRAMEPACE_BUILD_TESTS}")
  message(STATUS "  benchmarks   : ${FRAMEPACE_BUILD_BENCHMARKS}")
  message(STATUS "  examples     : ${FRAMEPACE_BUILD_EXAMPLES}")
  message(STATUS "")
endif()
