cmake_minimum_required(VERSION 3.18)
project(adapTQ CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
if(MSVC)
  add_compile_options(/O2 /W3 /wd4996 /wd4068)
else()
  add_compile_options(-O3 -Wall -Wextra -Wno-unused-parameter)
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
set(CORE_SOURCES
    core/fwht.cpp
    core/codebook.cpp
    core/quantizer.cpp
    core/adaptq_c_api.cpp
    core/adaptq_backend_vtable.cpp
    cache/ring_buffer.cpp
    attention/attention.cpp
    utils/timer.cpp
)

add_library(adapTQ_core STATIC ${CORE_SOURCES})
target_include_directories(adapTQ_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
  target_compile_options(adapTQ_core PRIVATE /openmp /arch:AVX2)
else()
  target_compile_options(adapTQ_core PRIVATE -mavx2 -mfma -fopenmp)
  target_link_options(adapTQ_core PUBLIC -fopenmp)
endif()

if(UNIX AND NOT APPLE)
  target_link_libraries(adapTQ_core PUBLIC m)
endif()

# ---------------------------------------------------------------------------
# Main demo executable
# ---------------------------------------------------------------------------
add_executable(adapTQ_demo main.cpp)
target_link_libraries(adapTQ_demo PRIVATE adapTQ_core)
target_include_directories(adapTQ_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ---------------------------------------------------------------------------
# Tests — Catch2 via find_package first, then FetchContent fallback
# ---------------------------------------------------------------------------
option(ADAPTQ_BUILD_TESTS "Build unit and conformance tests" ON)

if(ADAPTQ_BUILD_TESTS)
  # Try system Catch2 (apt/brew install catch2)
  find_package(Catch2 3 QUIET)

  if(NOT Catch2_FOUND)
    # Fallback: download Catch2 at configure time via FetchContent
    message(STATUS "Catch2 not found via find_package — using FetchContent")
    include(FetchContent)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.7.1
      GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Catch2)
  endif()

  enable_testing()
  include(CTest)
  include(Catch)

  # --- test_fwht -----------------------------------------------------------
  add_executable(test_fwht
      tests/unit/test_fwht.cpp
      core/fwht.cpp)
  target_include_directories(test_fwht PRIVATE include)
  target_link_libraries(test_fwht PRIVATE Catch2::Catch2WithMain)
  catch_discover_tests(test_fwht)

  # --- test_packing --------------------------------------------------------
  add_executable(test_packing
      tests/unit/test_packing.cpp)
  target_link_libraries(test_packing PRIVATE adapTQ_core Catch2::Catch2WithMain)
  catch_discover_tests(test_packing)

  # --- test_storage --------------------------------------------------------
  # Storage backends are header-only (inline classes in storage/*.h).
  # The .cpp stubs are intentionally NOT listed — they'd be empty TUs anyway.
  add_executable(test_storage
      tests/unit/test_storage.cpp)
  target_include_directories(test_storage PRIVATE include storage)
  target_link_libraries(test_storage PRIVATE Catch2::Catch2WithMain)
  catch_discover_tests(test_storage)

  # --- test_api ------------------------------------------------------------
  add_executable(test_api
      tests/unit/test_api.cpp)
  target_link_libraries(test_api PRIVATE adapTQ_core Catch2::Catch2WithMain)
  catch_discover_tests(test_api)

  # --- test_conformance ----------------------------------------------------
  # Strategy .cpp files are compiled inline within this single TU (not in
  # any library) to avoid ODR violations. The CMakeLists does not list them
  # as separate sources precisely because they are #included by the .cpp.
  add_executable(test_conformance
      tests/conformance/strategy_conformance.cpp)
  target_include_directories(test_conformance PRIVATE
      include storage strategies tests/conformance)
  target_link_libraries(test_conformance PRIVATE adapTQ_core Catch2::Catch2WithMain)
  if(NOT MSVC)
    target_compile_options(test_conformance PRIVATE -mavx2 -mfma)
  else()
    target_compile_options(test_conformance PRIVATE /arch:AVX2)
  endif()
  catch_discover_tests(test_conformance)

  message(STATUS "Test targets enabled")
endif()
