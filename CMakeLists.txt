cmake_minimum_required(VERSION 3.16)
project(OptimizationFromScratch
    VERSION     0.1.0
    DESCRIPTION "Gradient Descent & SGD from scratch in C++17 with Raylib visualizations"
    LANGUAGES   CXX
)

# ─── C++17 ────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)   # -std=c++17, not -std=gnu++17

# ─── Build type default ───────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ─── Compiler warnings ────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    # Bounds checks are on in Debug, disabled in Release via NDEBUG
    add_compile_options("$<$<CONFIG:Debug>:-fsanitize=address,undefined>")
    add_link_options(   "$<$<CONFIG:Debug>:-fsanitize=address,undefined>")
endif()

# ─── Raylib ───────────────────────────────────────────────────────────────────
# First try a system-installed Raylib; if not found, fetch & build it.
find_package(raylib 4.5 QUIET)
if(NOT raylib_FOUND)
    message(STATUS "Raylib not found on system — fetching via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG        5.0
        GIT_SHALLOW    TRUE   # only the tagged commit, much faster
    )
    # Don't build Raylib examples when fetching
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(raylib)
endif()

# ─── Core library ─────────────────────────────────────────────────────────────
# All shared logic lives here so every example binary links to a single copy.
add_library(optim_core STATIC
    src/optimization.cpp
    src/data_generation.cpp
    src/visualization.cpp
)
target_include_directories(optim_core PUBLIC include)
target_link_libraries(optim_core PUBLIC raylib)

# ─── Example executables ──────────────────────────────────────────────────────
# Add new examples here — one line per binary.
set(EXAMPLES
    ex1_limit_gd
    ex2_closed_gd
    ex3_sgd
)

foreach(EX IN LISTS EXAMPLES)
    add_executable(${EX} examples/${EX}.cpp)
    target_link_libraries(${EX} PRIVATE optim_core)

    # On Linux, Raylib needs these system libs
    if(UNIX AND NOT APPLE)
        target_link_libraries(${EX} PRIVATE m dl pthread)
    endif()

    # Copy the binary to a top-level bin/ dir for easy access
    set_target_properties(${EX} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()

# ─── Install (optional, for packaging) ───────────────────────────────────────
install(TARGETS ${EXAMPLES} DESTINATION bin)

# ─── Summary ─────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "=== OptimizationFromScratch ===")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ std    : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Binaries   : ${EXAMPLES}")
message(STATUS "  Output dir : ${CMAKE_BINARY_DIR}/bin")
message(STATUS "")
