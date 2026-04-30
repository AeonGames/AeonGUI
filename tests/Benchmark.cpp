/*
Copyright (C) 2026 Rodrigo Jose Hernandez Cordoba

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// AeonGUI per-frame micro-benchmark.
//
// Standalone executable (not run by ctest). Built only when CMake
// is configured with -DBUILD_BENCHMARKS=ON. Loads images/fps.xhtml
// (or a path supplied on the command line), then runs N frames of
// AdvanceTime + Draw and reports timing statistics.
//
// Optional allocation counting: configure with
//     -DBUILD_BENCHMARKS=ON -DAEONGUI_BENCH_ALLOC_COUNT=ON
// to enable global operator new/delete overrides that count
// allocations per frame. Off by default to avoid interfering with
// vendored allocators in vcpkg-built dependencies.
//
// NOTE: on shared-library builds (MSVC DLL, MinGW DLL) each module
// has its own CRT and operator new, so these overrides only catch
// allocations made directly inside the benchmark executable. The
// counter will read ~0 because all real work happens inside the
// AeonGUI shared library. Treat the timing numbers as primary;
// alloc counts are best-effort.
//
// SOURCE_PATH is defined via -DSOURCE_PATH=... by the build system
// and points at the repository root.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "aeongui/AeonGUI.hpp"
#include "aeongui/FontDatabase.hpp"
#include "aeongui/dom/Document.hpp"
#include "aeongui/dom/Location.hpp"
#include "aeongui/dom/Window.hpp"

#ifndef SOURCE_PATH
#define SOURCE_PATH "."
#endif

#if AEONGUI_BENCH_ALLOC_COUNT
namespace
{
    std::atomic<uint64_t> gAllocCount{0};
    std::atomic<uint64_t> gAllocBytes{0};
}

void* operator new ( std::size_t aSize )
{
    gAllocCount.fetch_add ( 1, std::memory_order_relaxed );
    gAllocBytes.fetch_add ( aSize, std::memory_order_relaxed );
    void* p = std::malloc ( aSize );
    if ( !p )
    {
        throw std::bad_alloc{};
    }
    return p;
}

void* operator new[] ( std::size_t aSize )
{
    return ::operator new ( aSize );
}

void operator delete ( void* aPtr ) noexcept
{
    std::free ( aPtr );
}

void operator delete[] ( void* aPtr ) noexcept
{
    std::free ( aPtr );
}

void operator delete ( void* aPtr, std::size_t ) noexcept
{
    std::free ( aPtr );
}

void operator delete[] ( void* aPtr, std::size_t ) noexcept
{
    std::free ( aPtr );
}
#endif

namespace
{
    struct FrameStats
    {
        double mean_us;
        double min_us;
        double max_us;
        double p50_us;
        double p95_us;
        double p99_us;
        uint64_t total_allocs;
        double allocs_per_frame;
    };

    FrameStats Summarize ( std::vector<double>& aSamplesUs, uint64_t aTotalAllocs )
    {
        FrameStats stats{};
        if ( aSamplesUs.empty() )
        {
            return stats;
        }
        double sum = 0.0;
        stats.min_us = aSamplesUs.front();
        stats.max_us = aSamplesUs.front();
        for ( double s : aSamplesUs )
        {
            sum += s;
            stats.min_us = std::min ( stats.min_us, s );
            stats.max_us = std::max ( stats.max_us, s );
        }
        stats.mean_us = sum / static_cast<double> ( aSamplesUs.size() );
        std::sort ( aSamplesUs.begin(), aSamplesUs.end() );
        auto pct = [&] ( double p ) -> double
        {
            const size_t idx = std::min ( aSamplesUs.size() - 1,
                                          static_cast<size_t> ( aSamplesUs.size() * p ) );
            return aSamplesUs[idx];
        };
        stats.p50_us = pct ( 0.50 );
        stats.p95_us = pct ( 0.95 );
        stats.p99_us = pct ( 0.99 );
        stats.total_allocs = aTotalAllocs;
        stats.allocs_per_frame = static_cast<double> ( aTotalAllocs ) /
                                 static_cast<double> ( aSamplesUs.size() );
        return stats;
    }

    void PrintStats ( const char* aLabel, const FrameStats& aStats, size_t aFrameCount )
    {
        std::printf ( "  %-12s frames=%-5zu  mean=%8.2f us  min=%8.2f  p50=%8.2f  p95=%8.2f  p99=%8.2f  max=%8.2f",
                      aLabel,
                      aFrameCount,
                      aStats.mean_us, aStats.min_us, aStats.p50_us,
                      aStats.p95_us, aStats.p99_us, aStats.max_us );
#if AEONGUI_BENCH_ALLOC_COUNT
        std::printf ( "  allocs/frame=%.2f", aStats.allocs_per_frame );
#endif
        std::printf ( "\n" );
    }
}

int main ( int argc, char* argv[] )
{
    AeonGUI::Initialize ( argc, argv );
    AeonGUI::FontDatabase::Initialize();

    const std::filesystem::path defaultDoc =
        std::filesystem::path ( SOURCE_PATH ) / "images" / "fps.xhtml";
    const std::string docPath =
        ( argc > 1 ) ? std::string ( argv[1] ) : defaultDoc.generic_string();

    constexpr uint32_t kViewportWidth = 800;
    constexpr uint32_t kViewportHeight = 600;
    constexpr size_t kWarmupFrames = 50;
    constexpr size_t kMeasuredFrames = 1000;
    constexpr double kDeltaTime = 1.0 / 60.0;

    std::printf ( "AeonGUI Benchmark\n" );
    std::printf ( "  document : %s\n", docPath.c_str() );
    std::printf ( "  viewport : %ux%u\n", kViewportWidth, kViewportHeight );
    std::printf ( "  warmup   : %zu frames\n", kWarmupFrames );
    std::printf ( "  measured : %zu frames\n", kMeasuredFrames );
#if AEONGUI_BENCH_ALLOC_COUNT
    std::printf ( "  alloc-counting : ON\n" );
#else
    std::printf ( "  alloc-counting : OFF (configure with -DAEONGUI_BENCH_ALLOC_COUNT=ON)\n" );
#endif

    AeonGUI::DOM::Window window ( kViewportWidth, kViewportHeight );
    window.location() = docPath;

    // Initial draw triggers parsing + layout.
    window.Draw();

    // Pragmatic: the benchmark needs to force a full redraw every
    // frame to measure the actual draw cost. The Document* exposed
    // by Window is const; const_cast is acceptable here since this
    // is a benchmark-only file and we are not modifying the public
    // API.
    auto* doc = const_cast<AeonGUI::DOM::Document*> ( window.document() );
    if ( !doc )
    {
        std::fprintf ( stderr, "Failed to load document: %s\n", docPath.c_str() );
        AeonGUI::Finalize();
        return 1;
    }

    auto runPhase = [&] ( size_t frameCount, std::vector<double>& outSamples,
                          uint64_t& outAllocs )
    {
        outSamples.clear();
        outSamples.reserve ( frameCount );
#if AEONGUI_BENCH_ALLOC_COUNT
        const uint64_t allocsBefore = gAllocCount.load ( std::memory_order_relaxed );
#endif
        for ( size_t i = 0; i < frameCount; ++i )
        {
            doc->MarkDirty();
            const auto t0 = std::chrono::steady_clock::now();
            window.Update ( kDeltaTime );
            window.Draw();
            const auto t1 = std::chrono::steady_clock::now();
            const double us = std::chrono::duration<double, std::micro> ( t1 - t0 ).count();
            outSamples.push_back ( us );
        }
#if AEONGUI_BENCH_ALLOC_COUNT
        outAllocs = gAllocCount.load ( std::memory_order_relaxed ) - allocsBefore;
#else
        outAllocs = 0;
#endif
    };

    std::vector<double> warmupSamples;
    uint64_t warmupAllocs = 0;
    runPhase ( kWarmupFrames, warmupSamples, warmupAllocs );

    std::vector<double> samples;
    uint64_t allocs = 0;
    runPhase ( kMeasuredFrames, samples, allocs );

    auto warmupStats = Summarize ( warmupSamples, warmupAllocs );
    auto stats = Summarize ( samples, allocs );

    std::printf ( "\nResults:\n" );
    PrintStats ( "warmup",   warmupStats, kWarmupFrames );
    PrintStats ( "measured", stats,       kMeasuredFrames );

    AeonGUI::Finalize();
    return 0;
}
