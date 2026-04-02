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

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <string>
#include <memory>
#include <fstream>
#include <filesystem>
#include "aeongui/AeonGUI.hpp"
#include "aeongui/FontDatabase.hpp"
#include "aeongui/ElementFactory.hpp"
#include "aeongui/DrawType.hpp"
#include "aeongui/dom/Document.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        int ParsePathData ( std::vector<DrawType>& aPath, const char* s );
    }
}

// ---- ParsePathData ----

TEST ( ParsePathDataThreadSafety, ConcurrentParseProducesSameResults )
{
    constexpr int kThreads = 8;
    const char* pathStr = "M10 20 L30 40 C50 60 70 80 90 100 Z";

    // Parse once on the main thread to get the expected result.
    std::vector<AeonGUI::DrawType> expected;
    ASSERT_EQ ( AeonGUI::DOM::ParsePathData ( expected, pathStr ), 0 );

    std::vector<std::vector<AeonGUI::DrawType>> results ( kThreads );
    std::vector<std::thread> threads;
    threads.reserve ( kThreads );

    for ( int i = 0; i < kThreads; ++i )
    {
        threads.emplace_back ( [&results, i, pathStr]()
        {
            EXPECT_EQ ( AeonGUI::DOM::ParsePathData ( results[i], pathStr ), 0 );
        } );
    }

    for ( auto& t : threads )
    {
        t.join();
    }

    for ( int i = 0; i < kThreads; ++i )
    {
        EXPECT_EQ ( results[i].size(), expected.size() ) << "Thread " << i;
    }
}

TEST ( ParsePathDataThreadSafety, ConcurrentParseDifferentPaths )
{
    const char* pathA = "M0 0 L100 0";
    const char* pathB = "M0 0 L100 0 L100 100 Z";

    std::vector<AeonGUI::DrawType> expectedA, expectedB;
    ASSERT_EQ ( AeonGUI::DOM::ParsePathData ( expectedA, pathA ), 0 );
    ASSERT_EQ ( AeonGUI::DOM::ParsePathData ( expectedB, pathB ), 0 );

    std::vector<AeonGUI::DrawType> resultA, resultB;
    std::thread tA ( [&]()
    {
        AeonGUI::DOM::ParsePathData ( resultA, pathA );
    } );
    std::thread tB ( [&]()
    {
        AeonGUI::DOM::ParsePathData ( resultB, pathB );
    } );
    tA.join();
    tB.join();

    EXPECT_EQ ( resultA.size(), expectedA.size() );
    EXPECT_EQ ( resultB.size(), expectedB.size() );
}

TEST ( ParsePathDataThreadSafety, StressTest )
{
    constexpr int kThreads = 16;
    constexpr int kIterations = 50;
    const char* pathStr = "M0 0 L10 20 Q30 40 50 60 Z";

    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    threads.reserve ( kThreads );

    for ( int i = 0; i < kThreads; ++i )
    {
        threads.emplace_back ( [&failures, pathStr]()
        {
            for ( int j = 0; j < kIterations; ++j )
            {
                std::vector<AeonGUI::DrawType> path;
                if ( AeonGUI::DOM::ParsePathData ( path, pathStr ) != 0 )
                {
                    ++failures;
                }
            }
        } );
    }

    for ( auto& t : threads )
    {
        t.join();
    }

    EXPECT_EQ ( failures.load(), 0 );
}

// ---- ElementFactory ----

TEST ( ElementFactoryThreadSafety, ConcurrentConstructIsSafe )
{
    constexpr int kThreads = 8;
    std::vector<std::unique_ptr<AeonGUI::DOM::Element>> elements ( kThreads );
    std::vector<std::thread> threads;
    threads.reserve ( kThreads );

    for ( int i = 0; i < kThreads; ++i )
    {
        threads.emplace_back ( [&elements, i]()
        {
            AeonGUI::AttributeMap attrs;
            elements[i] = AeonGUI::Construct ( "rect", std::move ( attrs ), nullptr );
        } );
    }

    for ( auto& t : threads )
    {
        t.join();
    }

    for ( int i = 0; i < kThreads; ++i )
    {
        EXPECT_NE ( elements[i], nullptr ) << "Thread " << i;
    }
}

TEST ( ElementFactoryThreadSafety, ConcurrentEnumerateIsSafe )
{
    constexpr int kThreads = 4;
    std::vector<std::vector<std::string>> tagSets ( kThreads );
    std::vector<std::thread> threads;
    threads.reserve ( kThreads );

    for ( int i = 0; i < kThreads; ++i )
    {
        threads.emplace_back ( [&tagSets, i]()
        {
            AeonGUI::EnumerateConstructors ( [&tagSets, i] ( const AeonGUI::StringLiteral & tag )
            {
                tagSets[i].emplace_back ( tag.GetString(), tag.GetStringSize() - 1 );
                return true;
            } );
        } );
    }

    for ( auto& t : threads )
    {
        t.join();
    }

    for ( int i = 1; i < kThreads; ++i )
    {
        EXPECT_EQ ( tagSets[i].size(), tagSets[0].size() ) << "Thread " << i;
    }
}

// ---- FontDatabase ----

class FontDatabaseThreadSafety : public ::testing::Test
{
protected:
    void SetUp() override
    {
        AeonGUI::FontDatabase::Finalize();
    }
    void TearDown() override
    {
        AeonGUI::FontDatabase::Finalize();
    }
};

TEST_F ( FontDatabaseThreadSafety, ConcurrentInitializeIsSafe )
{
    constexpr int kThreads = 4;
    std::vector<std::thread> threads;
    threads.reserve ( kThreads );

    for ( int i = 0; i < kThreads; ++i )
    {
        threads.emplace_back ( []()
        {
            ASSERT_NO_THROW ( AeonGUI::FontDatabase::Initialize() );
        } );
    }

    for ( auto& t : threads )
    {
        t.join();
    }

    EXPECT_NE ( AeonGUI::FontDatabase::GetFcConfig(), nullptr );
}

TEST_F ( FontDatabaseThreadSafety, ConcurrentGetFcConfigWhileInitialized )
{
    AeonGUI::FontDatabase::Initialize();

    constexpr int kThreads = 8;
    std::vector<void*> configs ( kThreads, nullptr );
    std::vector<std::thread> threads;
    threads.reserve ( kThreads );

    for ( int i = 0; i < kThreads; ++i )
    {
        threads.emplace_back ( [&configs, i]()
        {
            configs[i] = AeonGUI::FontDatabase::GetFcConfig();
        } );
    }

    for ( auto& t : threads )
    {
        t.join();
    }

    for ( int i = 0; i < kThreads; ++i )
    {
        EXPECT_NE ( configs[i], nullptr ) << "Thread " << i;
    }
}

TEST_F ( FontDatabaseThreadSafety, ConcurrentCreateContext )
{
    AeonGUI::FontDatabase::Initialize();

    constexpr int kThreads = 4;
    std::vector<void*> contexts ( kThreads, nullptr );
    std::vector<std::thread> threads;
    threads.reserve ( kThreads );

    for ( int i = 0; i < kThreads; ++i )
    {
        threads.emplace_back ( [&contexts, i]()
        {
            auto* ctx = AeonGUI::FontDatabase::CreateContext();
            contexts[i] = ctx;
        } );
    }

    for ( auto& t : threads )
    {
        t.join();
    }

    for ( int i = 0; i < kThreads; ++i )
    {
        ASSERT_NE ( contexts[i], nullptr ) << "Thread " << i;
    }
}

// ---- Document concurrent loads ----

TEST ( DocumentThreadSafety, ConcurrentDocumentLoads )
{
    constexpr int kThreads = 4;
    const std::string svgStr =
        R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
        R"(<rect x="10" y="10" width="80" height="80"/>)"
        R"(</svg>)";

    // Write temp files (one per thread to avoid file contention).
    std::vector<std::filesystem::path> paths;
    for ( int i = 0; i < kThreads; ++i )
    {
        auto p = std::filesystem::temp_directory_path() / ( "aeongui-thread-" + std::to_string ( i ) + ".svg" );
        std::ofstream f ( p, std::ios::binary | std::ios::out );
        f << svgStr;
        paths.push_back ( p );
    }

    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    threads.reserve ( kThreads );

    for ( int i = 0; i < kThreads; ++i )
    {
        threads.emplace_back ( [&paths, &failures, i]()
        {
            try
            {
                AeonGUI::DOM::Document doc;
                doc.Load ( paths[i].string() );
            }
            catch ( ... )
            {
                ++failures;
            }
        } );
    }

    for ( auto& t : threads )
    {
        t.join();
    }

    for ( const auto& p : paths )
    {
        std::error_code ec;
        std::filesystem::remove ( p, ec );
    }

    EXPECT_EQ ( failures.load(), 0 );
}
