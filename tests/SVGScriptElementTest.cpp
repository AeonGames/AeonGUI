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
#include <fstream>
#include <filesystem>
#include "aeongui/dom/Document.hpp"
#include "aeongui/dom/Event.hpp"
#include "aeongui/CairoCanvas.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace
{

    // Get a function from the test_plugin library at runtime
    // (the plugin is loaded by SVGScriptElement, but we also load it
    //  to read the test flags)
    class PluginProbe
    {
    public:
        PluginProbe()
        {
            auto libPath = GetPluginPath();
#ifdef _WIN32
            mLib = ::LoadLibraryA ( libPath.string().c_str() );
#else
            mLib = dlopen ( libPath.string().c_str(), RTLD_LAZY | RTLD_NOLOAD );
            if ( !mLib )
            {
                mLib = dlopen ( libPath.string().c_str(), RTLD_LAZY );
            }
#endif
        }
        ~PluginProbe()
        {
            if ( mLib )
            {
#ifdef _WIN32
                ::FreeLibrary ( mLib );
#else
                dlclose ( mLib );
#endif
            }
        }

        int GetLoadCalled()
        {
            return CallGetter ( "TestPlugin_GetLoadCalled" );
        }
        int GetUnloadCalled()
        {
            return CallGetter ( "TestPlugin_GetUnloadCalled" );
        }
        int GetClickCount()
        {
            return CallGetter ( "TestPlugin_GetClickCount" );
        }
        bool IsLoaded() const
        {
            return mLib != nullptr;
        }

    private:
        static std::filesystem::path GetPluginPath()
        {
#ifdef _WIN32
            // Get the directory of the current executable
            char buf[MAX_PATH];
            GetModuleFileNameA ( nullptr, buf, MAX_PATH );
            auto dir = std::filesystem::path ( buf ).parent_path();
            return dir / "test_plugin.dll";
#elif __APPLE__
            auto dir = std::filesystem::canonical ( "/proc/self/exe" ).parent_path();
            return dir / "libtest_plugin.dylib";
#else
            auto dir = std::filesystem::canonical ( "/proc/self/exe" ).parent_path();
            return dir / "libtest_plugin.so";
#endif
        }

        int CallGetter ( const char* name )
        {
            if ( !mLib )
            {
                return -1;
            }
            using GetterFunc = int ( * ) ();
#ifdef _WIN32
            auto fn = reinterpret_cast<GetterFunc> ( ::GetProcAddress ( mLib, name ) );
#else
            auto fn = reinterpret_cast<GetterFunc> ( dlsym ( mLib, name ) );
#endif
            return fn ? fn() : -1;
        }

#ifdef _WIN32
        HMODULE mLib {nullptr};
#else
        void* mLib {nullptr};
#endif
    };
}

TEST ( SVGScriptElementTest, ScriptElementConstructedForScriptTag )
{
    // Just test that <script> is parsed and doesn't crash
    const auto svgStr = R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
                        R"(<script type="native" href="nonexistent"/>)"
                        R"(<rect x="10" y="10" width="80" height="80" fill="#999"/>)"
                        R"(</svg>)";

    // Write to temp dir (plugin won't exist, but should not crash)
    std::filesystem::path path = std::filesystem::temp_directory_path() / "script-test.svg";
    {
        std::ofstream file ( path, std::ios::binary | std::ios::out );
        file << svgStr;
    }

    AeonGUI::DOM::Document document;
    ASSERT_NO_THROW ( document.Load ( path.string() ) );

    std::error_code ec;
    std::filesystem::remove ( path, ec );
}

TEST ( SVGScriptElementTest, NativePluginLoadsAndCallsOnLoad )
{
    const auto svgStr = R"(<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">)"
                        R"(<script type="native" href="test_plugin"/>)"
                        R"(<rect id="testBtn" x="50" y="50" width="100" height="100" fill="#999"/>)"
                        R"(</svg>)";

    // Write SVG next to the plugin DLL (in the bin directory where unit-tests runs)
#ifdef _WIN32
    char buf[MAX_PATH];
    GetModuleFileNameA ( nullptr, buf, MAX_PATH );
    auto binDir = std::filesystem::path ( buf ).parent_path();
#else
    auto binDir = std::filesystem::canonical ( "/proc/self/exe" ).parent_path();
#endif
    auto svgPath = binDir / "plugin-load-test.svg";
    {
        std::ofstream file ( svgPath, std::ios::binary | std::ios::out );
        file << svgStr;
    }

    {
        AeonGUI::DOM::Document document;
        document.Load ( svgPath.string() );

        PluginProbe probe;
        ASSERT_TRUE ( probe.IsLoaded() ) << "test_plugin library not found";
        EXPECT_EQ ( probe.GetLoadCalled(), 1 );
        EXPECT_EQ ( probe.GetClickCount(), 0 );
    }

    std::error_code ec;
    std::filesystem::remove ( svgPath, ec );
}

TEST ( SVGScriptElementTest, PluginReceivesClickEvents )
{
    const auto svgStr = R"(<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">)"
                        R"(<script type="native" href="test_plugin"/>)"
                        R"(<rect id="testBtn" x="50" y="50" width="100" height="100" fill="#999"/>)"
                        R"(</svg>)";

#ifdef _WIN32
    char buf[MAX_PATH];
    GetModuleFileNameA ( nullptr, buf, MAX_PATH );
    auto binDir = std::filesystem::path ( buf ).parent_path();
#else
    auto binDir = std::filesystem::canonical ( "/proc/self/exe" ).parent_path();
#endif
    auto svgPath = binDir / "plugin-click-test.svg";
    {
        std::ofstream file ( svgPath, std::ios::binary | std::ios::out );
        file << svgStr;
    }

    {
        AeonGUI::DOM::Document document;
        document.Load ( svgPath.string() );

        auto* target = document.getElementById ( "testBtn" );
        ASSERT_NE ( target, nullptr );

        PluginProbe probe;
        ASSERT_TRUE ( probe.IsLoaded() );
        EXPECT_EQ ( probe.GetClickCount(), 0 );

        // Dispatch click events
        AeonGUI::DOM::Event click1 ( "click" );
        target->dispatchEvent ( click1 );
        EXPECT_EQ ( probe.GetClickCount(), 1 );

        AeonGUI::DOM::Event click2 ( "click" );
        target->dispatchEvent ( click2 );
        EXPECT_EQ ( probe.GetClickCount(), 2 );
    }

    std::error_code ec;
    std::filesystem::remove ( svgPath, ec );
}

TEST ( SVGScriptElementTest, OnUnloadCalledOnDocumentDestruction )
{
    const auto svgStr = R"(<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">)"
                        R"(<script type="native" href="test_plugin"/>)"
                        R"(<rect id="testBtn" x="50" y="50" width="100" height="100" fill="#999"/>)"
                        R"(</svg>)";

#ifdef _WIN32
    char buf[MAX_PATH];
    GetModuleFileNameA ( nullptr, buf, MAX_PATH );
    auto binDir = std::filesystem::path ( buf ).parent_path();
#else
    auto binDir = std::filesystem::canonical ( "/proc/self/exe" ).parent_path();
#endif
    auto svgPath = binDir / "plugin-unload-test.svg";
    {
        std::ofstream file ( svgPath, std::ios::binary | std::ios::out );
        file << svgStr;
    }

    // Load the probe FIRST so the DLL stays in memory (extra ref count)
    // after SVGScriptElement's FreeLibrary call.
    PluginProbe probe;

    {
        AeonGUI::DOM::Document document;
        document.Load ( svgPath.string() );
        ASSERT_TRUE ( probe.IsLoaded() );
    }

    // After document destruction, OnUnload should have been called.
    EXPECT_EQ ( probe.GetUnloadCalled(), 1 );

    std::error_code ec;
    std::filesystem::remove ( svgPath, ec );
}

TEST ( SVGScriptElementTest, NonNativeScriptTypeIsIgnored )
{
    // script without type="native" should be silently ignored
    const auto svgStr = R"(<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">)"
                        R"(<script type="text/javascript">var x = 1;</script>)"
                        R"(<rect x="10" y="10" width="80" height="80" fill="#999"/>)"
                        R"(</svg>)";

    std::filesystem::path path = std::filesystem::temp_directory_path() / "script-js-test.svg";
    {
        std::ofstream file ( path, std::ios::binary | std::ios::out );
        file << svgStr;
    }

    AeonGUI::DOM::Document document;
    ASSERT_NO_THROW ( document.Load ( path.string() ) );

    std::error_code ec;
    std::filesystem::remove ( path, ec );
}

TEST ( SVGScriptElementTest, GetAttributeReturnsParsedAttributes )
{
    const auto svgStr = R"(<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">)"
                        R"(<script type="native" href="test_plugin"/>)"
                        R"(<rect id="testBtn" x="50" y="50" width="100" height="100" fill="#999"/>)"
                        R"(</svg>)";

#ifdef _WIN32
    char buf[MAX_PATH];
    GetModuleFileNameA ( nullptr, buf, MAX_PATH );
    auto binDir = std::filesystem::path ( buf ).parent_path();
#else
    auto binDir = std::filesystem::canonical ( "/proc/self/exe" ).parent_path();
#endif
    auto svgPath = binDir / "plugin-getattr-test.svg";
    {
        std::ofstream file ( svgPath, std::ios::binary | std::ios::out );
        file << svgStr;
    }

    {
        AeonGUI::DOM::Document document;
        document.Load ( svgPath.string() );

        auto* elem = document.getElementById ( "testBtn" );
        ASSERT_NE ( elem, nullptr );

        // Verify attributes exist via the C++ API (same path the C API uses)
        const auto& attrs = elem->attributes();
        auto it = attrs.find ( "fill" );
        ASSERT_NE ( it, attrs.end() );
        EXPECT_EQ ( it->second, "#999" );
    }

    std::error_code ec;
    std::filesystem::remove ( svgPath, ec );
}
