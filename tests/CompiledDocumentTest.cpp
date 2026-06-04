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
#include <string>
#include "aeongui/dom/Window.hpp"
#include "aeongui/dom/Document.hpp"
#include "aeongui/dom/Element.hpp"
#include "aeongui/dom/Event.hpp"
#include "CompiledDocumentFixture.xmlcxx.hpp"

// These tests exercise the xmlcxx-compiled-document pipeline end to end:
// the fixture XHTML is compiled to C++ by xmlcxx at build time, linked into
// the test, loaded into a host-owned Window via Window::Load, and then driven
// through the CompiledDocument host/guest contract.

TEST ( CompiledDocumentTest, BuildsDomTree )
{
    AeonGUI::DOM::Window window ( 200u, 200u );
    CompiledDocumentFixtureDocument doc;
    window.Load ( doc );

    EXPECT_EQ ( doc.window(), &window );
    ASSERT_NE ( doc.document(), nullptr );

    const AeonGUI::DOM::Element* quit = window.document()->getElementById ( "quit" );
    ASSERT_NE ( quit, nullptr );
}

TEST ( CompiledDocumentTest, ScriptSetsPropertyOnLoad )
{
    AeonGUI::DOM::Window window ( 200u, 200u );
    CompiledDocumentFixtureDocument doc;
    window.Load ( doc );

    // <body onload="Init"> binds the script-defined Init() free function,
    // which runs in OnLoad and sets this property.
    EXPECT_EQ ( doc.GetProperty ( "title" ), "Compiled Test" );
}

TEST ( CompiledDocumentTest, InlineHandlerEmitsToHostCallback )
{
    AeonGUI::DOM::Window window ( 200u, 200u );
    CompiledDocumentFixtureDocument doc;

    std::string detail;
    int count = 0;
    doc.SetCallback ( "exit", [&] ( const std::string & aDetail )
    {
        detail = aDetail;
        ++count;
    } );

    window.Load ( doc );

    AeonGUI::DOM::Element* quit = window.document()->getElementById ( "quit" );
    ASSERT_NE ( quit, nullptr );

    // Dispatching a "click" fires the onclick="RequestExit" binding, whose
    // script-defined function calls self.Emit("exit","clicked") -> the host
    // callback above.
    AeonGUI::DOM::Event event ( "click", AeonGUI::DOM::EventInit{true, true, false} );
    quit->dispatchEvent ( event );

    EXPECT_EQ ( count, 1 );
    EXPECT_EQ ( detail, "clicked" );
}

TEST ( CompiledDocumentTest, PropertyRoundTrip )
{
    CompiledDocumentFixtureDocument doc;
    EXPECT_EQ ( doc.GetProperty ( "missing" ), "" );
    doc.SetProperty ( "k", "v" );
    EXPECT_EQ ( doc.GetProperty ( "k" ), "v" );
}
