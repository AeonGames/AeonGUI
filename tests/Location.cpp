/*
Copyright (C) 2025 Rodrigo Jose Hernandez Cordoba

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
#include <gmock/gmock.h>
#include "aeongui/dom/Location.hpp"

TEST ( LocationTest, Assign )
{
    AeonGUI::DOM::Location location;
    EXPECT_NO_THROW ( location.assign ( "https://example.com/" ) );
    EXPECT_EQ ( location.href(), "https://example.com/" );
    EXPECT_EQ ( location.origin(), "https://example.com" );
    EXPECT_EQ ( location.protocol(), "https:" );
    EXPECT_EQ ( location.host(), "example.com" );
    EXPECT_EQ ( location.hostname(), "example.com" );
    EXPECT_EQ ( location.port(), "" );
    EXPECT_EQ ( location.pathname(), "/" );
    EXPECT_EQ ( location.search(), "" );
    EXPECT_EQ ( location.hash(), "" );

    EXPECT_NO_THROW ( location.assign ( "https://example.org:8080/foo/bar?q=baz#bang" ) );
    EXPECT_EQ ( location.href(), "https://example.org:8080/foo/bar?q=baz#bang" );
    EXPECT_EQ ( location.origin(), "https://example.org:8080" );
    EXPECT_EQ ( location.protocol(), "https:" );
    EXPECT_EQ ( location.host(), "example.org:8080" );
    EXPECT_EQ ( location.hostname(), "example.org" );
    EXPECT_EQ ( location.port(), "8080" );
    EXPECT_EQ ( location.pathname(), "/foo/bar" );
    EXPECT_EQ ( location.search(), "?q=baz" );
    EXPECT_EQ ( location.hash(), "#bang" );

    EXPECT_NO_THROW ( location.assign ( "about:blank" ) );
    EXPECT_EQ ( location.href(), "about:blank" );
    EXPECT_EQ ( location.origin(), "about:blank" );
    EXPECT_EQ ( location.protocol(), "about:" );
    EXPECT_EQ ( location.host(), "blank" );
    EXPECT_EQ ( location.hostname(), "blank" );
    EXPECT_EQ ( location.port(), "" );
    EXPECT_EQ ( location.pathname(), "" );
    EXPECT_EQ ( location.search(), "" );
    EXPECT_EQ ( location.hash(), "" );

    EXPECT_NO_THROW ( location.assign ( "file:///AeonGUI/images/tiger-style.svg" ) );
    EXPECT_EQ ( location.href(), "file:///AeonGUI/images/tiger-style.svg" );
    EXPECT_EQ ( location.origin(), "file://" );
    EXPECT_EQ ( location.protocol(), "file:" );
    EXPECT_EQ ( location.host(), "" );
    EXPECT_EQ ( location.hostname(), "" );
    EXPECT_EQ ( location.port(), "" );
    EXPECT_EQ ( location.pathname(), "/AeonGUI/images/tiger-style.svg" );
    EXPECT_EQ ( location.search(), "" );
    EXPECT_EQ ( location.hash(), "" );

    EXPECT_NO_THROW ( location.assign ( "file:///C:/AeonGUI/images/tiger-style.svg" ) );
    EXPECT_EQ ( location.href(), "file:///C:/AeonGUI/images/tiger-style.svg" );
    EXPECT_EQ ( location.origin(), "file://" );
    EXPECT_EQ ( location.protocol(), "file:" );
    EXPECT_EQ ( location.host(), "" );
    EXPECT_EQ ( location.hostname(), "" );
    EXPECT_EQ ( location.port(), "" );
    EXPECT_EQ ( location.pathname(), "/C:/AeonGUI/images/tiger-style.svg" );
    EXPECT_EQ ( location.search(), "" );
    EXPECT_EQ ( location.hash(), "" );
}

class MockLocationCallback
{
public:
    MOCK_METHOD ( void, Call, ( const AeonGUI::DOM::Location& ) );
};

TEST ( LocationTest, SetCallback )
{
    MockLocationCallback mockCallback;
    AeonGUI::DOM::Location location;
    EXPECT_CALL ( mockCallback, Call ( testing::Ref ( location ) ) ).Times ( 1 );
    location.SetCallback ( std::bind ( &MockLocationCallback::Call, &mockCallback, std::placeholders::_1 ) );
    EXPECT_NO_THROW ( location.assign ( "https://example.com/" ) );
}