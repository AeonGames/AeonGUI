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
#include "aeongui/dom/DOMRectReadOnly.hpp"
#include <cmath>

using namespace AeonGUI::DOM;

class DOMRectReadOnlyTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper function to check if two floats are approximately equal
    bool IsApproximatelyEqual ( float a, float b, float epsilon = 0.001f ) const
    {
        return std::abs ( a - b ) < epsilon;
    }
};

// Test default constructor
TEST_F ( DOMRectReadOnlyTest, DefaultConstructor )
{
    DOMRectReadOnly rect;

    EXPECT_FLOAT_EQ ( rect.x(), 0.0f );
    EXPECT_FLOAT_EQ ( rect.y(), 0.0f );
    EXPECT_FLOAT_EQ ( rect.width(), 0.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 0.0f );
}

// Test parameterized constructor
TEST_F ( DOMRectReadOnlyTest, ParameterizedConstructor )
{
    DOMRectReadOnly rect ( 10.5f, 20.3f, 100.0f, 200.0f );

    EXPECT_FLOAT_EQ ( rect.x(), 10.5f );
    EXPECT_FLOAT_EQ ( rect.y(), 20.3f );
    EXPECT_FLOAT_EQ ( rect.width(), 100.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 200.0f );
}

// Test partial parameter constructor (using default parameters)
TEST_F ( DOMRectReadOnlyTest, PartialParameterConstructor )
{
    DOMRectReadOnly rect1 ( 10.0f );
    EXPECT_FLOAT_EQ ( rect1.x(), 10.0f );
    EXPECT_FLOAT_EQ ( rect1.y(), 0.0f );
    EXPECT_FLOAT_EQ ( rect1.width(), 0.0f );
    EXPECT_FLOAT_EQ ( rect1.height(), 0.0f );

    DOMRectReadOnly rect2 ( 10.0f, 20.0f );
    EXPECT_FLOAT_EQ ( rect2.x(), 10.0f );
    EXPECT_FLOAT_EQ ( rect2.y(), 20.0f );
    EXPECT_FLOAT_EQ ( rect2.width(), 0.0f );
    EXPECT_FLOAT_EQ ( rect2.height(), 0.0f );

    DOMRectReadOnly rect3 ( 10.0f, 20.0f, 30.0f );
    EXPECT_FLOAT_EQ ( rect3.x(), 10.0f );
    EXPECT_FLOAT_EQ ( rect3.y(), 20.0f );
    EXPECT_FLOAT_EQ ( rect3.width(), 30.0f );
    EXPECT_FLOAT_EQ ( rect3.height(), 0.0f );
}

// Test getter methods for basic properties
TEST_F ( DOMRectReadOnlyTest, BasicGetters )
{
    DOMRectReadOnly rect ( 15.5f, 25.5f, 150.0f, 250.0f );

    EXPECT_FLOAT_EQ ( rect.x(), 15.5f );
    EXPECT_FLOAT_EQ ( rect.y(), 25.5f );
    EXPECT_FLOAT_EQ ( rect.width(), 150.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 250.0f );
}

// Test computed properties: top and left should equal y and x respectively
TEST_F ( DOMRectReadOnlyTest, TopAndLeftProperties )
{
    DOMRectReadOnly rect ( 100.0f, 200.0f, 300.0f, 400.0f );

    EXPECT_FLOAT_EQ ( rect.top(), rect.y() );
    EXPECT_FLOAT_EQ ( rect.left(), rect.x() );
    EXPECT_FLOAT_EQ ( rect.top(), 200.0f );
    EXPECT_FLOAT_EQ ( rect.left(), 100.0f );
}

// Test computed properties: right and bottom
TEST_F ( DOMRectReadOnlyTest, RightAndBottomProperties )
{
    DOMRectReadOnly rect ( 10.0f, 20.0f, 100.0f, 200.0f );

    EXPECT_FLOAT_EQ ( rect.right(), rect.x() + rect.width() );
    EXPECT_FLOAT_EQ ( rect.bottom(), rect.y() + rect.height() );
    EXPECT_FLOAT_EQ ( rect.right(), 110.0f ); // 10 + 100
    EXPECT_FLOAT_EQ ( rect.bottom(), 220.0f ); // 20 + 200
}

// Test with negative values
TEST_F ( DOMRectReadOnlyTest, NegativeValues )
{
    DOMRectReadOnly rect ( -50.0f, -30.0f, 100.0f, 80.0f );

    EXPECT_FLOAT_EQ ( rect.x(), -50.0f );
    EXPECT_FLOAT_EQ ( rect.y(), -30.0f );
    EXPECT_FLOAT_EQ ( rect.width(), 100.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 80.0f );

    EXPECT_FLOAT_EQ ( rect.left(), -50.0f );
    EXPECT_FLOAT_EQ ( rect.top(), -30.0f );
    EXPECT_FLOAT_EQ ( rect.right(), 50.0f ); // -50 + 100
    EXPECT_FLOAT_EQ ( rect.bottom(), 50.0f ); // -30 + 80
}

// Test with negative width and height
TEST_F ( DOMRectReadOnlyTest, NegativeWidthAndHeight )
{
    DOMRectReadOnly rect ( 100.0f, 200.0f, -50.0f, -75.0f );

    EXPECT_FLOAT_EQ ( rect.x(), 100.0f );
    EXPECT_FLOAT_EQ ( rect.y(), 200.0f );
    EXPECT_FLOAT_EQ ( rect.width(), -50.0f );
    EXPECT_FLOAT_EQ ( rect.height(), -75.0f );

    EXPECT_FLOAT_EQ ( rect.left(), 100.0f );
    EXPECT_FLOAT_EQ ( rect.top(), 200.0f );
    EXPECT_FLOAT_EQ ( rect.right(), 50.0f ); // 100 + (-50)
    EXPECT_FLOAT_EQ ( rect.bottom(), 125.0f ); // 200 + (-75)
}

// Test toJSON method
TEST_F ( DOMRectReadOnlyTest, ToJSONMethod )
{
    DOMRectReadOnly rect ( 10.5f, 20.3f, 100.7f, 200.9f );

    DOMString json = rect.toJSON();

    // The JSON should contain all four properties
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"x\"" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"y\"" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"width\"" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"height\"" ) );

    // Should be properly formatted JSON
    EXPECT_THAT ( json, ::testing::StartsWith ( "{" ) );
    EXPECT_THAT ( json, ::testing::EndsWith ( "}" ) );
}

// Test toJSON with zero values
TEST_F ( DOMRectReadOnlyTest, ToJSONZeroValues )
{
    DOMRectReadOnly rect ( 0.0f, 0.0f, 0.0f, 0.0f );

    DOMString json = rect.toJSON();

    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"x\": 0" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"y\": 0" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"width\": 0" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"height\": 0" ) );
}

// Test toJSON with negative values
TEST_F ( DOMRectReadOnlyTest, ToJSONNegativeValues )
{
    // Use values that can be exactly represented in float to avoid precision issues
    DOMRectReadOnly rect ( -10.5f, -20.25f, -100.0f, -200.0f );

    DOMString json = rect.toJSON();

    // Should contain negative values with exact representations
    EXPECT_THAT ( json, ::testing::HasSubstr ( "-10.5" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "-20.25" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "-100" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "-200" ) );
}

// Test toJSON format precision
TEST_F ( DOMRectReadOnlyTest, ToJSONPrecision )
{
    // Test with values that would show precision differences
    DOMRectReadOnly rect ( 1.0f, 2.0f, 3.0f, 4.0f );

    DOMString json = rect.toJSON();

    // With integer values, should show as integers (not 1.0, 2.0, etc.)
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"x\": 1" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"y\": 2" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"width\": 3" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"height\": 4" ) );
}

// Test edge cases with very small values
TEST_F ( DOMRectReadOnlyTest, VerySmallValues )
{
    DOMRectReadOnly rect ( 0.001f, 0.002f, 0.003f, 0.004f );

    EXPECT_TRUE ( IsApproximatelyEqual ( rect.x(), 0.001f ) );
    EXPECT_TRUE ( IsApproximatelyEqual ( rect.y(), 0.002f ) );
    EXPECT_TRUE ( IsApproximatelyEqual ( rect.width(), 0.003f ) );
    EXPECT_TRUE ( IsApproximatelyEqual ( rect.height(), 0.004f ) );

    EXPECT_TRUE ( IsApproximatelyEqual ( rect.right(), 0.004f ) ); // 0.001 + 0.003
    EXPECT_TRUE ( IsApproximatelyEqual ( rect.bottom(), 0.006f ) ); // 0.002 + 0.004
}

// Test edge cases with very large values
TEST_F ( DOMRectReadOnlyTest, VeryLargeValues )
{
    DOMRectReadOnly rect ( 1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f );

    EXPECT_FLOAT_EQ ( rect.x(), 1000000.0f );
    EXPECT_FLOAT_EQ ( rect.y(), 2000000.0f );
    EXPECT_FLOAT_EQ ( rect.width(), 3000000.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 4000000.0f );

    EXPECT_FLOAT_EQ ( rect.right(), 4000000.0f ); // 1000000 + 3000000
    EXPECT_FLOAT_EQ ( rect.bottom(), 6000000.0f ); // 2000000 + 4000000
}

// Test fromRect static method with a mock rect-like object
TEST_F ( DOMRectReadOnlyTest, FromRectStaticMethod )
{
    // Create a mock rect-like object
    struct MockRect
    {
        float x() const
        {
            return 50.0f;
        }
        float y() const
        {
            return 60.0f;
        }
        float width() const
        {
            return 70.0f;
        }
        float height() const
        {
            return 80.0f;
        }
    };

    MockRect mockRect;
    DOMRectReadOnly rect = DOMRectReadOnly::fromRect ( mockRect );

    EXPECT_FLOAT_EQ ( rect.x(), 50.0f );
    EXPECT_FLOAT_EQ ( rect.y(), 60.0f );
    EXPECT_FLOAT_EQ ( rect.width(), 70.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 80.0f );
}

// Test const correctness - all methods should be callable on const objects
TEST_F ( DOMRectReadOnlyTest, ConstCorrectness )
{
    const DOMRectReadOnly rect ( 10.0f, 20.0f, 30.0f, 40.0f );

    // All these should compile and work with const object
    EXPECT_FLOAT_EQ ( rect.x(), 10.0f );
    EXPECT_FLOAT_EQ ( rect.y(), 20.0f );
    EXPECT_FLOAT_EQ ( rect.width(), 30.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 40.0f );
    EXPECT_FLOAT_EQ ( rect.top(), 20.0f );
    EXPECT_FLOAT_EQ ( rect.left(), 10.0f );
    EXPECT_FLOAT_EQ ( rect.right(), 40.0f );
    EXPECT_FLOAT_EQ ( rect.bottom(), 60.0f );

    DOMString json = rect.toJSON();
    EXPECT_FALSE ( json.empty() );
}