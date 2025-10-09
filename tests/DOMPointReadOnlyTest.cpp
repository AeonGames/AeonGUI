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
#include "aeongui/dom/DOMPointReadOnly.hpp"
#include <cmath>

using namespace AeonGUI::DOM;

class DOMPointReadOnlyTest : public ::testing::Test
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
TEST_F ( DOMPointReadOnlyTest, DefaultConstructor )
{
    DOMPointReadOnly point;

    EXPECT_FLOAT_EQ ( point.x(), 0.0f );
    EXPECT_FLOAT_EQ ( point.y(), 0.0f );
    EXPECT_FLOAT_EQ ( point.z(), 0.0f );
    EXPECT_FLOAT_EQ ( point.w(), 1.0f ); // w defaults to 1.0f according to DOM spec
}

// Test parameterized constructor with all parameters
TEST_F ( DOMPointReadOnlyTest, ParameterizedConstructor )
{
    DOMPointReadOnly point ( 10.5f, -20.3f, 5.7f, 1.0f );

    EXPECT_FLOAT_EQ ( point.x(), 10.5f );
    EXPECT_FLOAT_EQ ( point.y(), -20.3f );
    EXPECT_FLOAT_EQ ( point.z(), 5.7f );
    EXPECT_FLOAT_EQ ( point.w(), 1.0f );
}

// Test constructor with partial parameters (using defaults)
TEST_F ( DOMPointReadOnlyTest, PartialParameterConstructor )
{
    // Test with just x
    DOMPointReadOnly point1 ( 15.0f );
    EXPECT_FLOAT_EQ ( point1.x(), 15.0f );
    EXPECT_FLOAT_EQ ( point1.y(), 0.0f );
    EXPECT_FLOAT_EQ ( point1.z(), 0.0f );
    EXPECT_FLOAT_EQ ( point1.w(), 1.0f );

    // Test with x and y
    DOMPointReadOnly point2 ( 25.0f, 35.0f );
    EXPECT_FLOAT_EQ ( point2.x(), 25.0f );
    EXPECT_FLOAT_EQ ( point2.y(), 35.0f );
    EXPECT_FLOAT_EQ ( point2.z(), 0.0f );
    EXPECT_FLOAT_EQ ( point2.w(), 1.0f );

    // Test with x, y, and z
    DOMPointReadOnly point3 ( 45.0f, 55.0f, 65.0f );
    EXPECT_FLOAT_EQ ( point3.x(), 45.0f );
    EXPECT_FLOAT_EQ ( point3.y(), 55.0f );
    EXPECT_FLOAT_EQ ( point3.z(), 65.0f );
    EXPECT_FLOAT_EQ ( point3.w(), 1.0f );
}

// Test getter methods
TEST_F ( DOMPointReadOnlyTest, GetterMethods )
{
    DOMPointReadOnly point ( 100.25f, -200.75f, 50.5f, 2.0f );

    EXPECT_FLOAT_EQ ( point.x(), 100.25f );
    EXPECT_FLOAT_EQ ( point.y(), -200.75f );
    EXPECT_FLOAT_EQ ( point.z(), 50.5f );
    EXPECT_FLOAT_EQ ( point.w(), 2.0f );
}

// Test with negative values
TEST_F ( DOMPointReadOnlyTest, NegativeValues )
{
    DOMPointReadOnly point ( -10.5f, -20.3f, -30.7f, -0.5f );

    EXPECT_FLOAT_EQ ( point.x(), -10.5f );
    EXPECT_FLOAT_EQ ( point.y(), -20.3f );
    EXPECT_FLOAT_EQ ( point.z(), -30.7f );
    EXPECT_FLOAT_EQ ( point.w(), -0.5f );
}

// Test with zero values
TEST_F ( DOMPointReadOnlyTest, ZeroValues )
{
    DOMPointReadOnly point ( 0.0f, 0.0f, 0.0f, 0.0f );

    EXPECT_FLOAT_EQ ( point.x(), 0.0f );
    EXPECT_FLOAT_EQ ( point.y(), 0.0f );
    EXPECT_FLOAT_EQ ( point.z(), 0.0f );
    EXPECT_FLOAT_EQ ( point.w(), 0.0f );
}

// Test with mixed positive, negative, and zero values
TEST_F ( DOMPointReadOnlyTest, MixedValues )
{
    DOMPointReadOnly point ( 42.0f, -0.0f, 0.0f, -1.0f );

    EXPECT_FLOAT_EQ ( point.x(), 42.0f );
    EXPECT_FLOAT_EQ ( point.y(), 0.0f ); // -0.0f should be 0.0f
    EXPECT_FLOAT_EQ ( point.z(), 0.0f );
    EXPECT_FLOAT_EQ ( point.w(), -1.0f );
}

// Test toJSON method
TEST_F ( DOMPointReadOnlyTest, ToJSONMethod )
{
    DOMPointReadOnly point ( 10.5f, 20.25f, 30.125f, 1.0f );

    DOMString json = point.toJSON();

    // The JSON should contain all four properties
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"x\"" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"y\"" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"z\"" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"w\"" ) );

    // Should be properly formatted JSON
    EXPECT_THAT ( json, ::testing::StartsWith ( "{" ) );
    EXPECT_THAT ( json, ::testing::EndsWith ( "}" ) );

    // Should contain the actual values (with proper floating-point formatting)
    EXPECT_THAT ( json, ::testing::HasSubstr ( "10.5" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "20.25" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "30.125" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "1" ) );
}

// Test toJSON with zero values
TEST_F ( DOMPointReadOnlyTest, ToJSONZeroValues )
{
    DOMPointReadOnly point ( 0.0f, 0.0f, 0.0f, 0.0f );

    DOMString json = point.toJSON();

    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"x\": 0" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"y\": 0" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"z\": 0" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"w\": 0" ) );
}

// Test toJSON with negative values
TEST_F ( DOMPointReadOnlyTest, ToJSONNegativeValues )
{
    DOMPointReadOnly point ( -15.5f, -25.25f, -35.125f, -1.0f );

    DOMString json = point.toJSON();

    // Should contain negative values
    EXPECT_THAT ( json, ::testing::HasSubstr ( "-15.5" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "-25.25" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "-35.125" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "-1" ) );
}

// Test toJSON format precision
TEST_F ( DOMPointReadOnlyTest, ToJSONPrecision )
{
    // Test with values that would show precision differences
    DOMPointReadOnly point ( 1.0f, 2.0f, 3.0f, 4.0f );

    DOMString json = point.toJSON();

    // With integer-like values, should show as clean numbers
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"x\": 1" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"y\": 2" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"z\": 3" ) );
    EXPECT_THAT ( json, ::testing::HasSubstr ( "\"w\": 4" ) );
}

// Test edge cases with very small values
TEST_F ( DOMPointReadOnlyTest, VerySmallValues )
{
    DOMPointReadOnly point ( 0.001f, 0.002f, 0.003f, 0.004f );

    EXPECT_TRUE ( IsApproximatelyEqual ( point.x(), 0.001f ) );
    EXPECT_TRUE ( IsApproximatelyEqual ( point.y(), 0.002f ) );
    EXPECT_TRUE ( IsApproximatelyEqual ( point.z(), 0.003f ) );
    EXPECT_TRUE ( IsApproximatelyEqual ( point.w(), 0.004f ) );
}

// Test edge cases with very large values
TEST_F ( DOMPointReadOnlyTest, VeryLargeValues )
{
    DOMPointReadOnly point ( 1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f );

    EXPECT_FLOAT_EQ ( point.x(), 1000000.0f );
    EXPECT_FLOAT_EQ ( point.y(), 2000000.0f );
    EXPECT_FLOAT_EQ ( point.z(), 3000000.0f );
    EXPECT_FLOAT_EQ ( point.w(), 4000000.0f );
}

// Test fromPoint static method with a mock point-like object
TEST_F ( DOMPointReadOnlyTest, FromPointStaticMethod )
{
    // Create a mock point-like object
    struct MockPoint
    {
        float x() const
        {
            return 75.0f;
        }
        float y() const
        {
            return 85.0f;
        }
        float z() const
        {
            return 95.0f;
        }
        float w() const
        {
            return 1.5f;
        }
    };

    MockPoint mockPoint;
    DOMPointReadOnly point = DOMPointReadOnly::fromPoint ( mockPoint );

    EXPECT_FLOAT_EQ ( point.x(), 75.0f );
    EXPECT_FLOAT_EQ ( point.y(), 85.0f );
    EXPECT_FLOAT_EQ ( point.z(), 95.0f );
    EXPECT_FLOAT_EQ ( point.w(), 1.5f );
}

// Test fromPoint with different point-like object
TEST_F ( DOMPointReadOnlyTest, FromPointDifferentObject )
{
    // Create another mock point-like object with different values
    struct AnotherPoint
    {
        float x() const
        {
            return -10.0f;
        }
        float y() const
        {
            return 20.0f;
        }
        float z() const
        {
            return -30.0f;
        }
        float w() const
        {
            return 0.5f;
        }
    };

    AnotherPoint anotherPoint;
    DOMPointReadOnly point = DOMPointReadOnly::fromPoint ( anotherPoint );

    EXPECT_FLOAT_EQ ( point.x(), -10.0f );
    EXPECT_FLOAT_EQ ( point.y(), 20.0f );
    EXPECT_FLOAT_EQ ( point.z(), -30.0f );
    EXPECT_FLOAT_EQ ( point.w(), 0.5f );
}

// Test const correctness - all methods should be callable on const objects
TEST_F ( DOMPointReadOnlyTest, ConstCorrectness )
{
    const DOMPointReadOnly point ( 33.0f, 44.0f, 55.0f, 2.0f );

    // All these should compile and work with const object
    EXPECT_FLOAT_EQ ( point.x(), 33.0f );
    EXPECT_FLOAT_EQ ( point.y(), 44.0f );
    EXPECT_FLOAT_EQ ( point.z(), 55.0f );
    EXPECT_FLOAT_EQ ( point.w(), 2.0f );

    DOMString json = point.toJSON();
    EXPECT_FALSE ( json.empty() );
}

// Test default w parameter behavior (homogeneous coordinates)
TEST_F ( DOMPointReadOnlyTest, HomogeneousCoordinates )
{
    // In homogeneous coordinates, w=1.0 represents a point in 3D space
    DOMPointReadOnly point3D ( 10.0f, 20.0f, 30.0f ); // w defaults to 1.0f

    EXPECT_FLOAT_EQ ( point3D.x(), 10.0f );
    EXPECT_FLOAT_EQ ( point3D.y(), 20.0f );
    EXPECT_FLOAT_EQ ( point3D.z(), 30.0f );
    EXPECT_FLOAT_EQ ( point3D.w(), 1.0f ); // Default for 3D point

    // w=0.0 can represent a vector (direction without position)
    DOMPointReadOnly vector ( 5.0f, 10.0f, 15.0f, 0.0f );

    EXPECT_FLOAT_EQ ( vector.x(), 5.0f );
    EXPECT_FLOAT_EQ ( vector.y(), 10.0f );
    EXPECT_FLOAT_EQ ( vector.z(), 15.0f );
    EXPECT_FLOAT_EQ ( vector.w(), 0.0f ); // Vector representation
}

// Test floating-point precision edge cases
TEST_F ( DOMPointReadOnlyTest, FloatingPointPrecision )
{
    // Test values that might have precision issues
    float preciseX = 1.0f / 3.0f;  // 0.333333...
    float preciseY = 2.0f / 7.0f;  // 0.285714...

    DOMPointReadOnly point ( preciseX, preciseY, 0.1f + 0.2f, 1.0f ); // 0.1 + 0.2 != 0.3 exactly

    // These should be approximately equal due to floating-point representation
    EXPECT_TRUE ( IsApproximatelyEqual ( point.x(), 1.0f / 3.0f ) );
    EXPECT_TRUE ( IsApproximatelyEqual ( point.y(), 2.0f / 7.0f ) );
    EXPECT_TRUE ( IsApproximatelyEqual ( point.z(), 0.3f, 0.0001f ) ); // More lenient epsilon for 0.1 + 0.2
    EXPECT_FLOAT_EQ ( point.w(), 1.0f );
}