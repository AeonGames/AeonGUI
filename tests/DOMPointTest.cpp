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
#include "aeongui/dom/DOMPoint.hpp"
#include <cmath>

using namespace AeonGUI::DOM;

class DOMPointTest : public ::testing::Test
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
TEST_F ( DOMPointTest, DefaultConstructor )
{
    DOMPoint point;

    EXPECT_FLOAT_EQ ( point.x(), 0.0f );
    EXPECT_FLOAT_EQ ( point.y(), 0.0f );
    EXPECT_FLOAT_EQ ( point.z(), 0.0f );
    EXPECT_FLOAT_EQ ( point.w(), 1.0f ); // w defaults to 1.0f according to DOM spec
}

// Test parameterized constructor
TEST_F ( DOMPointTest, ParameterizedConstructor )
{
    DOMPoint point ( 10.5f, 20.3f, 30.7f, 2.0f );

    EXPECT_FLOAT_EQ ( point.x(), 10.5f );
    EXPECT_FLOAT_EQ ( point.y(), 20.3f );
    EXPECT_FLOAT_EQ ( point.z(), 30.7f );
    EXPECT_FLOAT_EQ ( point.w(), 2.0f );
}

// Test partial parameter constructor (using default parameters)
TEST_F ( DOMPointTest, PartialParameterConstructor )
{
    DOMPoint point1 ( 15.0f );
    EXPECT_FLOAT_EQ ( point1.x(), 15.0f );
    EXPECT_FLOAT_EQ ( point1.y(), 0.0f );
    EXPECT_FLOAT_EQ ( point1.z(), 0.0f );
    EXPECT_FLOAT_EQ ( point1.w(), 1.0f );

    DOMPoint point2 ( 25.0f, 35.0f );
    EXPECT_FLOAT_EQ ( point2.x(), 25.0f );
    EXPECT_FLOAT_EQ ( point2.y(), 35.0f );
    EXPECT_FLOAT_EQ ( point2.z(), 0.0f );
    EXPECT_FLOAT_EQ ( point2.w(), 1.0f );

    DOMPoint point3 ( 45.0f, 55.0f, 65.0f );
    EXPECT_FLOAT_EQ ( point3.x(), 45.0f );
    EXPECT_FLOAT_EQ ( point3.y(), 55.0f );
    EXPECT_FLOAT_EQ ( point3.z(), 65.0f );
    EXPECT_FLOAT_EQ ( point3.w(), 1.0f );
}

// Test inheritance from DOMPointReadOnly - should have access to getter methods
TEST_F ( DOMPointTest, InheritanceFromDOMPointReadOnly )
{
    DOMPoint point ( 50.0f, 60.0f, 70.0f, 80.0f );

    // Test inherited getter methods (const versions)
    EXPECT_FLOAT_EQ ( point.x(), 50.0f );
    EXPECT_FLOAT_EQ ( point.y(), 60.0f );
    EXPECT_FLOAT_EQ ( point.z(), 70.0f );
    EXPECT_FLOAT_EQ ( point.w(), 80.0f );

    // Test inherited toJSON method
    DOMString json = point.toJSON();
    EXPECT_FALSE ( json.empty() );
}

// Test x setter
TEST_F ( DOMPointTest, XSetter )
{
    DOMPoint point ( 10.0f, 20.0f, 30.0f, 40.0f );

    float result = point.x ( 55.5f );

    EXPECT_FLOAT_EQ ( result, 55.5f );     // Return value should be new x
    EXPECT_FLOAT_EQ ( point.x(), 55.5f );   // x should be updated
    EXPECT_FLOAT_EQ ( point.y(), 20.0f );   // Other values unchanged
    EXPECT_FLOAT_EQ ( point.z(), 30.0f );
    EXPECT_FLOAT_EQ ( point.w(), 40.0f );
}

// Test y setter
TEST_F ( DOMPointTest, YSetter )
{
    DOMPoint point ( 10.0f, 20.0f, 30.0f, 40.0f );

    float result = point.y ( 75.25f );

    EXPECT_FLOAT_EQ ( result, 75.25f );    // Return value should be new y
    EXPECT_FLOAT_EQ ( point.y(), 75.25f );  // y should be updated
    EXPECT_FLOAT_EQ ( point.x(), 10.0f );   // Other values unchanged
    EXPECT_FLOAT_EQ ( point.z(), 30.0f );
    EXPECT_FLOAT_EQ ( point.w(), 40.0f );
}

// Test z setter
TEST_F ( DOMPointTest, ZSetter )
{
    DOMPoint point ( 10.0f, 20.0f, 30.0f, 40.0f );

    float result = point.z ( 85.125f );

    EXPECT_FLOAT_EQ ( result, 85.125f );   // Return value should be new z
    EXPECT_FLOAT_EQ ( point.z(), 85.125f ); // z should be updated
    EXPECT_FLOAT_EQ ( point.x(), 10.0f );  // Other values unchanged
    EXPECT_FLOAT_EQ ( point.y(), 20.0f );
    EXPECT_FLOAT_EQ ( point.w(), 40.0f );
}

// Test w setter
TEST_F ( DOMPointTest, WSetter )
{
    DOMPoint point ( 10.0f, 20.0f, 30.0f, 40.0f );

    float result = point.w ( 95.375f );

    EXPECT_FLOAT_EQ ( result, 95.375f );   // Return value should be new w
    EXPECT_FLOAT_EQ ( point.w(), 95.375f ); // w should be updated
    EXPECT_FLOAT_EQ ( point.x(), 10.0f );  // Other values unchanged
    EXPECT_FLOAT_EQ ( point.y(), 20.0f );
    EXPECT_FLOAT_EQ ( point.z(), 30.0f );
}

// Test negative values in setters
TEST_F ( DOMPointTest, NegativeValueSetters )
{
    DOMPoint point ( 10.0f, 20.0f, 30.0f, 40.0f );

    point.x ( -50.0f );
    point.y ( -60.0f );
    point.z ( -70.0f );
    point.w ( -80.0f );

    EXPECT_FLOAT_EQ ( point.x(), -50.0f );
    EXPECT_FLOAT_EQ ( point.y(), -60.0f );
    EXPECT_FLOAT_EQ ( point.z(), -70.0f );
    EXPECT_FLOAT_EQ ( point.w(), -80.0f );
}

// Test zero values in setters
TEST_F ( DOMPointTest, ZeroValueSetters )
{
    DOMPoint point ( 10.0f, 20.0f, 30.0f, 40.0f );

    point.x ( 0.0f );
    point.y ( 0.0f );
    point.z ( 0.0f );
    point.w ( 0.0f );

    EXPECT_FLOAT_EQ ( point.x(), 0.0f );
    EXPECT_FLOAT_EQ ( point.y(), 0.0f );
    EXPECT_FLOAT_EQ ( point.z(), 0.0f );
    EXPECT_FLOAT_EQ ( point.w(), 0.0f );
}

// Test chaining operations (modifying multiple coordinates)
TEST_F ( DOMPointTest, ChainedOperations )
{
    DOMPoint point;

    point.x ( 100.0f );
    point.y ( 200.0f );
    point.z ( 300.0f );
    point.w ( 400.0f );

    EXPECT_FLOAT_EQ ( point.x(), 100.0f );
    EXPECT_FLOAT_EQ ( point.y(), 200.0f );
    EXPECT_FLOAT_EQ ( point.z(), 300.0f );
    EXPECT_FLOAT_EQ ( point.w(), 400.0f );
}

// Test very small values
TEST_F ( DOMPointTest, VerySmallValues )
{
    DOMPoint point;

    point.x ( 0.001f );
    point.y ( 0.002f );
    point.z ( 0.003f );
    point.w ( 0.004f );

    EXPECT_TRUE ( IsApproximatelyEqual ( point.x(), 0.001f ) );
    EXPECT_TRUE ( IsApproximatelyEqual ( point.y(), 0.002f ) );
    EXPECT_TRUE ( IsApproximatelyEqual ( point.z(), 0.003f ) );
    EXPECT_TRUE ( IsApproximatelyEqual ( point.w(), 0.004f ) );
}

// Test very large values
TEST_F ( DOMPointTest, VeryLargeValues )
{
    DOMPoint point;

    point.x ( 1000000.0f );
    point.y ( 2000000.0f );
    point.z ( 3000000.0f );
    point.w ( 4000000.0f );

    EXPECT_FLOAT_EQ ( point.x(), 1000000.0f );
    EXPECT_FLOAT_EQ ( point.y(), 2000000.0f );
    EXPECT_FLOAT_EQ ( point.z(), 3000000.0f );
    EXPECT_FLOAT_EQ ( point.w(), 4000000.0f );
}

// Test fromPoint static method
TEST_F ( DOMPointTest, FromPointStaticMethod )
{
    // Create a mock point-like object
    struct MockPoint
    {
        float x() const
        {
            return 25.0f;
        }
        float y() const
        {
            return 35.0f;
        }
        float z() const
        {
            return 45.0f;
        }
        float w() const
        {
            return 55.0f;
        }
    };

    MockPoint mockPoint;
    DOMPoint point = DOMPoint::fromPoint ( mockPoint );

    EXPECT_FLOAT_EQ ( point.x(), 25.0f );
    EXPECT_FLOAT_EQ ( point.y(), 35.0f );
    EXPECT_FLOAT_EQ ( point.z(), 45.0f );
    EXPECT_FLOAT_EQ ( point.w(), 55.0f );

    // Verify the returned object is mutable (can use setters)
    point.x ( 100.0f );
    EXPECT_FLOAT_EQ ( point.x(), 100.0f );
}

// Test that setters work correctly after construction via fromPoint
TEST_F ( DOMPointTest, SettersAfterFromPoint )
{
    struct MockPoint
    {
        float x() const
        {
            return 10.0f;
        }
        float y() const
        {
            return 20.0f;
        }
        float z() const
        {
            return 30.0f;
        }
        float w() const
        {
            return 40.0f;
        }
    };

    DOMPoint point = DOMPoint::fromPoint ( MockPoint{} );

    // Modify via setters
    point.x ( 100.0f );
    point.y ( 200.0f );
    point.z ( 300.0f );
    point.w ( 400.0f );

    EXPECT_FLOAT_EQ ( point.x(), 100.0f );
    EXPECT_FLOAT_EQ ( point.y(), 200.0f );
    EXPECT_FLOAT_EQ ( point.z(), 300.0f );
    EXPECT_FLOAT_EQ ( point.w(), 400.0f );
}

// Test polymorphism (DOMPoint can be used as DOMPointReadOnly)
TEST_F ( DOMPointTest, PolymorphismWithDOMPointReadOnly )
{
    DOMPoint point ( 10.0f, 20.0f, 30.0f, 40.0f );

    // Use as DOMPointReadOnly pointer
    DOMPointReadOnly* readOnlyPtr = &point;

    // Should be able to call getter methods
    EXPECT_FLOAT_EQ ( readOnlyPtr->x(), 10.0f );
    EXPECT_FLOAT_EQ ( readOnlyPtr->y(), 20.0f );
    EXPECT_FLOAT_EQ ( readOnlyPtr->z(), 30.0f );
    EXPECT_FLOAT_EQ ( readOnlyPtr->w(), 40.0f );

    // Modify via concrete DOMPoint reference
    point.x ( 100.0f );

    // Changes should be visible through base class pointer
    EXPECT_FLOAT_EQ ( readOnlyPtr->x(), 100.0f );
}

// Test const correctness - getters should work on const objects
TEST_F ( DOMPointTest, ConstCorrectness )
{
    const DOMPoint point ( 15.0f, 25.0f, 35.0f, 45.0f );

    // All getter methods should work with const object
    EXPECT_FLOAT_EQ ( point.x(), 15.0f );
    EXPECT_FLOAT_EQ ( point.y(), 25.0f );
    EXPECT_FLOAT_EQ ( point.z(), 35.0f );
    EXPECT_FLOAT_EQ ( point.w(), 45.0f );

    DOMString json = point.toJSON();
    EXPECT_FALSE ( json.empty() );
}

// Test homogeneous coordinates behavior
TEST_F ( DOMPointTest, HomogeneousCoordinates )
{
    // Test default w=1.0 (3D point)
    DOMPoint point3D ( 10.0f, 20.0f, 30.0f );
    EXPECT_FLOAT_EQ ( point3D.w(), 1.0f );

    // Test w=0.0 (vector representation)
    DOMPoint vector ( 5.0f, 10.0f, 15.0f, 0.0f );
    EXPECT_FLOAT_EQ ( vector.w(), 0.0f );

    // Convert point to vector
    point3D.w ( 0.0f );
    EXPECT_FLOAT_EQ ( point3D.w(), 0.0f );

    // Convert vector to point
    vector.w ( 1.0f );
    EXPECT_FLOAT_EQ ( vector.w(), 1.0f );
}

// Test getter/setter method overloading
TEST_F ( DOMPointTest, GetterSetterOverloading )
{
    DOMPoint point ( 1.0f, 2.0f, 3.0f, 4.0f );

    // Test that both getter and setter work for each coordinate

    // X coordinate
    EXPECT_FLOAT_EQ ( point.x(), 1.0f );     // Getter
    EXPECT_FLOAT_EQ ( point.x ( 10.0f ), 10.0f ); // Setter (returns new value)
    EXPECT_FLOAT_EQ ( point.x(), 10.0f );    // Getter (confirms value changed)

    // Y coordinate
    EXPECT_FLOAT_EQ ( point.y(), 2.0f );     // Getter
    EXPECT_FLOAT_EQ ( point.y ( 20.0f ), 20.0f ); // Setter (returns new value)
    EXPECT_FLOAT_EQ ( point.y(), 20.0f );    // Getter (confirms value changed)

    // Z coordinate
    EXPECT_FLOAT_EQ ( point.z(), 3.0f );     // Getter
    EXPECT_FLOAT_EQ ( point.z ( 30.0f ), 30.0f ); // Setter (returns new value)
    EXPECT_FLOAT_EQ ( point.z(), 30.0f );    // Getter (confirms value changed)

    // W coordinate
    EXPECT_FLOAT_EQ ( point.w(), 4.0f );     // Getter
    EXPECT_FLOAT_EQ ( point.w ( 40.0f ), 40.0f ); // Setter (returns new value)
    EXPECT_FLOAT_EQ ( point.w(), 40.0f );    // Getter (confirms value changed)
}

// Test modification after inheritance usage
TEST_F ( DOMPointTest, ModificationAfterInheritanceUsage )
{
    DOMPoint point ( 5.0f, 10.0f, 15.0f, 20.0f );

    // Use inherited toJSON
    DOMString json1 = point.toJSON();
    EXPECT_THAT ( json1, ::testing::HasSubstr ( "5" ) );
    EXPECT_THAT ( json1, ::testing::HasSubstr ( "10" ) );

    // Modify coordinates using setters
    point.x ( 50.0f );
    point.y ( 100.0f );

    // Verify JSON reflects changes
    DOMString json2 = point.toJSON();
    EXPECT_THAT ( json2, ::testing::HasSubstr ( "50" ) );
    EXPECT_THAT ( json2, ::testing::HasSubstr ( "100" ) );

    // Verify getters also reflect changes
    EXPECT_FLOAT_EQ ( point.x(), 50.0f );
    EXPECT_FLOAT_EQ ( point.y(), 100.0f );
}

// Test floating-point precision with setters
TEST_F ( DOMPointTest, FloatingPointPrecisionWithSetters )
{
    DOMPoint point;

    // Test values that might have precision issues
    float preciseX = 1.0f / 3.0f;  // 0.333333...
    float preciseY = 2.0f / 7.0f;  // 0.285714...

    point.x ( preciseX );
    point.y ( preciseY );
    point.z ( 0.1f + 0.2f ); // 0.1 + 0.2 != 0.3 exactly

    // These should be approximately equal due to floating-point representation
    EXPECT_TRUE ( IsApproximatelyEqual ( point.x(), 1.0f / 3.0f ) );
    EXPECT_TRUE ( IsApproximatelyEqual ( point.y(), 2.0f / 7.0f ) );
    EXPECT_TRUE ( IsApproximatelyEqual ( point.z(), 0.3f, 0.0001f ) ); // More lenient epsilon
}

// Test rapid successive modifications
TEST_F ( DOMPointTest, RapidSuccessiveModifications )
{
    DOMPoint point;

    // Perform rapid successive modifications
    for ( int i = 0; i < 100; ++i )
    {
        float value = static_cast<float> ( i );
        point.x ( value );
        point.y ( value * 2 );
        point.z ( value * 3 );
        point.w ( value * 4 );

        // Verify each modification took effect
        EXPECT_FLOAT_EQ ( point.x(), value );
        EXPECT_FLOAT_EQ ( point.y(), value * 2 );
        EXPECT_FLOAT_EQ ( point.z(), value * 3 );
        EXPECT_FLOAT_EQ ( point.w(), value * 4 );
    }
}