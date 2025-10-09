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
#include "aeongui/dom/DOMRect.hpp"
#include <cmath>

using namespace AeonGUI::DOM;

class DOMRectTest : public ::testing::Test
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
TEST_F ( DOMRectTest, DefaultConstructor )
{
    DOMRect rect;

    EXPECT_FLOAT_EQ ( rect.x(), 0.0f );
    EXPECT_FLOAT_EQ ( rect.y(), 0.0f );
    EXPECT_FLOAT_EQ ( rect.width(), 0.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 0.0f );
}

// Test parameterized constructor
TEST_F ( DOMRectTest, ParameterizedConstructor )
{
    DOMRect rect ( 10.5f, 20.3f, 100.0f, 200.0f );

    EXPECT_FLOAT_EQ ( rect.x(), 10.5f );
    EXPECT_FLOAT_EQ ( rect.y(), 20.3f );
    EXPECT_FLOAT_EQ ( rect.width(), 100.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 200.0f );
}

// Test inheritance from DOMRectReadOnly - should have access to getter methods
TEST_F ( DOMRectTest, InheritanceFromDOMRectReadOnly )
{
    DOMRect rect ( 50.0f, 60.0f, 70.0f, 80.0f );

    // Test inherited getter methods
    EXPECT_FLOAT_EQ ( rect.x(), 50.0f );
    EXPECT_FLOAT_EQ ( rect.y(), 60.0f );
    EXPECT_FLOAT_EQ ( rect.width(), 70.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 80.0f );

    // Test inherited computed properties
    EXPECT_FLOAT_EQ ( rect.top(), 60.0f );
    EXPECT_FLOAT_EQ ( rect.left(), 50.0f );
    EXPECT_FLOAT_EQ ( rect.right(), 120.0f ); // 50 + 70
    EXPECT_FLOAT_EQ ( rect.bottom(), 140.0f ); // 60 + 80

    // Test inherited toJSON method
    DOMString json = rect.toJSON();
    EXPECT_FALSE ( json.empty() );
}

// Test x setter
TEST_F ( DOMRectTest, XSetter )
{
    DOMRect rect ( 10.0f, 20.0f, 30.0f, 40.0f );

    float result = rect.x ( 55.5f );

    EXPECT_FLOAT_EQ ( result, 55.5f );     // Return value should be new x
    EXPECT_FLOAT_EQ ( rect.x(), 55.5f );   // x should be updated
    EXPECT_FLOAT_EQ ( rect.y(), 20.0f );   // Other values unchanged
    EXPECT_FLOAT_EQ ( rect.width(), 30.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 40.0f );
}

// Test y setter
TEST_F ( DOMRectTest, YSetter )
{
    DOMRect rect ( 10.0f, 20.0f, 30.0f, 40.0f );

    float result = rect.y ( 75.25f );

    EXPECT_FLOAT_EQ ( result, 75.25f );    // Return value should be new y
    EXPECT_FLOAT_EQ ( rect.y(), 75.25f );  // y should be updated
    EXPECT_FLOAT_EQ ( rect.x(), 10.0f );   // Other values unchanged
    EXPECT_FLOAT_EQ ( rect.width(), 30.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 40.0f );
}

// Test width setter
TEST_F ( DOMRectTest, WidthSetter )
{
    DOMRect rect ( 10.0f, 20.0f, 30.0f, 40.0f );

    float result = rect.width ( 150.0f );

    EXPECT_FLOAT_EQ ( result, 150.0f );    // Return value should be new width
    EXPECT_FLOAT_EQ ( rect.width(), 150.0f ); // width should be updated
    EXPECT_FLOAT_EQ ( rect.x(), 10.0f );   // Other values unchanged
    EXPECT_FLOAT_EQ ( rect.y(), 20.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 40.0f );
}

// Test height setter
TEST_F ( DOMRectTest, HeightSetter )
{
    DOMRect rect ( 10.0f, 20.0f, 30.0f, 40.0f );

    float result = rect.height ( 250.0f );

    EXPECT_FLOAT_EQ ( result, 250.0f );     // Return value should be new height
    EXPECT_FLOAT_EQ ( rect.height(), 250.0f ); // height should be updated
    EXPECT_FLOAT_EQ ( rect.x(), 10.0f );    // Other values unchanged
    EXPECT_FLOAT_EQ ( rect.y(), 20.0f );
    EXPECT_FLOAT_EQ ( rect.width(), 30.0f );
}

// Test top setter (should modify y)
TEST_F ( DOMRectTest, TopSetter )
{
    DOMRect rect ( 10.0f, 20.0f, 30.0f, 40.0f );

    float result = rect.top ( 100.0f );

    EXPECT_FLOAT_EQ ( result, 100.0f );    // Return value should be new top
    EXPECT_FLOAT_EQ ( rect.top(), 100.0f ); // top should be updated
    EXPECT_FLOAT_EQ ( rect.y(), 100.0f );  // y should be updated to match top
    EXPECT_FLOAT_EQ ( rect.x(), 10.0f );   // Other values unchanged
    EXPECT_FLOAT_EQ ( rect.width(), 30.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 40.0f );
}

// Test left setter (should modify x)
TEST_F ( DOMRectTest, LeftSetter )
{
    DOMRect rect ( 10.0f, 20.0f, 30.0f, 40.0f );

    float result = rect.left ( 200.0f );

    EXPECT_FLOAT_EQ ( result, 200.0f );    // Return value should be new left
    EXPECT_FLOAT_EQ ( rect.left(), 200.0f ); // left should be updated
    EXPECT_FLOAT_EQ ( rect.x(), 200.0f );  // x should be updated to match left
    EXPECT_FLOAT_EQ ( rect.y(), 20.0f );   // Other values unchanged
    EXPECT_FLOAT_EQ ( rect.width(), 30.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 40.0f );
}

// Test right setter (should modify x based on width)
TEST_F ( DOMRectTest, RightSetter )
{
    DOMRect rect ( 10.0f, 20.0f, 30.0f, 40.0f );
    // Initial right = x + width = 10 + 30 = 40

    float result = rect.right ( 100.0f );

    EXPECT_FLOAT_EQ ( result, 70.0f );     // Return value should be new x (right - width = 100 - 30 = 70)
    EXPECT_FLOAT_EQ ( rect.right(), 100.0f ); // right should be updated
    EXPECT_FLOAT_EQ ( rect.x(), 70.0f );   // x should be updated (right - width = 100 - 30 = 70)
    EXPECT_FLOAT_EQ ( rect.y(), 20.0f );   // Other values unchanged
    EXPECT_FLOAT_EQ ( rect.width(), 30.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 40.0f );
}

// Test bottom setter (should modify y based on height)
TEST_F ( DOMRectTest, BottomSetter )
{
    DOMRect rect ( 10.0f, 20.0f, 30.0f, 40.0f );
    // Initial bottom = y + height = 20 + 40 = 60

    float result = rect.bottom ( 150.0f );

    EXPECT_FLOAT_EQ ( result, 110.0f );     // Return value should be new y (bottom - height = 150 - 40 = 110)
    EXPECT_FLOAT_EQ ( rect.bottom(), 150.0f ); // bottom should be updated
    EXPECT_FLOAT_EQ ( rect.y(), 110.0f );   // y should be updated (bottom - height = 150 - 40 = 110)
    EXPECT_FLOAT_EQ ( rect.x(), 10.0f );    // Other values unchanged
    EXPECT_FLOAT_EQ ( rect.width(), 30.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 40.0f );
}

// Test negative values in setters
TEST_F ( DOMRectTest, NegativeValueSetters )
{
    DOMRect rect ( 10.0f, 20.0f, 30.0f, 40.0f );

    rect.x ( -50.0f );
    rect.y ( -60.0f );
    rect.width ( -70.0f );
    rect.height ( -80.0f );

    EXPECT_FLOAT_EQ ( rect.x(), -50.0f );
    EXPECT_FLOAT_EQ ( rect.y(), -60.0f );
    EXPECT_FLOAT_EQ ( rect.width(), -70.0f );
    EXPECT_FLOAT_EQ ( rect.height(), -80.0f );
}

// Test chaining operations (modifying multiple properties)
TEST_F ( DOMRectTest, ChainedOperations )
{
    DOMRect rect;

    rect.x ( 100.0f );
    rect.y ( 200.0f );
    rect.width ( 300.0f );
    rect.height ( 400.0f );

    EXPECT_FLOAT_EQ ( rect.x(), 100.0f );
    EXPECT_FLOAT_EQ ( rect.y(), 200.0f );
    EXPECT_FLOAT_EQ ( rect.width(), 300.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 400.0f );

    // Verify computed properties are correct
    EXPECT_FLOAT_EQ ( rect.top(), 200.0f );
    EXPECT_FLOAT_EQ ( rect.left(), 100.0f );
    EXPECT_FLOAT_EQ ( rect.right(), 400.0f ); // 100 + 300
    EXPECT_FLOAT_EQ ( rect.bottom(), 600.0f ); // 200 + 400
}

// Test edge case: zero dimensions
TEST_F ( DOMRectTest, ZeroDimensions )
{
    DOMRect rect ( 50.0f, 60.0f, 0.0f, 0.0f );

    EXPECT_FLOAT_EQ ( rect.width(), 0.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 0.0f );
    EXPECT_FLOAT_EQ ( rect.right(), 50.0f ); // x + 0 = 50
    EXPECT_FLOAT_EQ ( rect.bottom(), 60.0f ); // y + 0 = 60

    // Test setters with zero dimensions
    rect.width ( 0.0f );
    rect.height ( 0.0f );
    EXPECT_FLOAT_EQ ( rect.width(), 0.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 0.0f );
}

// Test setting computed properties with zero width/height
TEST_F ( DOMRectTest, ComputedPropertiesWithZeroDimensions )
{
    DOMRect rect ( 10.0f, 20.0f, 0.0f, 0.0f );

    // Setting right with zero width
    rect.right ( 100.0f );
    EXPECT_FLOAT_EQ ( rect.x(), 100.0f );   // right - width = 100 - 0 = 100
    EXPECT_FLOAT_EQ ( rect.right(), 100.0f );

    // Setting bottom with zero height
    rect.bottom ( 200.0f );
    EXPECT_FLOAT_EQ ( rect.y(), 200.0f );   // bottom - height = 200 - 0 = 200
    EXPECT_FLOAT_EQ ( rect.bottom(), 200.0f );
}

// Test very large values
TEST_F ( DOMRectTest, VeryLargeValues )
{
    DOMRect rect;

    rect.x ( 1000000.0f );
    rect.y ( 2000000.0f );
    rect.width ( 3000000.0f );
    rect.height ( 4000000.0f );

    EXPECT_FLOAT_EQ ( rect.x(), 1000000.0f );
    EXPECT_FLOAT_EQ ( rect.y(), 2000000.0f );
    EXPECT_FLOAT_EQ ( rect.width(), 3000000.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 4000000.0f );
    EXPECT_FLOAT_EQ ( rect.right(), 4000000.0f ); // 1000000 + 3000000
    EXPECT_FLOAT_EQ ( rect.bottom(), 6000000.0f ); // 2000000 + 4000000
}

// Test very small values
TEST_F ( DOMRectTest, VerySmallValues )
{
    DOMRect rect;

    rect.x ( 0.001f );
    rect.y ( 0.002f );
    rect.width ( 0.003f );
    rect.height ( 0.004f );

    EXPECT_TRUE ( IsApproximatelyEqual ( rect.x(), 0.001f ) );
    EXPECT_TRUE ( IsApproximatelyEqual ( rect.y(), 0.002f ) );
    EXPECT_TRUE ( IsApproximatelyEqual ( rect.width(), 0.003f ) );
    EXPECT_TRUE ( IsApproximatelyEqual ( rect.height(), 0.004f ) );
}

// Test fromRect static method
TEST_F ( DOMRectTest, FromRectStaticMethod )
{
    // Create a mock rect-like object
    struct MockRect
    {
        float x() const
        {
            return 25.0f;
        }
        float y() const
        {
            return 35.0f;
        }
        float width() const
        {
            return 45.0f;
        }
        float height() const
        {
            return 55.0f;
        }
    };

    MockRect mockRect;
    DOMRect rect = DOMRect::fromRect ( mockRect );

    EXPECT_FLOAT_EQ ( rect.x(), 25.0f );
    EXPECT_FLOAT_EQ ( rect.y(), 35.0f );
    EXPECT_FLOAT_EQ ( rect.width(), 45.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 55.0f );

    // Verify the returned object is mutable (can use setters)
    rect.x ( 100.0f );
    EXPECT_FLOAT_EQ ( rect.x(), 100.0f );
}

// Test that setters work correctly after construction via fromRect
TEST_F ( DOMRectTest, SettersAfterFromRect )
{
    struct MockRect
    {
        float x() const
        {
            return 10.0f;
        }
        float y() const
        {
            return 20.0f;
        }
        float width() const
        {
            return 30.0f;
        }
        float height() const
        {
            return 40.0f;
        }
    };

    DOMRect rect = DOMRect::fromRect ( MockRect{} );

    // Modify via setters
    rect.width ( 100.0f );
    rect.height ( 200.0f );
    rect.right ( 500.0f ); // Should set x to 400 (500 - 100)

    EXPECT_FLOAT_EQ ( rect.x(), 400.0f );   // right - width = 500 - 100
    EXPECT_FLOAT_EQ ( rect.y(), 20.0f );    // unchanged
    EXPECT_FLOAT_EQ ( rect.width(), 100.0f ); // changed
    EXPECT_FLOAT_EQ ( rect.height(), 200.0f ); // changed
    EXPECT_FLOAT_EQ ( rect.right(), 500.0f ); // as set
}

// Test polymorphism (DOMRect can be used as DOMRectReadOnly)
TEST_F ( DOMRectTest, PolymorphismWithDOMRectReadOnly )
{
    DOMRect rect ( 10.0f, 20.0f, 30.0f, 40.0f );

    // Use as DOMRectReadOnly pointer
    DOMRectReadOnly* readOnlyPtr = &rect;

    // Should be able to call getter methods
    EXPECT_FLOAT_EQ ( readOnlyPtr->x(), 10.0f );
    EXPECT_FLOAT_EQ ( readOnlyPtr->y(), 20.0f );
    EXPECT_FLOAT_EQ ( readOnlyPtr->width(), 30.0f );
    EXPECT_FLOAT_EQ ( readOnlyPtr->height(), 40.0f );

    // Should be able to call inherited methods
    EXPECT_FLOAT_EQ ( readOnlyPtr->top(), 20.0f );
    EXPECT_FLOAT_EQ ( readOnlyPtr->left(), 10.0f );
    EXPECT_FLOAT_EQ ( readOnlyPtr->right(), 40.0f );
    EXPECT_FLOAT_EQ ( readOnlyPtr->bottom(), 60.0f );

    // Modify via concrete DOMRect reference
    rect.x ( 100.0f );

    // Changes should be visible through base class pointer
    EXPECT_FLOAT_EQ ( readOnlyPtr->x(), 100.0f );
}

// Test const correctness - getters should work on const objects
TEST_F ( DOMRectTest, ConstCorrectness )
{
    const DOMRect rect ( 15.0f, 25.0f, 35.0f, 45.0f );

    // All getter methods should work with const object
    EXPECT_FLOAT_EQ ( rect.x(), 15.0f );
    EXPECT_FLOAT_EQ ( rect.y(), 25.0f );
    EXPECT_FLOAT_EQ ( rect.width(), 35.0f );
    EXPECT_FLOAT_EQ ( rect.height(), 45.0f );
    EXPECT_FLOAT_EQ ( rect.top(), 25.0f );
    EXPECT_FLOAT_EQ ( rect.left(), 15.0f );
    EXPECT_FLOAT_EQ ( rect.right(), 50.0f );
    EXPECT_FLOAT_EQ ( rect.bottom(), 70.0f );

    DOMString json = rect.toJSON();
    EXPECT_FALSE ( json.empty() );
}