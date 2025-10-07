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
#include "aeongui/dom/SVGLength.hpp"
#include <stdexcept>
#include <cmath>

using namespace AeonGUI::DOM;

class SVGLengthTest : public ::testing::Test
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

// Test default constructor and initial state
TEST_F ( SVGLengthTest, DefaultConstructor )
{
    SVGLength length;

    EXPECT_EQ ( length.unitType(), SVGLengthType::UNKNOWN );
    EXPECT_FLOAT_EQ ( length.value(), 0.0f );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 0.0f );
    EXPECT_EQ ( length.valueAsString(), "" );
}

// Test unitType getter
TEST_F ( SVGLengthTest, UnitTypeGetter )
{
    SVGLength length;

    length.newValueSpecifiedUnits ( SVGLengthType::PX, 10.0f );
    EXPECT_EQ ( length.unitType(), SVGLengthType::PX );

    length.newValueSpecifiedUnits ( SVGLengthType::CM, 5.0f );
    EXPECT_EQ ( length.unitType(), SVGLengthType::CM );

    length.newValueSpecifiedUnits ( SVGLengthType::PERCENTAGE, 50.0f );
    EXPECT_EQ ( length.unitType(), SVGLengthType::PERCENTAGE );
}

// Test value() getter and setter
TEST_F ( SVGLengthTest, ValueGetterSetter )
{
    SVGLength length;

    // Test setter
    float returnValue = length.value ( 100.0f );
    EXPECT_FLOAT_EQ ( returnValue, 100.0f );
    EXPECT_FLOAT_EQ ( length.value(), 100.0f );

    // Test with different values
    length.value ( 256.5f );
    EXPECT_FLOAT_EQ ( length.value(), 256.5f );

    length.value ( -50.0f );
    EXPECT_FLOAT_EQ ( length.value(), -50.0f );
}

// Test valueInSpecifiedUnits() getter and setter
TEST_F ( SVGLengthTest, ValueInSpecifiedUnitsGetterSetter )
{
    SVGLength length;

    // Test setter
    float returnValue = length.valueInSpecifiedUnits ( 25.0f );
    EXPECT_FLOAT_EQ ( returnValue, 25.0f );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 25.0f );

    // Test with different values
    length.valueInSpecifiedUnits ( 100.75f );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 100.75f );
}

// Test valueAsString parsing with valid formats
TEST_F ( SVGLengthTest, ValueAsStringValidFormats )
{
    SVGLength length;

    // Test number without unit (should default to NUMBER)
    length.valueAsString ( "100" );
    EXPECT_EQ ( length.unitType(), SVGLengthType::NUMBER );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 100.0f );

    // Test pixels
    length.valueAsString ( "50px" );
    EXPECT_EQ ( length.unitType(), SVGLengthType::PX );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 50.0f );

    // Test centimeters
    length.valueAsString ( "2.54cm" );
    EXPECT_EQ ( length.unitType(), SVGLengthType::CM );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 2.54f );

    // Test millimeters
    length.valueAsString ( "25.4mm" );
    EXPECT_EQ ( length.unitType(), SVGLengthType::MM );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 25.4f );

    // Test inches
    length.valueAsString ( "1in" );
    EXPECT_EQ ( length.unitType(), SVGLengthType::IN );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 1.0f );

    // Test points
    length.valueAsString ( "72pt" );
    EXPECT_EQ ( length.unitType(), SVGLengthType::PT );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 72.0f );

    // Test picas
    length.valueAsString ( "6pc" );
    EXPECT_EQ ( length.unitType(), SVGLengthType::PC );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 6.0f );

    // Test percentage
    length.valueAsString ( "50%" );
    EXPECT_EQ ( length.unitType(), SVGLengthType::PERCENTAGE );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 50.0f );

    // Test em units
    length.valueAsString ( "1.5em" );
    EXPECT_EQ ( length.unitType(), SVGLengthType::EMS );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 1.5f );

    // Test ex units
    length.valueAsString ( "2ex" );
    EXPECT_EQ ( length.unitType(), SVGLengthType::EXS );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 2.0f );
}

// Test valueAsString parsing with negative values
TEST_F ( SVGLengthTest, ValueAsStringNegativeValues )
{
    SVGLength length;

    length.valueAsString ( "-10px" );
    EXPECT_EQ ( length.unitType(), SVGLengthType::PX );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), -10.0f );

    length.valueAsString ( "-2.5cm" );
    EXPECT_EQ ( length.unitType(), SVGLengthType::CM );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), -2.5f );
}

// Test valueAsString parsing with decimal values
TEST_F ( SVGLengthTest, ValueAsStringDecimalValues )
{
    SVGLength length;

    length.valueAsString ( "3.14159px" );
    EXPECT_EQ ( length.unitType(), SVGLengthType::PX );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 3.14159f );

    length.valueAsString ( "0.5em" );
    EXPECT_EQ ( length.unitType(), SVGLengthType::EMS );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 0.5f );
}

// Test valueAsString parsing with invalid formats
TEST_F ( SVGLengthTest, ValueAsStringInvalidFormats )
{
    SVGLength length;

    // Test invalid formats that should throw exceptions
    EXPECT_THROW ( length.valueAsString ( "invalid" ), std::invalid_argument );
    EXPECT_THROW ( length.valueAsString ( "px10" ), std::invalid_argument );
    EXPECT_THROW ( length.valueAsString ( "10 px" ), std::invalid_argument );
    EXPECT_THROW ( length.valueAsString ( "" ), std::invalid_argument );
    EXPECT_THROW ( length.valueAsString ( "abc" ), std::invalid_argument );
    EXPECT_THROW ( length.valueAsString ( "10xyz" ), std::invalid_argument );
}

// Test newValueSpecifiedUnits with all supported unit types
TEST_F ( SVGLengthTest, NewValueSpecifiedUnits )
{
    SVGLength length;

    // Test pixels (should be same as user units)
    length.newValueSpecifiedUnits ( SVGLengthType::PX, 96.0f );
    EXPECT_EQ ( length.unitType(), SVGLengthType::PX );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 96.0f );
    EXPECT_FLOAT_EQ ( length.value(), 96.0f ); // px = user units
    EXPECT_EQ ( length.valueAsString(), "96px" );

    // Test number (should be same as user units)
    length.newValueSpecifiedUnits ( SVGLengthType::NUMBER, 50.0f );
    EXPECT_EQ ( length.unitType(), SVGLengthType::NUMBER );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 50.0f );
    EXPECT_FLOAT_EQ ( length.value(), 50.0f );
    EXPECT_EQ ( length.valueAsString(), "50" );

    // Test inches (1 inch = 96 px)
    length.newValueSpecifiedUnits ( SVGLengthType::IN, 1.0f );
    EXPECT_EQ ( length.unitType(), SVGLengthType::IN );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 1.0f );
    EXPECT_FLOAT_EQ ( length.value(), 96.0f );
    EXPECT_EQ ( length.valueAsString(), "1in" );

    // Test centimeters (1 cm = 96/2.54 px ≈ 37.795 px)
    length.newValueSpecifiedUnits ( SVGLengthType::CM, 1.0f );
    EXPECT_EQ ( length.unitType(), SVGLengthType::CM );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 1.0f );
    EXPECT_TRUE ( IsApproximatelyEqual ( length.value(), 96.0f / 2.54f ) );
    EXPECT_EQ ( length.valueAsString(), "1cm" );

    // Test millimeters (1 mm = 96/25.4 px ≈ 3.7795 px)
    length.newValueSpecifiedUnits ( SVGLengthType::MM, 1.0f );
    EXPECT_EQ ( length.unitType(), SVGLengthType::MM );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 1.0f );
    EXPECT_TRUE ( IsApproximatelyEqual ( length.value(), 96.0f / 25.4f ) );
    EXPECT_EQ ( length.valueAsString(), "1mm" );

    // Test points (1 pt = 96/72 px = 4/3 px ≈ 1.333 px)
    length.newValueSpecifiedUnits ( SVGLengthType::PT, 1.0f );
    EXPECT_EQ ( length.unitType(), SVGLengthType::PT );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 1.0f );
    EXPECT_TRUE ( IsApproximatelyEqual ( length.value(), 96.0f / 72.0f ) );
    EXPECT_EQ ( length.valueAsString(), "1pt" );

    // Test picas (1 pc = 96/6 px = 16 px)
    length.newValueSpecifiedUnits ( SVGLengthType::PC, 1.0f );
    EXPECT_EQ ( length.unitType(), SVGLengthType::PC );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 1.0f );
    EXPECT_FLOAT_EQ ( length.value(), 16.0f );
    EXPECT_EQ ( length.valueAsString(), "1pc" );

    // Test ems (1 em = 16 px by default)
    length.newValueSpecifiedUnits ( SVGLengthType::EMS, 1.0f );
    EXPECT_EQ ( length.unitType(), SVGLengthType::EMS );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 1.0f );
    EXPECT_FLOAT_EQ ( length.value(), 16.0f );
    EXPECT_EQ ( length.valueAsString(), "1em" );

    // Test exs (1 ex = 8 px by default)
    length.newValueSpecifiedUnits ( SVGLengthType::EXS, 1.0f );
    EXPECT_EQ ( length.unitType(), SVGLengthType::EXS );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 1.0f );
    EXPECT_FLOAT_EQ ( length.value(), 8.0f );
    EXPECT_EQ ( length.valueAsString(), "1ex" );

    // Test percentage
    length.newValueSpecifiedUnits ( SVGLengthType::PERCENTAGE, 50.0f );
    EXPECT_EQ ( length.unitType(), SVGLengthType::PERCENTAGE );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 50.0f );
    EXPECT_FLOAT_EQ ( length.value(), 50.0f ); // Simple assumption for now
    EXPECT_EQ ( length.valueAsString(), "50%" );
}

// Test convertToSpecifiedUnits - no conversion needed
TEST_F ( SVGLengthTest, ConvertToSpecifiedUnitsSameUnit )
{
    SVGLength length;
    length.newValueSpecifiedUnits ( SVGLengthType::PX, 100.0f );

    // Convert to same unit should not change anything
    length.convertToSpecifiedUnits ( SVGLengthType::PX );
    EXPECT_EQ ( length.unitType(), SVGLengthType::PX );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 100.0f );
    EXPECT_FLOAT_EQ ( length.value(), 100.0f );
}

// Test convertToSpecifiedUnits - px to other units
TEST_F ( SVGLengthTest, ConvertToSpecifiedUnitsFromPixels )
{
    SVGLength length;

    // Start with 96 pixels (= 1 inch)
    length.newValueSpecifiedUnits ( SVGLengthType::PX, 96.0f );

    // Convert to inches
    length.convertToSpecifiedUnits ( SVGLengthType::IN );
    EXPECT_EQ ( length.unitType(), SVGLengthType::IN );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 1.0f );
    EXPECT_FLOAT_EQ ( length.value(), 96.0f ); // Absolute value preserved

    // Reset and convert to centimeters
    length.newValueSpecifiedUnits ( SVGLengthType::PX, 96.0f );
    length.convertToSpecifiedUnits ( SVGLengthType::CM );
    EXPECT_EQ ( length.unitType(), SVGLengthType::CM );
    EXPECT_TRUE ( IsApproximatelyEqual ( length.valueInSpecifiedUnits(), 2.54f ) );
    EXPECT_FLOAT_EQ ( length.value(), 96.0f );

    // Reset and convert to millimeters
    length.newValueSpecifiedUnits ( SVGLengthType::PX, 96.0f );
    length.convertToSpecifiedUnits ( SVGLengthType::MM );
    EXPECT_EQ ( length.unitType(), SVGLengthType::MM );
    EXPECT_TRUE ( IsApproximatelyEqual ( length.valueInSpecifiedUnits(), 25.4f ) );
    EXPECT_FLOAT_EQ ( length.value(), 96.0f );

    // Reset and convert to points
    length.newValueSpecifiedUnits ( SVGLengthType::PX, 96.0f );
    length.convertToSpecifiedUnits ( SVGLengthType::PT );
    EXPECT_EQ ( length.unitType(), SVGLengthType::PT );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 72.0f );
    EXPECT_FLOAT_EQ ( length.value(), 96.0f );

    // Reset and convert to picas
    length.newValueSpecifiedUnits ( SVGLengthType::PX, 96.0f );
    length.convertToSpecifiedUnits ( SVGLengthType::PC );
    EXPECT_EQ ( length.unitType(), SVGLengthType::PC );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 6.0f );
    EXPECT_FLOAT_EQ ( length.value(), 96.0f );
}

// Test convertToSpecifiedUnits - inches to other units
TEST_F ( SVGLengthTest, ConvertToSpecifiedUnitsFromInches )
{
    SVGLength length;

    // Start with 2 inches
    length.newValueSpecifiedUnits ( SVGLengthType::IN, 2.0f );

    // Convert to pixels
    length.convertToSpecifiedUnits ( SVGLengthType::PX );
    EXPECT_EQ ( length.unitType(), SVGLengthType::PX );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 192.0f ); // 2 * 96
    EXPECT_FLOAT_EQ ( length.value(), 192.0f );

    // Reset and convert to centimeters
    length.newValueSpecifiedUnits ( SVGLengthType::IN, 2.0f );
    length.convertToSpecifiedUnits ( SVGLengthType::CM );
    EXPECT_EQ ( length.unitType(), SVGLengthType::CM );
    EXPECT_TRUE ( IsApproximatelyEqual ( length.valueInSpecifiedUnits(), 5.08f ) ); // 2 * 2.54
    EXPECT_FLOAT_EQ ( length.value(), 192.0f );
}

// Test convertToSpecifiedUnits - font-relative units
TEST_F ( SVGLengthTest, ConvertToSpecifiedUnitsFontRelative )
{
    SVGLength length;

    // Start with 32 pixels (= 2em at default 16px font size)
    length.newValueSpecifiedUnits ( SVGLengthType::PX, 32.0f );

    // Convert to ems
    length.convertToSpecifiedUnits ( SVGLengthType::EMS );
    EXPECT_EQ ( length.unitType(), SVGLengthType::EMS );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 2.0f );
    EXPECT_FLOAT_EQ ( length.value(), 32.0f );

    // Reset and convert to exs
    length.newValueSpecifiedUnits ( SVGLengthType::PX, 16.0f );
    length.convertToSpecifiedUnits ( SVGLengthType::EXS );
    EXPECT_EQ ( length.unitType(), SVGLengthType::EXS );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 2.0f ); // 16px / 8px per ex
    EXPECT_FLOAT_EQ ( length.value(), 16.0f );
}

// Test convertToSpecifiedUnits - percentage
TEST_F ( SVGLengthTest, ConvertToSpecifiedUnitsPercentage )
{
    SVGLength length;

    // Start with 50 pixels
    length.newValueSpecifiedUnits ( SVGLengthType::PX, 50.0f );

    // Convert to percentage (assuming simple 1:1 mapping for now)
    length.convertToSpecifiedUnits ( SVGLengthType::PERCENTAGE );
    EXPECT_EQ ( length.unitType(), SVGLengthType::PERCENTAGE );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 50.0f );
    EXPECT_FLOAT_EQ ( length.value(), 50.0f );
}

// Test convertToSpecifiedUnits - round trip conversions
TEST_F ( SVGLengthTest, ConvertToSpecifiedUnitsRoundTrip )
{
    SVGLength length;

    // Start with 1 inch, convert to cm, then back to inches
    length.newValueSpecifiedUnits ( SVGLengthType::IN, 1.0f );
    float originalValue = length.value();

    length.convertToSpecifiedUnits ( SVGLengthType::CM );
    length.convertToSpecifiedUnits ( SVGLengthType::IN );

    EXPECT_EQ ( length.unitType(), SVGLengthType::IN );
    EXPECT_TRUE ( IsApproximatelyEqual ( length.valueInSpecifiedUnits(), 1.0f, 0.01f ) );
    EXPECT_TRUE ( IsApproximatelyEqual ( length.value(), originalValue, 0.01f ) );
}

// Test convertToSpecifiedUnits - multiple conversions preserve absolute value
TEST_F ( SVGLengthTest, ConvertToSpecifiedUnitsPreservesAbsoluteValue )
{
    SVGLength length;

    // Start with a specific pixel value
    length.newValueSpecifiedUnits ( SVGLengthType::PX, 144.0f );
    float originalValue = length.value();

    // Convert through several units
    length.convertToSpecifiedUnits ( SVGLengthType::IN );
    EXPECT_TRUE ( IsApproximatelyEqual ( length.value(), originalValue, 0.01f ) );

    length.convertToSpecifiedUnits ( SVGLengthType::CM );
    EXPECT_TRUE ( IsApproximatelyEqual ( length.value(), originalValue, 0.01f ) );

    length.convertToSpecifiedUnits ( SVGLengthType::PT );
    EXPECT_TRUE ( IsApproximatelyEqual ( length.value(), originalValue, 0.01f ) );

    length.convertToSpecifiedUnits ( SVGLengthType::PC );
    EXPECT_TRUE ( IsApproximatelyEqual ( length.value(), originalValue, 0.01f ) );

    length.convertToSpecifiedUnits ( SVGLengthType::PX );
    EXPECT_TRUE ( IsApproximatelyEqual ( length.value(), originalValue, 0.01f ) );
}

// Test string representation updates
TEST_F ( SVGLengthTest, StringRepresentationUpdates )
{
    SVGLength length;

    // Test that string representation updates with unit changes
    length.newValueSpecifiedUnits ( SVGLengthType::PX, 100.0f );
    EXPECT_EQ ( length.valueAsString(), "100px" );

    length.convertToSpecifiedUnits ( SVGLengthType::IN );
    EXPECT_TRUE ( IsApproximatelyEqual ( std::stof ( length.valueAsString().substr ( 0, length.valueAsString().length() - 2 ) ),
                                         100.0f / 96.0f, 0.01f ) );
    EXPECT_TRUE ( length.valueAsString().ends_with ( "in" ) );

    length.convertToSpecifiedUnits ( SVGLengthType::PERCENTAGE );
    EXPECT_TRUE ( length.valueAsString().ends_with ( "%" ) );
}

// Test edge cases
TEST_F ( SVGLengthTest, EdgeCases )
{
    SVGLength length;

    // Test zero values
    length.newValueSpecifiedUnits ( SVGLengthType::PX, 0.0f );
    EXPECT_FLOAT_EQ ( length.value(), 0.0f );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 0.0f );

    length.convertToSpecifiedUnits ( SVGLengthType::CM );
    EXPECT_FLOAT_EQ ( length.value(), 0.0f );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 0.0f );

    // Test negative values
    length.newValueSpecifiedUnits ( SVGLengthType::PX, -50.0f );
    EXPECT_FLOAT_EQ ( length.value(), -50.0f );
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), -50.0f );

    length.convertToSpecifiedUnits ( SVGLengthType::IN );
    EXPECT_TRUE ( IsApproximatelyEqual ( length.valueInSpecifiedUnits(), -50.0f / 96.0f ) );
    EXPECT_FLOAT_EQ ( length.value(), -50.0f );

    // Test very small values
    length.newValueSpecifiedUnits ( SVGLengthType::PX, 0.001f );
    length.convertToSpecifiedUnits ( SVGLengthType::IN );
    EXPECT_TRUE ( IsApproximatelyEqual ( length.value(), 0.001f, 0.0001f ) );

    // Test very large values
    length.newValueSpecifiedUnits ( SVGLengthType::PX, 10000.0f );
    length.convertToSpecifiedUnits ( SVGLengthType::IN );
    EXPECT_TRUE ( IsApproximatelyEqual ( length.value(), 10000.0f, 0.1f ) );
}

// Test unknown unit type behavior
TEST_F ( SVGLengthTest, UnknownUnitType )
{
    SVGLength length;

    // Set a known value first
    length.newValueSpecifiedUnits ( SVGLengthType::PX, 100.0f );

    // Try to convert to unknown unit type - should not change anything
    length.convertToSpecifiedUnits ( SVGLengthType::UNKNOWN );
    EXPECT_EQ ( length.unitType(), SVGLengthType::PX ); // Should remain unchanged
    EXPECT_FLOAT_EQ ( length.valueInSpecifiedUnits(), 100.0f );
    EXPECT_FLOAT_EQ ( length.value(), 100.0f );
}
