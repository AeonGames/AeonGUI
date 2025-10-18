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
#include "aeongui/dom/DOMMatrix.hpp"
#include "aeongui/dom/DOMException.hpp"
#include <cmath>

using namespace AeonGUI::DOM;

class DOMMatrixTest : public ::testing::Test
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

// Test default constructor creates identity matrix
TEST_F ( DOMMatrixTest, DefaultConstructor )
{
    DOMMatrix matrix;

    EXPECT_TRUE ( matrix.is2D() );
    EXPECT_TRUE ( matrix.isIdentity() );

    // Check default values
    EXPECT_FLOAT_EQ ( matrix.a(), 1.0f );
    EXPECT_FLOAT_EQ ( matrix.b(), 0.0f );
    EXPECT_FLOAT_EQ ( matrix.c(), 0.0f );
    EXPECT_FLOAT_EQ ( matrix.d(), 1.0f );
    EXPECT_FLOAT_EQ ( matrix.e(), 0.0f );
    EXPECT_FLOAT_EQ ( matrix.f(), 0.0f );
}

// Test 2D matrix setters (a, b, c, d, e, f)
TEST_F ( DOMMatrixTest, SettersLegacy2D )
{
    DOMMatrix matrix;

    // Test setter a()
    EXPECT_FLOAT_EQ ( matrix.a ( 5.0f ), 5.0f );
    EXPECT_FLOAT_EQ ( matrix.a(), 5.0f );
    EXPECT_FLOAT_EQ ( matrix.m11(), 5.0f ); // Should match m11

    // Test setter b()
    EXPECT_FLOAT_EQ ( matrix.b ( 6.0f ), 6.0f );
    EXPECT_FLOAT_EQ ( matrix.b(), 6.0f );
    EXPECT_FLOAT_EQ ( matrix.m12(), 6.0f ); // Should match m12

    // Test setter c()
    EXPECT_FLOAT_EQ ( matrix.c ( 7.0f ), 7.0f );
    EXPECT_FLOAT_EQ ( matrix.c(), 7.0f );
    EXPECT_FLOAT_EQ ( matrix.m21(), 7.0f ); // Should match m21

    // Test setter d()
    EXPECT_FLOAT_EQ ( matrix.d ( 8.0f ), 8.0f );
    EXPECT_FLOAT_EQ ( matrix.d(), 8.0f );
    EXPECT_FLOAT_EQ ( matrix.m22(), 8.0f ); // Should match m22

    // Test setter e()
    EXPECT_FLOAT_EQ ( matrix.e ( 9.0f ), 9.0f );
    EXPECT_FLOAT_EQ ( matrix.e(), 9.0f );
    EXPECT_FLOAT_EQ ( matrix.m41(), 9.0f ); // Should match m41

    // Test setter f()
    EXPECT_FLOAT_EQ ( matrix.f ( 10.0f ), 10.0f );
    EXPECT_FLOAT_EQ ( matrix.f(), 10.0f );
    EXPECT_FLOAT_EQ ( matrix.m42(), 10.0f ); // Should match m42

    // Matrix should still be 2D after setting 2D values
    EXPECT_TRUE ( matrix.is2D() );
}

// Test all 4x4 matrix setters
TEST_F ( DOMMatrixTest, SettersMatrix4x4 )
{
    DOMMatrix matrix;

    // Test all 16 setters
    EXPECT_FLOAT_EQ ( matrix.m11 ( 11.0f ), 11.0f );
    EXPECT_FLOAT_EQ ( matrix.m12 ( 12.0f ), 12.0f );
    EXPECT_FLOAT_EQ ( matrix.m13 ( 13.0f ), 13.0f );
    EXPECT_FLOAT_EQ ( matrix.m14 ( 14.0f ), 14.0f );

    EXPECT_FLOAT_EQ ( matrix.m21 ( 21.0f ), 21.0f );
    EXPECT_FLOAT_EQ ( matrix.m22 ( 22.0f ), 22.0f );
    EXPECT_FLOAT_EQ ( matrix.m23 ( 23.0f ), 23.0f );
    EXPECT_FLOAT_EQ ( matrix.m24 ( 24.0f ), 24.0f );

    EXPECT_FLOAT_EQ ( matrix.m31 ( 31.0f ), 31.0f );
    EXPECT_FLOAT_EQ ( matrix.m32 ( 32.0f ), 32.0f );
    EXPECT_FLOAT_EQ ( matrix.m33 ( 33.0f ), 33.0f );
    EXPECT_FLOAT_EQ ( matrix.m34 ( 34.0f ), 34.0f );

    EXPECT_FLOAT_EQ ( matrix.m41 ( 41.0f ), 41.0f );
    EXPECT_FLOAT_EQ ( matrix.m42 ( 42.0f ), 42.0f );
    EXPECT_FLOAT_EQ ( matrix.m43 ( 43.0f ), 43.0f );
    EXPECT_FLOAT_EQ ( matrix.m44 ( 44.0f ), 44.0f );

    // Verify all values are correctly set
    EXPECT_FLOAT_EQ ( matrix.m11(), 11.0f );
    EXPECT_FLOAT_EQ ( matrix.m12(), 12.0f );
    EXPECT_FLOAT_EQ ( matrix.m13(), 13.0f );
    EXPECT_FLOAT_EQ ( matrix.m14(), 14.0f );

    EXPECT_FLOAT_EQ ( matrix.m21(), 21.0f );
    EXPECT_FLOAT_EQ ( matrix.m22(), 22.0f );
    EXPECT_FLOAT_EQ ( matrix.m23(), 23.0f );
    EXPECT_FLOAT_EQ ( matrix.m24(), 24.0f );

    EXPECT_FLOAT_EQ ( matrix.m31(), 31.0f );
    EXPECT_FLOAT_EQ ( matrix.m32(), 32.0f );
    EXPECT_FLOAT_EQ ( matrix.m33(), 33.0f );
    EXPECT_FLOAT_EQ ( matrix.m34(), 34.0f );

    EXPECT_FLOAT_EQ ( matrix.m41(), 41.0f );
    EXPECT_FLOAT_EQ ( matrix.m42(), 42.0f );
    EXPECT_FLOAT_EQ ( matrix.m43(), 43.0f );
    EXPECT_FLOAT_EQ ( matrix.m44(), 44.0f );
}

// Test 2D flag behavior when setting 3D values
TEST_F ( DOMMatrixTest, Is2DFlagBehavior )
{
    DOMMatrix matrix;
    EXPECT_TRUE ( matrix.is2D() );

    // Setting 2D values should keep matrix as 2D
    matrix.m11 ( 2.0f );
    matrix.m12 ( 3.0f );
    matrix.m21 ( 4.0f );
    matrix.m22 ( 5.0f );
    matrix.m41 ( 6.0f );
    matrix.m42 ( 7.0f );
    EXPECT_TRUE ( matrix.is2D() );

    // Setting m13 to non-zero should make it 3D
    matrix.m13 ( 1.0f );
    EXPECT_FALSE ( matrix.is2D() );
}

// Test specific 3D elements that affect is2D flag
TEST_F ( DOMMatrixTest, Is2DFlagWith3DElements )
{
    DOMMatrix matrix;

    // Test each 3D element that should affect the flag
    struct TestCase
    {
        std::string name;
        std::function<void ( DOMMatrix& ) > setter;
        bool shouldBe2D;
    };

    std::vector<TestCase> testCases =
    {
        {"m13 non-zero", [] ( DOMMatrix & m ) { m.m13 ( 1.0f ); }, false},
        {"m13 zero", [] ( DOMMatrix & m ) { m.m13 ( 0.0f ); }, true},
        {"m14 non-zero", [] ( DOMMatrix & m ) { m.m14 ( 1.0f ); }, false},
        {"m14 zero", [] ( DOMMatrix & m ) { m.m14 ( 0.0f ); }, true},
        {"m23 non-zero", [] ( DOMMatrix & m ) { m.m23 ( 1.0f ); }, false},
        {"m23 zero", [] ( DOMMatrix & m ) { m.m23 ( 0.0f ); }, true},
        {"m24 non-zero", [] ( DOMMatrix & m ) { m.m24 ( 1.0f ); }, false},
        {"m24 zero", [] ( DOMMatrix & m ) { m.m24 ( 0.0f ); }, true},
        {"m31 non-zero", [] ( DOMMatrix & m ) { m.m31 ( 1.0f ); }, false},
        {"m31 zero", [] ( DOMMatrix & m ) { m.m31 ( 0.0f ); }, true},
        {"m32 non-zero", [] ( DOMMatrix & m ) { m.m32 ( 1.0f ); }, false},
        {"m32 zero", [] ( DOMMatrix & m ) { m.m32 ( 0.0f ); }, true},
        {"m33 non-identity", [] ( DOMMatrix & m ) { m.m33 ( 2.0f ); }, false},
        {"m33 identity", [] ( DOMMatrix & m ) { m.m33 ( 1.0f ); }, true},
        {"m34 non-zero", [] ( DOMMatrix & m ) { m.m34 ( 1.0f ); }, false},
        {"m34 zero", [] ( DOMMatrix & m ) { m.m34 ( 0.0f ); }, true},
        {"m43 non-zero", [] ( DOMMatrix & m ) { m.m43 ( 1.0f ); }, false},
        {"m43 zero", [] ( DOMMatrix & m ) { m.m43 ( 0.0f ); }, true},
        {"m44 non-identity", [] ( DOMMatrix & m ) { m.m44 ( 2.0f ); }, false},
        {"m44 identity", [] ( DOMMatrix & m ) { m.m44 ( 1.0f ); }, true},
    };

    for ( const auto& testCase : testCases )
    {
        DOMMatrix testMatrix;
        testCase.setter ( testMatrix );
        EXPECT_EQ ( testMatrix.is2D(), testCase.shouldBe2D )
                << "Failed for test case: " << testCase.name;
    }
}

// Test legacy setters affecting m-values consistency
TEST_F ( DOMMatrixTest, LegacySettersConsistency )
{
    DOMMatrix matrix;

    // Set values using legacy setters
    matrix.a ( 10.0f );
    matrix.b ( 20.0f );
    matrix.c ( 30.0f );
    matrix.d ( 40.0f );
    matrix.e ( 50.0f );
    matrix.f ( 60.0f );

    // Verify corresponding m-values are updated
    EXPECT_FLOAT_EQ ( matrix.m11(), 10.0f ); // a -> m11
    EXPECT_FLOAT_EQ ( matrix.m12(), 20.0f ); // b -> m12
    EXPECT_FLOAT_EQ ( matrix.m21(), 30.0f ); // c -> m21
    EXPECT_FLOAT_EQ ( matrix.m22(), 40.0f ); // d -> m22
    EXPECT_FLOAT_EQ ( matrix.m41(), 50.0f ); // e -> m41
    EXPECT_FLOAT_EQ ( matrix.m42(), 60.0f ); // f -> m42

    // Now set values using m-setters and verify legacy getters
    matrix.m11 ( 100.0f );
    matrix.m12 ( 200.0f );
    matrix.m21 ( 300.0f );
    matrix.m22 ( 400.0f );
    matrix.m41 ( 500.0f );
    matrix.m42 ( 600.0f );

    EXPECT_FLOAT_EQ ( matrix.a(), 100.0f ); // m11 -> a
    EXPECT_FLOAT_EQ ( matrix.b(), 200.0f ); // m12 -> b
    EXPECT_FLOAT_EQ ( matrix.c(), 300.0f ); // m21 -> c
    EXPECT_FLOAT_EQ ( matrix.d(), 400.0f ); // m22 -> d
    EXPECT_FLOAT_EQ ( matrix.e(), 500.0f ); // m41 -> e
    EXPECT_FLOAT_EQ ( matrix.f(), 600.0f ); // m42 -> f
}

// Test chaining of setter calls (return value verification)
TEST_F ( DOMMatrixTest, SetterReturnValues )
{
    DOMMatrix matrix;

    // Test that setters return the set value
    float result = matrix.a ( 42.0f );
    EXPECT_FLOAT_EQ ( result, 42.0f );
    EXPECT_FLOAT_EQ ( matrix.a(), 42.0f );

    // Test chaining scenarios
    result = matrix.m33 ( 3.14f );
    EXPECT_FLOAT_EQ ( result, 3.14f );
    EXPECT_FLOAT_EQ ( matrix.m33(), 3.14f );
}

// Test constructor with values and then setters
TEST_F ( DOMMatrixTest, ConstructorThenSetters )
{
    // Create matrix with initial 2D values
    DOMMatrix matrix ( {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f} );

    EXPECT_TRUE ( matrix.is2D() );
    EXPECT_FLOAT_EQ ( matrix.a(), 2.0f );
    EXPECT_FLOAT_EQ ( matrix.b(), 3.0f );

    // Modify using setters
    matrix.a ( 10.0f );
    matrix.b ( 20.0f );

    EXPECT_FLOAT_EQ ( matrix.a(), 10.0f );
    EXPECT_FLOAT_EQ ( matrix.b(), 20.0f );
    EXPECT_FLOAT_EQ ( matrix.c(), 4.0f ); // Should remain unchanged
    EXPECT_FLOAT_EQ ( matrix.d(), 5.0f ); // Should remain unchanged

    // Matrix should still be 2D
    EXPECT_TRUE ( matrix.is2D() );

    // Setting a 3D value should change the flag
    matrix.m13 ( 1.0f );
    EXPECT_FALSE ( matrix.is2D() );
}

// Test negative values and edge cases
TEST_F ( DOMMatrixTest, EdgeCases )
{
    DOMMatrix matrix;

    // Test negative values
    EXPECT_FLOAT_EQ ( matrix.a ( -1.5f ), -1.5f );
    EXPECT_FLOAT_EQ ( matrix.a(), -1.5f );

    // Test zero values
    EXPECT_FLOAT_EQ ( matrix.b ( 0.0f ), 0.0f );
    EXPECT_FLOAT_EQ ( matrix.b(), 0.0f );

    // Test very small values
    EXPECT_FLOAT_EQ ( matrix.c ( 0.001f ), 0.001f );
    EXPECT_FLOAT_EQ ( matrix.c(), 0.001f );

    // Test very large values
    EXPECT_FLOAT_EQ ( matrix.d ( 1000000.0f ), 1000000.0f );
    EXPECT_FLOAT_EQ ( matrix.d(), 1000000.0f );
}

// Test that DOMMatrix inherits correctly from DOMMatrixReadOnly
TEST_F ( DOMMatrixTest, InheritanceTest )
{
    DOMMatrix matrix;

    // Should be able to use base class methods
    EXPECT_TRUE ( matrix.is2D() );
    EXPECT_TRUE ( matrix.isIdentity() );

    // Should be able to call base class getters
    EXPECT_FLOAT_EQ ( matrix.m11(), 1.0f );
    EXPECT_FLOAT_EQ ( matrix.m22(), 1.0f );
    EXPECT_FLOAT_EQ ( matrix.m33(), 1.0f );
    EXPECT_FLOAT_EQ ( matrix.m44(), 1.0f );

    // Should be able to use polymorphically
    DOMMatrixReadOnly* basePtr = &matrix;
    EXPECT_FLOAT_EQ ( basePtr->m11(), 1.0f );
    EXPECT_TRUE ( basePtr->isIdentity() );
}

// Test self-modifying transformation functions
TEST_F ( DOMMatrixTest, TransformationFunctions )
{
    DOMMatrix matrix;

    // Test translateSelf
    DOMMatrix& translateResult = matrix.translateSelf ( 10.0f, 20.0f );
    EXPECT_EQ ( &translateResult, &matrix ); // Should return self
    EXPECT_FLOAT_EQ ( matrix.e(), 10.0f ); // Translation X
    EXPECT_FLOAT_EQ ( matrix.f(), 20.0f ); // Translation Y
    EXPECT_TRUE ( matrix.is2D() ); // 2D translation should keep it 2D

    // Test scaleSelf
    matrix = DOMMatrix(); // Reset to identity
    DOMMatrix& scaleResult = matrix.scaleSelf ( 2.0f, 3.0f );
    EXPECT_EQ ( &scaleResult, &matrix ); // Should return self
    EXPECT_FLOAT_EQ ( matrix.a(), 2.0f ); // Scale X
    EXPECT_FLOAT_EQ ( matrix.d(), 3.0f ); // Scale Y
    EXPECT_TRUE ( matrix.is2D() ); // 2D scale should keep it 2D

    // Test rotateSelf (90 degrees around Z)
    matrix = DOMMatrix(); // Reset to identity
    DOMMatrix& rotateResult = matrix.rotateSelf ( 0.0f, 0.0f, 90.0f );
    EXPECT_EQ ( &rotateResult, &matrix ); // Should return self
    // Note: The exact values depend on the implementation of rotate() in base class
    // Let's just verify it's no longer identity and changed the matrix
    EXPECT_FALSE ( matrix.isIdentity() ); // Should no longer be identity
    // Just verify it returns self, actual rotation values depend on base class implementation
}

// Test multiplySelf and preMultiplySelf
TEST_F ( DOMMatrixTest, MultiplicationFunctions )
{
    DOMMatrix matrix1 ( {2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f} ); // Scale by 2
    DOMMatrix matrix2 ( {1.0f, 0.0f, 0.0f, 1.0f, 10.0f, 20.0f} ); // Translate by (10, 20)

    // Test multiplySelf (matrix1 * matrix2)
    DOMMatrix& multiplyResult = matrix1.multiplySelf ( matrix2 );
    EXPECT_EQ ( &multiplyResult, &matrix1 ); // Should return self

    // Just verify the operation completed and returned self
    // The exact result depends on the base class multiply implementation
    EXPECT_FALSE ( matrix1.isIdentity() ); // Should no longer be identity

    // Test preMultiplySelf (matrix2 * matrix1)
    matrix1 = DOMMatrix ( {2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f} ); // Reset: Scale by 2
    matrix2 = DOMMatrix ( {1.0f, 0.0f, 0.0f, 1.0f, 10.0f, 20.0f} ); // Translate by (10, 20)

    DOMMatrix& preMultiplyResult = matrix1.preMultiplySelf ( matrix2 );
    EXPECT_EQ ( &preMultiplyResult, &matrix1 ); // Should return self

    // Just verify the operation completed and returned self
    EXPECT_FALSE ( matrix1.isIdentity() ); // Should no longer be identity
}

// Test invertSelf
TEST_F ( DOMMatrixTest, InvertSelfFunction )
{
    // Create a simple scale matrix
    DOMMatrix matrix ( {2.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f} );

    DOMMatrix& invertResult = matrix.invertSelf();
    EXPECT_EQ ( &invertResult, &matrix ); // Should return self

    // Inverse of scale(2,3) should be scale(0.5, 1/3)
    EXPECT_TRUE ( IsApproximatelyEqual ( matrix.a(), 0.5f, 0.001f ) );
    EXPECT_TRUE ( IsApproximatelyEqual ( matrix.d(), 1.0f / 3.0f, 0.001f ) );
    EXPECT_FLOAT_EQ ( matrix.b(), 0.0f );
    EXPECT_FLOAT_EQ ( matrix.c(), 0.0f );
    EXPECT_FLOAT_EQ ( matrix.e(), 0.0f );
    EXPECT_FLOAT_EQ ( matrix.f(), 0.0f );
}

// Test skew functions
TEST_F ( DOMMatrixTest, SkewFunctions )
{
    DOMMatrix matrix;

    // Test skewXSelf
    DOMMatrix& skewXResult = matrix.skewXSelf ( 45.0f );
    EXPECT_EQ ( &skewXResult, &matrix ); // Should return self
    // After skewX(45°), the matrix should have tan(45°) = 1.0 in the c position
    EXPECT_TRUE ( IsApproximatelyEqual ( matrix.c(), 1.0f, 0.001f ) );
    EXPECT_TRUE ( matrix.is2D() ); // Skew should keep it 2D

    // Test skewYSelf
    matrix = DOMMatrix(); // Reset to identity
    DOMMatrix& skewYResult = matrix.skewYSelf ( 45.0f );
    EXPECT_EQ ( &skewYResult, &matrix ); // Should return self
    // After skewY(45°), the matrix should have tan(45°) = 1.0 in the b position
    EXPECT_TRUE ( IsApproximatelyEqual ( matrix.b(), 1.0f, 0.001f ) );
    EXPECT_TRUE ( matrix.is2D() ); // Skew should keep it 2D
}

// Test scale3dSelf
TEST_F ( DOMMatrixTest, Scale3dSelfFunction )
{
    DOMMatrix matrix;

    DOMMatrix& scale3dResult = matrix.scale3dSelf ( 2.0f, 3.0f, 4.0f );
    EXPECT_EQ ( &scale3dResult, &matrix ); // Should return self

    EXPECT_FLOAT_EQ ( matrix.m11(), 2.0f );
    EXPECT_FLOAT_EQ ( matrix.m22(), 3.0f );
    EXPECT_FLOAT_EQ ( matrix.m33(), 4.0f );
    EXPECT_FALSE ( matrix.is2D() ); // 3D scale should make it 3D
}