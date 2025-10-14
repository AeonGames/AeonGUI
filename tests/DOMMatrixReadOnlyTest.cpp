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
#include "aeongui/dom/DOMMatrixReadOnly.hpp"
#include "aeongui/dom/DOMPoint.hpp"
#include "aeongui/dom/DOMException.hpp"
#include <cmath>

using namespace AeonGUI::DOM;

class DOMMatrixReadOnlyTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper function to check if two floats are approximately equal
    bool IsApproximatelyEqual ( float a, float b, float epsilon = 0.001f ) const
    {
        return std::abs ( a - b ) < epsilon;
    }

    // Helper to check if matrix is approximately identity
    bool IsApproximatelyIdentity ( const DOMMatrixReadOnly& matrix ) const
    {
        return IsApproximatelyEqual ( matrix.m11(), 1.0f ) &&
               IsApproximatelyEqual ( matrix.m12(), 0.0f ) &&
               IsApproximatelyEqual ( matrix.m13(), 0.0f ) &&
               IsApproximatelyEqual ( matrix.m14(), 0.0f ) &&
               IsApproximatelyEqual ( matrix.m21(), 0.0f ) &&
               IsApproximatelyEqual ( matrix.m22(), 1.0f ) &&
               IsApproximatelyEqual ( matrix.m23(), 0.0f ) &&
               IsApproximatelyEqual ( matrix.m24(), 0.0f ) &&
               IsApproximatelyEqual ( matrix.m31(), 0.0f ) &&
               IsApproximatelyEqual ( matrix.m32(), 0.0f ) &&
               IsApproximatelyEqual ( matrix.m33(), 1.0f ) &&
               IsApproximatelyEqual ( matrix.m34(), 0.0f ) &&
               IsApproximatelyEqual ( matrix.m41(), 0.0f ) &&
               IsApproximatelyEqual ( matrix.m42(), 0.0f ) &&
               IsApproximatelyEqual ( matrix.m43(), 0.0f ) &&
               IsApproximatelyEqual ( matrix.m44(), 1.0f );
    }
};

// Test default constructor (2D identity matrix)
TEST_F ( DOMMatrixReadOnlyTest, DefaultConstructor2D )
{
    DOMMatrixReadOnly matrix;

    // Should be 2D
    EXPECT_TRUE ( matrix.is2D() );
    EXPECT_TRUE ( matrix.isIdentity() );

    // Check 2D matrix values (a, b, c, d, e, f)
    EXPECT_FLOAT_EQ ( matrix.a(), 1.0f );
    EXPECT_FLOAT_EQ ( matrix.b(), 0.0f );
    EXPECT_FLOAT_EQ ( matrix.c(), 0.0f );
    EXPECT_FLOAT_EQ ( matrix.d(), 1.0f );
    EXPECT_FLOAT_EQ ( matrix.e(), 0.0f );
    EXPECT_FLOAT_EQ ( matrix.f(), 0.0f );
}

// Test 2D constructor with initializer list
TEST_F ( DOMMatrixReadOnlyTest, Constructor2DValues )
{
    DOMMatrixReadOnly matrix ( {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f} );

    EXPECT_TRUE ( matrix.is2D() );
    EXPECT_FALSE ( matrix.isIdentity() );

    // Check 2D values
    EXPECT_FLOAT_EQ ( matrix.a(), 2.0f );
    EXPECT_FLOAT_EQ ( matrix.b(), 3.0f );
    EXPECT_FLOAT_EQ ( matrix.c(), 4.0f );
    EXPECT_FLOAT_EQ ( matrix.d(), 5.0f );
    EXPECT_FLOAT_EQ ( matrix.e(), 6.0f );
    EXPECT_FLOAT_EQ ( matrix.f(), 7.0f );

    // Check 4x4 matrix equivalents
    EXPECT_FLOAT_EQ ( matrix.m11(), 2.0f ); // a
    EXPECT_FLOAT_EQ ( matrix.m12(), 3.0f ); // b
    EXPECT_FLOAT_EQ ( matrix.m21(), 4.0f ); // c
    EXPECT_FLOAT_EQ ( matrix.m22(), 5.0f ); // d
    EXPECT_FLOAT_EQ ( matrix.m41(), 6.0f ); // e
    EXPECT_FLOAT_EQ ( matrix.m42(), 7.0f ); // f
}

// Test 3D constructor with 16 values
TEST_F ( DOMMatrixReadOnlyTest, Constructor3DValues )
{
    DOMMatrixReadOnly matrix (
    {
        1.0f, 2.0f, 3.0f, 4.0f,    // row 1
        5.0f, 6.0f, 7.0f, 8.0f,    // row 2
        9.0f, 10.0f, 11.0f, 12.0f, // row 3
        13.0f, 14.0f, 15.0f, 16.0f // row 4
    } );

    EXPECT_FALSE ( matrix.is2D() );
    EXPECT_FALSE ( matrix.isIdentity() );

    // Check all 16 values
    EXPECT_FLOAT_EQ ( matrix.m11(), 1.0f );
    EXPECT_FLOAT_EQ ( matrix.m12(), 2.0f );
    EXPECT_FLOAT_EQ ( matrix.m13(), 3.0f );
    EXPECT_FLOAT_EQ ( matrix.m14(), 4.0f );
    EXPECT_FLOAT_EQ ( matrix.m21(), 5.0f );
    EXPECT_FLOAT_EQ ( matrix.m22(), 6.0f );
    EXPECT_FLOAT_EQ ( matrix.m23(), 7.0f );
    EXPECT_FLOAT_EQ ( matrix.m24(), 8.0f );
    EXPECT_FLOAT_EQ ( matrix.m31(), 9.0f );
    EXPECT_FLOAT_EQ ( matrix.m32(), 10.0f );
    EXPECT_FLOAT_EQ ( matrix.m33(), 11.0f );
    EXPECT_FLOAT_EQ ( matrix.m34(), 12.0f );
    EXPECT_FLOAT_EQ ( matrix.m41(), 13.0f );
    EXPECT_FLOAT_EQ ( matrix.m42(), 14.0f );
    EXPECT_FLOAT_EQ ( matrix.m43(), 15.0f );
    EXPECT_FLOAT_EQ ( matrix.m44(), 16.0f );
}

// Test invalid constructor arguments
TEST_F ( DOMMatrixReadOnlyTest, ConstructorInvalidArguments )
{
    // Should throw for wrong number of arguments
    EXPECT_THROW ( DOMMatrixReadOnly ( {1.0f, 2.0f, 3.0f} ), DOMTypeMismatchError );
    EXPECT_THROW ( DOMMatrixReadOnly ( {1.0f, 2.0f, 3.0f, 4.0f, 5.0f} ), DOMTypeMismatchError );
    EXPECT_THROW ( DOMMatrixReadOnly ( {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f} ), DOMTypeMismatchError );
}

// Test identity matrix recognition
TEST_F ( DOMMatrixReadOnlyTest, IsIdentity )
{
    // Default constructor creates identity
    DOMMatrixReadOnly identity;
    EXPECT_TRUE ( identity.isIdentity() );

    // Explicit 2D identity
    DOMMatrixReadOnly identity2D ( {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f} );
    EXPECT_TRUE ( identity2D.isIdentity() );

    // Explicit 3D identity
    DOMMatrixReadOnly identity3D (
    {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    } );
    EXPECT_TRUE ( identity3D.isIdentity() );

    // Non-identity matrix
    DOMMatrixReadOnly nonIdentity ( {2.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f} );
    EXPECT_FALSE ( nonIdentity.isIdentity() );
}

// Test matrix multiplication
TEST_F ( DOMMatrixReadOnlyTest, Multiply )
{
    DOMMatrixReadOnly matrix1 ( {2.0f, 0.0f, 0.0f, 2.0f, 10.0f, 20.0f} ); // Scale by 2, translate (10,20)
    DOMMatrixReadOnly matrix2 ( {1.0f, 0.0f, 0.0f, 1.0f, 5.0f, 10.0f} ); // Translate (5,10)

    DOMMatrixReadOnly result = matrix1.multiply ( matrix2 );

    // Matrix multiplication: [2 0 10] × [1 0 5] = [2 0 15]
    //                        [0 2 20]   [0 1 10]   [0 2 30]
    //                        [0 0  1]   [0 0  1]   [0 0  1]
    EXPECT_FLOAT_EQ ( result.a(), 2.0f ); // scaling preserved
    EXPECT_FLOAT_EQ ( result.d(), 2.0f ); // scaling preserved
    EXPECT_FLOAT_EQ ( result.e(), 15.0f ); // 10 + 5 = 15
    EXPECT_FLOAT_EQ ( result.f(), 30.0f ); // 20 + 10 = 30
}

// Test translation
TEST_F ( DOMMatrixReadOnlyTest, Translate )
{
    DOMMatrixReadOnly identity;

    // 2D translation
    DOMMatrixReadOnly translated2D = identity.translate ( 10.0f, 20.0f, 0.0f );
    EXPECT_TRUE ( translated2D.is2D() );
    EXPECT_FLOAT_EQ ( translated2D.e(), 10.0f );
    EXPECT_FLOAT_EQ ( translated2D.f(), 20.0f );

    // 3D translation
    DOMMatrixReadOnly translated3D = identity.translate ( 10.0f, 20.0f, 30.0f );
    EXPECT_FALSE ( translated3D.is2D() );
    EXPECT_FLOAT_EQ ( translated3D.m41(), 10.0f );
    EXPECT_FLOAT_EQ ( translated3D.m42(), 20.0f );
    EXPECT_FLOAT_EQ ( translated3D.m43(), 30.0f );
}

// Test scaling
TEST_F ( DOMMatrixReadOnlyTest, Scale )
{
    DOMMatrixReadOnly identity;

    // Uniform scaling from origin
    DOMMatrixReadOnly scaled = identity.scale ( 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f );
    EXPECT_FLOAT_EQ ( scaled.m11(), 2.0f );
    EXPECT_FLOAT_EQ ( scaled.m22(), 3.0f );
    EXPECT_FLOAT_EQ ( scaled.m33(), 4.0f );

    // 3D uniform scaling
    DOMMatrixReadOnly scaled3D = identity.scale3d ( 2.0f, 0.0f, 0.0f, 0.0f );
    EXPECT_FLOAT_EQ ( scaled3D.m11(), 2.0f );
    EXPECT_FLOAT_EQ ( scaled3D.m22(), 2.0f );
    EXPECT_FLOAT_EQ ( scaled3D.m33(), 2.0f );
}

// Test rotation around Z axis (2D rotation)
TEST_F ( DOMMatrixReadOnlyTest, RotateZ )
{
    DOMMatrixReadOnly identity;

    // 90 degree rotation around Z axis
    DOMMatrixReadOnly rotated = identity.rotateAxisAngle ( 0.0f, 0.0f, 1.0f, 90.0f );

    // 2D rotation matrix: [cos(θ) -sin(θ)]  =  [0 -1]
    //                     [sin(θ)  cos(θ)]     [1  0]
    EXPECT_TRUE ( IsApproximatelyEqual ( rotated.m11(), 0.0f ) ); // cos(90°) = 0
    EXPECT_TRUE ( IsApproximatelyEqual ( rotated.m12(), -1.0f ) ); // -sin(90°) = -1
    EXPECT_TRUE ( IsApproximatelyEqual ( rotated.m21(), 1.0f ) ); // sin(90°) = 1
    EXPECT_TRUE ( IsApproximatelyEqual ( rotated.m22(), 0.0f ) ); // cos(90°) = 0
}

// Test flip operations
TEST_F ( DOMMatrixReadOnlyTest, FlipOperations )
{
    DOMMatrixReadOnly identity;

    // Flip X (negate first column)
    DOMMatrixReadOnly flippedX = identity.flipX();
    EXPECT_FLOAT_EQ ( flippedX.m11(), -1.0f );
    EXPECT_FLOAT_EQ ( flippedX.m21(), 0.0f );
    EXPECT_FLOAT_EQ ( flippedX.m31(), 0.0f );
    EXPECT_FLOAT_EQ ( flippedX.m41(), 0.0f );

    // Flip Y (negate second row elements)
    DOMMatrixReadOnly flippedY = identity.flipY();
    EXPECT_FLOAT_EQ ( flippedY.m12(), 0.0f ); // m12 unchanged (was 0)
    EXPECT_FLOAT_EQ ( flippedY.m22(), -1.0f ); // m22 negated (was 1)
    EXPECT_FLOAT_EQ ( flippedY.m32(), 0.0f ); // m32 unchanged (was 0)
    EXPECT_FLOAT_EQ ( flippedY.m42(), 0.0f ); // m42 unchanged (was 0)
}

// Test skew operations
TEST_F ( DOMMatrixReadOnlyTest, SkewOperations )
{
    DOMMatrixReadOnly identity;

    // Skew X by 45 degrees (tan(45°) = 1)
    DOMMatrixReadOnly skewedX = identity.skewX ( 45.0f );
    EXPECT_TRUE ( IsApproximatelyEqual ( skewedX.m21(), 1.0f ) );

    // Skew Y by 45 degrees
    DOMMatrixReadOnly skewedY = identity.skewY ( 45.0f );
    EXPECT_TRUE ( IsApproximatelyEqual ( skewedY.m12(), 1.0f ) );
}

// Test matrix inversion
TEST_F ( DOMMatrixReadOnlyTest, Inverse )
{
    // Simple 2D scaling matrix
    DOMMatrixReadOnly scaling ( {2.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f} );
    DOMMatrixReadOnly inverse = scaling.inverse();

    // Inverse of scaling should be 1/scale
    EXPECT_TRUE ( IsApproximatelyEqual ( inverse.a(), 0.5f ) ); // 1/2
    EXPECT_TRUE ( IsApproximatelyEqual ( inverse.d(), 1.0f / 3.0f ) ); // 1/3

    // Test that matrix * inverse = identity
    DOMMatrixReadOnly shouldBeIdentity = scaling.multiply ( inverse );
    EXPECT_TRUE ( IsApproximatelyIdentity ( shouldBeIdentity ) );
}

// Test singular matrix inversion (should throw)
TEST_F ( DOMMatrixReadOnlyTest, InverseSingularMatrix )
{
    // Zero determinant matrix
    DOMMatrixReadOnly singular ( {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f} );
    EXPECT_THROW ( singular.inverse(), DOMInvalidStateError );
}

// Test point transformation
TEST_F ( DOMMatrixReadOnlyTest, TransformPoint )
{
    // Translation matrix
    DOMMatrixReadOnly translation ( {1.0f, 0.0f, 0.0f, 1.0f, 10.0f, 20.0f} );
    DOMPoint point ( 5.0f, 15.0f, 0.0f, 1.0f );

    DOMPoint transformed = translation.transformPoint ( point );

    // Point should be translated by (10, 20)
    EXPECT_FLOAT_EQ ( transformed.x(), 15.0f ); // 5 + 10
    EXPECT_FLOAT_EQ ( transformed.y(), 35.0f ); // 15 + 20
    EXPECT_FLOAT_EQ ( transformed.z(), 0.0f );
    EXPECT_FLOAT_EQ ( transformed.w(), 1.0f );
}

// Test array conversion methods
TEST_F ( DOMMatrixReadOnlyTest, ArrayConversion )
{
    DOMMatrixReadOnly matrix ( {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f} );

    // Test Float32Array
    auto float32Array = matrix.toFloat32Array();
    EXPECT_EQ ( float32Array.size(), 16 );
    EXPECT_FLOAT_EQ ( float32Array[0], 1.0f ); // m11
    EXPECT_FLOAT_EQ ( float32Array[1], 2.0f ); // m12
    EXPECT_FLOAT_EQ ( float32Array[4], 3.0f ); // m21
    EXPECT_FLOAT_EQ ( float32Array[5], 4.0f ); // m22

    // Test Float64Array
    auto float64Array = matrix.toFloat64Array();
    EXPECT_EQ ( float64Array.size(), 16 );
    EXPECT_DOUBLE_EQ ( float64Array[0], 1.0 );
    EXPECT_DOUBLE_EQ ( float64Array[1], 2.0 );
}

// Test JSON serialization
TEST_F ( DOMMatrixReadOnlyTest, JSONSerialization )
{
    // 2D matrix JSON
    DOMMatrixReadOnly matrix2D ( {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f} );
    DOMString json2D = matrix2D.toJSON();

    EXPECT_THAT ( json2D, ::testing::HasSubstr ( "\"a\":1" ) );
    EXPECT_THAT ( json2D, ::testing::HasSubstr ( "\"b\":2" ) );
    EXPECT_THAT ( json2D, ::testing::HasSubstr ( "\"is2D\":true" ) );
    EXPECT_THAT ( json2D, ::testing::HasSubstr ( "\"isIdentity\":false" ) );

    // 3D matrix JSON
    DOMMatrixReadOnly matrix3D (
    {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    } );
    DOMString json3D = matrix3D.toJSON();

    EXPECT_THAT ( json3D, ::testing::HasSubstr ( "\"m11\":1" ) );
    EXPECT_THAT ( json3D, ::testing::HasSubstr ( "\"is2D\":false" ) );
    EXPECT_THAT ( json3D, ::testing::HasSubstr ( "\"isIdentity\":true" ) );
}

// Test CSS string serialization
TEST_F ( DOMMatrixReadOnlyTest, CSSStringSerialization )
{
    // 2D matrix string
    DOMMatrixReadOnly matrix2D ( {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f} );
    DOMString css2D = matrix2D.toString();

    EXPECT_THAT ( css2D, ::testing::HasSubstr ( "matrix(" ) );
    EXPECT_THAT ( css2D, ::testing::HasSubstr ( "1.000000" ) );
    EXPECT_THAT ( css2D, ::testing::HasSubstr ( "2.000000" ) );

    // 3D matrix string
    DOMMatrixReadOnly matrix3D (
    {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    } );
    DOMString css3D = matrix3D.toString();

    EXPECT_THAT ( css3D, ::testing::HasSubstr ( "matrix3d(" ) );
}

// Test copy constructor and assignment
TEST_F ( DOMMatrixReadOnlyTest, CopyOperations )
{
    DOMMatrixReadOnly original ( {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f} );

    // Copy constructor
    DOMMatrixReadOnly copied ( original );
    EXPECT_FLOAT_EQ ( copied.a(), 2.0f );
    EXPECT_FLOAT_EQ ( copied.b(), 3.0f );
    EXPECT_EQ ( copied.is2D(), original.is2D() );

    // Assignment operator
    DOMMatrixReadOnly assigned;
    assigned = original;
    EXPECT_FLOAT_EQ ( assigned.a(), 2.0f );
    EXPECT_FLOAT_EQ ( assigned.b(), 3.0f );
    EXPECT_EQ ( assigned.is2D(), original.is2D() );
}

// Test chained operations (3D)
TEST_F ( DOMMatrixReadOnlyTest, ChainedOperations )
{
    DOMMatrixReadOnly identity;

    // Chain multiple transformations including 3D rotation
    DOMMatrixReadOnly result = identity
                               .translate ( 10.0f, 20.0f, 0.0f )
                               .scale ( 2.0f, 2.0f, 1.0f, 0.0f, 0.0f, 0.0f )
                               .rotateAxisAngle ( 0.0f, 0.0f, 1.0f, 45.0f );

    // Verify the result is not identity
    EXPECT_FALSE ( result.isIdentity() );

    // rotateAxisAngle always creates 3D matrix, even for Z-axis rotations
    EXPECT_FALSE ( result.is2D() );
}

// Test chained 2D operations
TEST_F ( DOMMatrixReadOnlyTest, ChainedOperations2D )
{
    DOMMatrixReadOnly identity;

    // Chain only 2D transformations
    DOMMatrixReadOnly result = identity
                               .translate ( 10.0f, 20.0f, 0.0f ) // 2D translation (z=0)
                               .scale ( 2.0f, 2.0f, 1.0f, 0.0f, 0.0f, 0.0f ); // 2D scaling (z=1, origin at 0,0,0)

    // Verify the result is not identity but remains 2D
    EXPECT_FALSE ( result.isIdentity() );
    EXPECT_TRUE ( result.is2D() );

    // Check some values - scale applied first, then translation gets scaled
    EXPECT_FLOAT_EQ ( result.m11(), 2.0f ); // scale X
    EXPECT_FLOAT_EQ ( result.m22(), 2.0f ); // scale Y
    EXPECT_FLOAT_EQ ( result.m41(), 20.0f ); // translate X scaled: 10 * 2 = 20
    EXPECT_FLOAT_EQ ( result.m42(), 40.0f ); // translate Y scaled: 20 * 2 = 40
}

// Test Euler angle rotations
TEST_F ( DOMMatrixReadOnlyTest, EulerRotations )
{
    DOMMatrixReadOnly identity;

    // Single axis rotations
    DOMMatrixReadOnly rotX = identity.rotate ( 90.0f, 0.0f, 0.0f );
    DOMMatrixReadOnly rotY = identity.rotate ( 0.0f, 90.0f, 0.0f );
    DOMMatrixReadOnly rotZ = identity.rotate ( 0.0f, 0.0f, 90.0f );

    // All should be 3D matrices (except pure Z rotation which could be 2D)
    EXPECT_FALSE ( rotX.is2D() );
    EXPECT_FALSE ( rotY.is2D() );

    // Combined rotation
    DOMMatrixReadOnly rotXYZ = identity.rotate ( 30.0f, 45.0f, 60.0f );
    EXPECT_FALSE ( rotXYZ.is2D() );
    EXPECT_FALSE ( rotXYZ.isIdentity() );
}

// Test floating point precision edge cases
TEST_F ( DOMMatrixReadOnlyTest, FloatingPointPrecision )
{
    // Very small values
    DOMMatrixReadOnly tiny ( {0.001f, 0.002f, 0.003f, 0.004f, 0.005f, 0.006f} );
    EXPECT_TRUE ( IsApproximatelyEqual ( tiny.a(), 0.001f, 0.0001f ) );

    // Very large values
    DOMMatrixReadOnly large ( {1000000.0f, 2000000.0f, 0.0f, 1.0f, 0.0f, 0.0f} );
    EXPECT_FLOAT_EQ ( large.a(), 1000000.0f );
    EXPECT_FLOAT_EQ ( large.b(), 2000000.0f );
}

// Test 2D/3D state preservation
TEST_F ( DOMMatrixReadOnlyTest, DimensionalityPreservation )
{
    DOMMatrixReadOnly matrix2D ( {2.0f, 0.0f, 0.0f, 2.0f, 10.0f, 20.0f} );
    EXPECT_TRUE ( matrix2D.is2D() );

    // 2D operations should preserve 2D state
    DOMMatrixReadOnly still2D = matrix2D.translate ( 5.0f, 5.0f, 0.0f );
    EXPECT_TRUE ( still2D.is2D() );

    // 3D operations should make it 3D
    DOMMatrixReadOnly now3D = matrix2D.translate ( 0.0f, 0.0f, 5.0f );
    EXPECT_FALSE ( now3D.is2D() );
}