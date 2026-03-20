/*
Copyright (C) 2019,2024-2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_MATRIX2X3_H
#define AEONGUI_MATRIX2X3_H
#include <array>
#include <cstdint>
#include <cstddef>
#include <string>
#include "aeongui/Platform.hpp"
namespace AeonGUI
{
    class Vector2;
    /** @brief 2x3 affine transformation matrix.
     *
     *  Represents a 2D affine transform as a 2x3 matrix of the form:
     *  @code
     *  | xx  yx |
     *  | xy  yy |
     *  | x0  y0 |
     *  @endcode
     *  Supports construction from scale, rotation, and translation components.
     */
    class Matrix2x3
    {
    public:
        /** @brief Default constructor. Initializes to the identity matrix. */
        DLL Matrix2x3();
        /** @brief Construct from individual matrix elements.
         *  @param xx Row 0, column 0.
         *  @param yx Row 0, column 1.
         *  @param xy Row 1, column 0.
         *  @param yy Row 1, column 1.
         *  @param x0 Translation X.
         *  @param y0 Translation Y.
         */
        DLL Matrix2x3 (
            double xx, double yx,
            double xy, double yy,
            double x0, double y0
        );
        /** @brief Construct from an array of 6 doubles.
         *  @param aMatrixArray Array in row-major order {xx, yx, xy, yy, x0, y0}.
         */
        DLL Matrix2x3 ( const std::array<const double, 6> aMatrixArray );
        /** @brief Construct a rotation matrix.
         *  @param aRotation Rotation angle in radians.
         */
        DLL Matrix2x3 ( double aRotation );
        /** @brief Construct a scale matrix.
         *  @param aScale Scale factors for each axis.
         */
        DLL Matrix2x3 ( const Vector2& aScale );
        /** @brief Construct a scale-rotation matrix.
         *  @param aScale    Scale factors for each axis.
         *  @param aRotation Rotation angle in radians.
         */
        DLL Matrix2x3 ( const Vector2& aScale, double aRotation );
        /** @brief Construct a full scale-rotation-translation matrix.
         *  @param aScale       Scale factors for each axis.
         *  @param aRotation    Rotation angle in radians.
         *  @param aTranslation Translation vector.
         */
        DLL Matrix2x3 ( const Vector2& aScale, double aRotation, const Vector2& aTranslation );
        /** @brief Multiply this matrix by another (post-multiply).
         *  @param aRight The right-hand-side matrix.
         *  @return Reference to this matrix after multiplication.
         */
        DLL Matrix2x3& operator*= ( const Matrix2x3& aRight );
        /** @brief Access a matrix element by index.
         *  @param aIndex Element index [0..5].
         *  @return Const reference to the element.
         */
        DLL const double& operator[] ( size_t aIndex ) const;
    private:
        double mMatrix2x3[6];
    };
    /** @brief Compute the absolute value of each matrix element.
     *  @param aMatrix2x3 The input matrix.
     *  @return A new matrix with absolute values.
     */
    const Matrix2x3 Abs ( const Matrix2x3& aMatrix2x3 );
    /** @brief Parse an SVG transform attribute string.
     *  @param value The SVG transform string (e.g. "rotate(45)").
     *  @return The resulting transformation matrix.
     */
    DLL Matrix2x3 ParseSVGTransform ( const std::string& value );
}
#endif
