/*
Copyright (C) 2025,2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_DOMMATRIXREADONLY_HPP
#define AEONGUI_DOMMATRIXREADONLY_HPP

#include <initializer_list>
#include <array>
#include "aeongui/Platform.hpp"
#include "DOMString.hpp"
#include "DOMPoint.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Immutable 4x4 transformation matrix.
         *
         *  Implements the DOM DOMMatrixReadOnly interface. Supports both
         *  2D and 3D transforms. Provides methods for multiplication,
         *  rotation, scaling, skewing, translation, and inversion.
         *  @see https://drafts.fxtf.org/geometry/#dommatrixreadonly
         *  @todo Implement CSS transform and transform-functions related functions.
         */
        class DLL DOMMatrixReadOnly
        {
        public:
            /** @brief Construct from an initializer list.
             *  @param values 6 values for 2D or 16 for 3D (defaults to 2D identity).
             */
            DOMMatrixReadOnly ( std::initializer_list<float> values = {1, 0, 0, 1, 0, 0} );
            /** @brief Destructor. */
            virtual ~DOMMatrixReadOnly();
            /** @brief Check whether this is a 2D matrix.
             *  @return true if the matrix is 2D. */
            bool is2D() const;
            /** @brief Check whether this is the identity matrix.
             *  @return true if this is the identity matrix. */
            bool isIdentity() const;
            /** @brief Get element a (alias m11).
             *  @return The a component. */
            float a() const;
            /** @brief Get element b (alias m12).
             *  @return The b component. */
            float b() const;
            /** @brief Get element c (alias m21).
             *  @return The c component. */
            float c() const;
            /** @brief Get element d (alias m22).
             *  @return The d component. */
            float d() const;
            /** @brief Get element e (alias m41 / translateX).
             *  @return The e component. */
            float e() const;
            /** @brief Get element f (alias m42 / translateY).
             *  @return The f component. */
            float f() const;

            /** @brief Get row 1, column 1. @return The m11 element. */
            float m11() const;
            /** @brief Get row 1, column 2. @return The m12 element. */
            float m12() const;
            /** @brief Get row 1, column 3. @return The m13 element. */
            float m13() const;
            /** @brief Get row 1, column 4. @return The m14 element. */
            float m14() const;
            /** @brief Get row 2, column 1. @return The m21 element. */
            float m21() const;
            /** @brief Get row 2, column 2. @return The m22 element. */
            float m22() const;
            /** @brief Get row 2, column 3. @return The m23 element. */
            float m23() const;
            /** @brief Get row 2, column 4. @return The m24 element. */
            float m24() const;
            /** @brief Get row 3, column 1. @return The m31 element. */
            float m31() const;
            /** @brief Get row 3, column 2. @return The m32 element. */
            float m32() const;
            /** @brief Get row 3, column 3. @return The m33 element. */
            float m33() const;
            /** @brief Get row 3, column 4. @return The m34 element. */
            float m34() const;
            /** @brief Get row 4, column 1. @return The m41 element. */
            float m41() const;
            /** @brief Get row 4, column 2. @return The m42 element. */
            float m42() const;
            /** @brief Get row 4, column 3. @return The m43 element. */
            float m43() const;
            /** @brief Get row 4, column 4. @return The m44 element. */
            float m44() const;

            /** @brief Flip around the X axis.
             *  @return The flipped matrix. */
            DOMMatrixReadOnly flipX() const;
            /** @brief Flip around the Y axis.
             *  @return The flipped matrix. */
            DOMMatrixReadOnly flipY() const;
            /** @brief Compute the inverse.
             *  @return The inverted matrix. */
            DOMMatrixReadOnly inverse() const;
            /** @brief Post-multiply with another matrix.
             *  @param other The matrix to multiply by.
             *  @return The product matrix. */
            DOMMatrixReadOnly multiply ( const DOMMatrixReadOnly& other ) const;
            /** @brief Rotate around an arbitrary axis.
             *  @param x     X component of the axis.
             *  @param y     Y component of the axis.
             *  @param z     Z component of the axis.
             *  @param angle Rotation angle in degrees.
             *  @return The rotated matrix.
             */
            DOMMatrixReadOnly rotateAxisAngle ( float x, float y, float z, float angle ) const;
            /** @brief Rotate around each axis.
             *  @param rotX Rotation around X axis in degrees.
             *  @param rotY Rotation around Y axis in degrees.
             *  @param rotZ Rotation around Z axis in degrees.
             *  @return The rotated matrix.
             */
            DOMMatrixReadOnly rotate ( float rotX, float rotY, float rotZ ) const;
            /** @brief Rotate from a direction vector.
             *  @param x X component.
             *  @param y Y component.
             *  @return The rotated matrix. */
            DOMMatrixReadOnly rotateFromVector ( float x, float y ) const;
            /** @brief Scale with separate factors and origin.
             *  @param scaleX  Scale factor along X.
             *  @param scaleY  Scale factor along Y.
             *  @param scaleZ  Scale factor along Z.
             *  @param originX Origin X coordinate.
             *  @param originY Origin Y coordinate.
             *  @param originZ Origin Z coordinate.
             *  @return The scaled matrix. */
            DOMMatrixReadOnly scale ( float scaleX, float scaleY, float scaleZ, float originX, float originY, float originZ ) const;
            /** @brief Uniform 3D scale with origin.
             *  @param scale   Uniform scale factor.
             *  @param originX Origin X coordinate.
             *  @param originY Origin Y coordinate.
             *  @param originZ Origin Z coordinate.
             *  @return The scaled matrix. */
            DOMMatrixReadOnly scale3d ( float scale, float originX, float originY, float originZ ) const;
            /** @brief Skew along the X axis.
             *  @param sx Skew angle in degrees.
             *  @return The skewed matrix.
             */
            DOMMatrixReadOnly skewX ( float sx ) const;
            /** @brief Skew along the Y axis.
             *  @param sy Skew angle in degrees.
             *  @return The skewed matrix.
             */
            DOMMatrixReadOnly skewY ( float sy ) const;

            /** @brief Return the matrix as a 16-element float array.
             *  @return Array of 16 floats. */
            std::array<float, 16> toFloat32Array() const;
            /** @brief Return the matrix as a 16-element double array.
             *  @return Array of 16 doubles. */
            std::array<double, 16> toFloat64Array() const;
            /** @brief Serialize to JSON.
             *  @return JSON string representation. */
            DOMString toJSON() const;
            /** @brief Serialize to a CSS matrix/matrix3d string.
             *  @return CSS string representation. */
            DOMString toString() const;
            /** @brief Transform a point by this matrix.
             *  @param point The point to transform.
             *  @return The transformed point. */
            DOMPoint transformPoint ( const DOMPoint& point ) const;
            /** @brief Translate by (x, y, z).
             *  @param x Translation along X.
             *  @param y Translation along Y.
             *  @param z Translation along Z.
             *  @return The translated matrix. */
            DOMMatrixReadOnly translate ( float x, float y, float z ) const;

        protected:
            std::array<float, 16> mValues{}; ///< The 4x4 matrix values in column-major order.
            bool mIs2D{}; ///< Whether the matrix represents a 2D transform.
        };
    }
}

#endif
