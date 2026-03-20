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
            /** @brief Check whether this is a 2D matrix. */
            bool is2D() const;
            /** @brief Check whether this is the identity matrix. */
            bool isIdentity() const;
            /** @brief Get element a (alias m11). */
            float a() const;
            /** @brief Get element b (alias m12). */
            float b() const;
            /** @brief Get element c (alias m21). */
            float c() const;
            /** @brief Get element d (alias m22). */
            float d() const;
            /** @brief Get element e (alias m41 / translateX). */
            float e() const;
            /** @brief Get element f (alias m42 / translateY). */
            float f() const;

            float m11() const; ///< Row 1, column 1.
            float m12() const; ///< Row 1, column 2.
            float m13() const; ///< Row 1, column 3.
            float m14() const; ///< Row 1, column 4.
            float m21() const; ///< Row 2, column 1.
            float m22() const; ///< Row 2, column 2.
            float m23() const; ///< Row 2, column 3.
            float m24() const; ///< Row 2, column 4.
            float m31() const; ///< Row 3, column 1.
            float m32() const; ///< Row 3, column 2.
            float m33() const; ///< Row 3, column 3.
            float m34() const; ///< Row 3, column 4.
            float m41() const; ///< Row 4, column 1.
            float m42() const; ///< Row 4, column 2.
            float m43() const; ///< Row 4, column 3.
            float m44() const; ///< Row 4, column 4.

            /** @brief Flip around the X axis. */
            DOMMatrixReadOnly flipX() const;
            /** @brief Flip around the Y axis. */
            DOMMatrixReadOnly flipY() const;
            /** @brief Compute the inverse. */
            DOMMatrixReadOnly inverse() const;
            /** @brief Post-multiply with another matrix. */
            DOMMatrixReadOnly multiply ( const DOMMatrixReadOnly& other ) const;
            /** @brief Rotate around an arbitrary axis.
             *  @param x     X component of the axis.
             *  @param y     Y component of the axis.
             *  @param z     Z component of the axis.
             *  @param angle Rotation angle in degrees.
             */
            DOMMatrixReadOnly rotateAxisAngle ( float x, float y, float z, float angle ) const;
            /** @brief Rotate around each axis.
             *  @param rotX Rotation around X axis in degrees.
             *  @param rotY Rotation around Y axis in degrees.
             *  @param rotZ Rotation around Z axis in degrees.
             */
            DOMMatrixReadOnly rotate ( float rotX, float rotY, float rotZ ) const;
            /** @brief Rotate from a direction vector. */
            DOMMatrixReadOnly rotateFromVector ( float x, float y ) const;
            /** @brief Scale with separate factors and origin. */
            DOMMatrixReadOnly scale ( float scaleX, float scaleY, float scaleZ, float originX, float originY, float originZ ) const;
            /** @brief Uniform 3D scale with origin. */
            DOMMatrixReadOnly scale3d ( float scale, float originX, float originY, float originZ ) const;
            /** @brief Skew along the X axis.
             *  @param sx Skew angle in degrees.
             */
            DOMMatrixReadOnly skewX ( float sx ) const;
            /** @brief Skew along the Y axis.
             *  @param sy Skew angle in degrees.
             */
            DOMMatrixReadOnly skewY ( float sy ) const;

            /** @brief Return the matrix as a 16-element float array. */
            std::array<float, 16> toFloat32Array() const;
            /** @brief Return the matrix as a 16-element double array. */
            std::array<double, 16> toFloat64Array() const;
            /** @brief Serialize to JSON. */
            DOMString toJSON() const;
            /** @brief Serialize to a CSS matrix/matrix3d string. */
            DOMString toString() const;
            /** @brief Transform a point by this matrix. */
            DOMPoint transformPoint ( const DOMPoint& point ) const;
            /** @brief Translate by (x, y, z). */
            DOMMatrixReadOnly translate ( float x, float y, float z ) const;

        protected:
            std::array<float, 16> mValues{};
            bool mIs2D{};
        };
    }
}

#endif
