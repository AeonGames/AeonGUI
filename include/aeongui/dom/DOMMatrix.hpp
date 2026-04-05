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
#ifndef AEONGUI_DOMMATRIX_HPP
#define AEONGUI_DOMMATRIX_HPP

#include "aeongui/Platform.hpp"
#include "aeongui/dom/DOMMatrixReadOnly.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Mutable 4x4 transformation matrix.
         *
         *  Extends DOMMatrixReadOnly with in-place mutation methods
         *  (e.g. translateSelf, rotateSelf).
         *  @see https://drafts.fxtf.org/geometry/#dommatrix
         */
        class AEONGUI_DLL DOMMatrix : public DOMMatrixReadOnly
        {
        public:
            /** @brief Construct from an initializer list.
             *  @param values 6 values for 2D or 16 for 3D (defaults to 2D identity).
             */
            DOMMatrix ( std::initializer_list<float> values = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f} );
            /** @brief Destructor. */
            virtual ~DOMMatrix();

            // Bring base class getters into scope
            using DOMMatrixReadOnly::a;
            using DOMMatrixReadOnly::b;
            using DOMMatrixReadOnly::c;
            using DOMMatrixReadOnly::d;
            using DOMMatrixReadOnly::e;
            using DOMMatrixReadOnly::f;
            using DOMMatrixReadOnly::m11;
            using DOMMatrixReadOnly::m12;
            using DOMMatrixReadOnly::m13;
            using DOMMatrixReadOnly::m14;
            using DOMMatrixReadOnly::m21;
            using DOMMatrixReadOnly::m22;
            using DOMMatrixReadOnly::m23;
            using DOMMatrixReadOnly::m24;
            using DOMMatrixReadOnly::m31;
            using DOMMatrixReadOnly::m32;
            using DOMMatrixReadOnly::m33;
            using DOMMatrixReadOnly::m34;
            using DOMMatrixReadOnly::m41;
            using DOMMatrixReadOnly::m42;
            using DOMMatrixReadOnly::m43;
            using DOMMatrixReadOnly::m44;

            /** @name 2D Setters
             *  Set 2D matrix components (a-f aliases). @{ */
            /** @brief Set element a. @param newA New value. @return The new value. */
            float a ( float newA );
            /** @brief Set element b. @param newB New value. @return The new value. */
            float b ( float newB );
            /** @brief Set element c. @param newC New value. @return The new value. */
            float c ( float newC );
            /** @brief Set element d. @param newD New value. @return The new value. */
            float d ( float newD );
            /** @brief Set element e. @param newE New value. @return The new value. */
            float e ( float newE );
            /** @brief Set element f. @param newF New value. @return The new value. */
            float f ( float newF );
            /** @} */

            /** @name 4x4 Setters
             *  Set individual elements of the 4x4 matrix. @{ */
            /** @brief Set m11. @param newM11 New value. @return The new value. */
            float m11 ( float newM11 );
            /** @brief Set m12. @param newM12 New value. @return The new value. */
            float m12 ( float newM12 );
            /** @brief Set m13. @param newM13 New value. @return The new value. */
            float m13 ( float newM13 );
            /** @brief Set m14. @param newM14 New value. @return The new value. */
            float m14 ( float newM14 );
            /** @brief Set m21. @param newM21 New value. @return The new value. */
            float m21 ( float newM21 );
            /** @brief Set m22. @param newM22 New value. @return The new value. */
            float m22 ( float newM22 );
            /** @brief Set m23. @param newM23 New value. @return The new value. */
            float m23 ( float newM23 );
            /** @brief Set m24. @param newM24 New value. @return The new value. */
            float m24 ( float newM24 );
            /** @brief Set m31. @param newM31 New value. @return The new value. */
            float m31 ( float newM31 );
            /** @brief Set m32. @param newM32 New value. @return The new value. */
            float m32 ( float newM32 );
            /** @brief Set m33. @param newM33 New value. @return The new value. */
            float m33 ( float newM33 );
            /** @brief Set m34. @param newM34 New value. @return The new value. */
            float m34 ( float newM34 );
            /** @brief Set m41. @param newM41 New value. @return The new value. */
            float m41 ( float newM41 );
            /** @brief Set m42. @param newM42 New value. @return The new value. */
            float m42 ( float newM42 );
            /** @brief Set m43. @param newM43 New value. @return The new value. */
            float m43 ( float newM43 );
            /** @brief Set m44. @param newM44 New value. @return The new value. */
            float m44 ( float newM44 );
            /** @} */

            /** @brief Invert this matrix in place.
             *  @return Reference to this matrix. */
            DOMMatrix& invertSelf();
            /** @brief Post-multiply this matrix by another in place.
             *  @param other The other matrix.
             *  @return Reference to this matrix. */
            DOMMatrix& multiplySelf ( const DOMMatrixReadOnly& other );
            /** @brief Pre-multiply this matrix by another in place.
             *  @param other The other matrix.
             *  @return Reference to this matrix. */
            DOMMatrix& preMultiplySelf ( const DOMMatrixReadOnly& other );
            /** @brief Translate in place.
             *  @param tx X translation.
             *  @param ty Y translation.
             *  @param tz Z translation.
             *  @return Reference to this matrix. */
            DOMMatrix& translateSelf ( float tx, float ty = 0, float tz = 0 );
            /** @brief Scale in place.
             *  @param sx X scale factor.
             *  @param sy Y scale factor.
             *  @param sz Z scale factor.
             *  @return Reference to this matrix. */
            DOMMatrix& scaleSelf ( float sx, float sy = 1, float sz = 1 );
            /** @brief Uniform 3D scale in place.
             *  @param sx X scale factor.
             *  @param sy Y scale factor.
             *  @param sz Z scale factor.
             *  @return Reference to this matrix. */
            DOMMatrix& scale3dSelf ( float sx, float sy = 1, float sz = 1 );
            /** @brief Rotate in place.
             *  @param rx Rotation around X in degrees.
             *  @param ry Rotation around Y in degrees.
             *  @param rz Rotation around Z in degrees.
             *  @return Reference to this matrix. */
            DOMMatrix& rotateSelf ( float rx, float ry = 0, float rz = 0 );
            /** @brief Rotate around an axis in place.
             *  @param x     X component of axis.
             *  @param y     Y component of axis.
             *  @param z     Z component of axis.
             *  @param angle Rotation angle in degrees.
             *  @return Reference to this matrix. */
            DOMMatrix& rotateAxisAngleSelf ( float x, float y, float z, float angle );
            /** @brief Rotate from a direction vector in place.
             *  @param rotX X component.
             *  @param rotY Y component.
             *  @return Reference to this matrix. */
            DOMMatrix& rotateFromVectorSelf ( float rotX, float rotY );
            /** @brief Skew along X in place.
             *  @param angle Skew angle in degrees.
             *  @return Reference to this matrix. */
            DOMMatrix& skewXSelf ( float angle );
            /** @brief Skew along Y in place.
             *  @param angle Skew angle in degrees.
             *  @return Reference to this matrix. */
            DOMMatrix& skewYSelf ( float angle );
        private:
        };
    }
}

#endif
