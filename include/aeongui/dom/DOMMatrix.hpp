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
        class DLL DOMMatrix : public DOMMatrixReadOnly
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
            float a ( float newA );
            float b ( float newB );
            float c ( float newC );
            float d ( float newD );
            float e ( float newE );
            float f ( float newF );
            /** @} */

            /** @name 4x4 Setters
             *  Set individual elements of the 4x4 matrix. @{ */
            float m11 ( float newM11 );
            float m12 ( float newM12 );
            float m13 ( float newM13 );
            float m14 ( float newM14 );
            float m21 ( float newM21 );
            float m22 ( float newM22 );
            float m23 ( float newM23 );
            float m24 ( float newM24 );
            float m31 ( float newM31 );
            float m32 ( float newM32 );
            float m33 ( float newM33 );
            float m34 ( float newM34 );
            float m41 ( float newM41 );
            float m42 ( float newM42 );
            float m43 ( float newM43 );
            float m44 ( float newM44 );
            /** @} */

            /** @brief Invert this matrix in place. */
            DOMMatrix& invertSelf();
            /** @brief Post-multiply this matrix by another in place. */
            DOMMatrix& multiplySelf ( const DOMMatrixReadOnly& other );
            /** @brief Pre-multiply this matrix by another in place. */
            DOMMatrix& preMultiplySelf ( const DOMMatrixReadOnly& other );
            /** @brief Translate in place. */
            DOMMatrix& translateSelf ( float tx, float ty = 0, float tz = 0 );
            /** @brief Scale in place. */
            DOMMatrix& scaleSelf ( float sx, float sy = 1, float sz = 1 );
            /** @brief Uniform 3D scale in place. */
            DOMMatrix& scale3dSelf ( float sx, float sy = 1, float sz = 1 );
            /** @brief Rotate in place. */
            DOMMatrix& rotateSelf ( float rx, float ry = 0, float rz = 0 );
            /** @brief Rotate around an axis in place. */
            DOMMatrix& rotateAxisAngleSelf ( float x, float y, float z, float angle );
            /** @brief Rotate from a direction vector in place. */
            DOMMatrix& rotateFromVectorSelf ( float rotX, float rotY );
            /** @brief Skew along X in place. */
            DOMMatrix& skewXSelf ( float angle );
            /** @brief Skew along Y in place. */
            DOMMatrix& skewYSelf ( float angle );
        private:
        };
    }
}

#endif
