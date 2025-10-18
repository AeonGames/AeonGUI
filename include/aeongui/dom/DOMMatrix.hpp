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
#ifndef AEONGUI_DOMMATRIX_HPP
#define AEONGUI_DOMMATRIX_HPP

#include "aeongui/Platform.hpp"
#include "aeongui/dom/DOMMatrixReadOnly.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class DLL DOMMatrix : public DOMMatrixReadOnly
        {
        public:
            DOMMatrix ( std::initializer_list<float> values = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f} );
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

            // Setters for matrix values
            float a ( float newA );
            float b ( float newB );
            float c ( float newC );
            float d ( float newD );
            float e ( float newE );
            float f ( float newF );

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

            DOMMatrix& invertSelf();

            DOMMatrix& multiplySelf ( const DOMMatrixReadOnly& other );

            DOMMatrix& preMultiplySelf ( const DOMMatrixReadOnly& other );

            DOMMatrix& translateSelf ( float tx, float ty = 0, float tz = 0 );

            DOMMatrix& scaleSelf ( float sx, float sy = 1, float sz = 1 );

            DOMMatrix& scale3dSelf ( float sx, float sy = 1, float sz = 1 );

            DOMMatrix& rotateSelf ( float rx, float ry = 0, float rz = 0 );

            DOMMatrix& rotateAxisAngleSelf ( float x, float y, float z, float angle );

            DOMMatrix& rotateFromVectorSelf ( float rotX, float rotY );

            DOMMatrix& skewXSelf ( float angle );

            DOMMatrix& skewYSelf ( float angle );
        private:
        };
    }
}

#endif
