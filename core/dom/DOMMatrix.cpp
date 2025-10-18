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
#include "aeongui/dom/DOMMatrix.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        DOMMatrix::DOMMatrix ( std::initializer_list<float> values ) : DOMMatrixReadOnly ( values )
        {
        }

        DOMMatrix::~DOMMatrix() = default;

        // 2D Matrix setters (legacy names)
        float DOMMatrix::a ( float newA )
        {
            mValues[0] = newA;  // m11
            return mValues[0];
        }

        float DOMMatrix::b ( float newB )
        {
            mValues[1] = newB;  // m12
            return mValues[1];
        }

        float DOMMatrix::c ( float newC )
        {
            mValues[4] = newC;  // m21
            return mValues[4];
        }

        float DOMMatrix::d ( float newD )
        {
            mValues[5] = newD;  // m22
            return mValues[5];
        }

        float DOMMatrix::e ( float newE )
        {
            mValues[12] = newE;  // m41
            return mValues[12];
        }

        float DOMMatrix::f ( float newF )
        {
            mValues[13] = newF;  // m42
            return mValues[13];
        }

        // 4x4 Matrix setters
        float DOMMatrix::m11 ( float newM11 )
        {
            mValues[0] = newM11;
            return mValues[0];
        }

        float DOMMatrix::m12 ( float newM12 )
        {
            mValues[1] = newM12;
            return mValues[1];
        }

        float DOMMatrix::m13 ( float newM13 )
        {
            mValues[2] = newM13;
            if ( newM13 != 0.0f )
            {
                mIs2D = false;
            }
            return mValues[2];
        }

        float DOMMatrix::m14 ( float newM14 )
        {
            mValues[3] = newM14;
            if ( newM14 != 0.0f )
            {
                mIs2D = false;
            }
            return mValues[3];
        }

        float DOMMatrix::m21 ( float newM21 )
        {
            mValues[4] = newM21;
            return mValues[4];
        }

        float DOMMatrix::m22 ( float newM22 )
        {
            mValues[5] = newM22;
            return mValues[5];
        }

        float DOMMatrix::m23 ( float newM23 )
        {
            mValues[6] = newM23;
            if ( newM23 != 0.0f )
            {
                mIs2D = false;
            }
            return mValues[6];
        }

        float DOMMatrix::m24 ( float newM24 )
        {
            mValues[7] = newM24;
            if ( newM24 != 0.0f )
            {
                mIs2D = false;
            }
            return mValues[7];
        }

        float DOMMatrix::m31 ( float newM31 )
        {
            mValues[8] = newM31;
            if ( newM31 != 0.0f )
            {
                mIs2D = false;
            }
            return mValues[8];
        }

        float DOMMatrix::m32 ( float newM32 )
        {
            mValues[9] = newM32;
            if ( newM32 != 0.0f )
            {
                mIs2D = false;
            }
            return mValues[9];
        }

        float DOMMatrix::m33 ( float newM33 )
        {
            mValues[10] = newM33;
            if ( newM33 != 1.0f )
            {
                mIs2D = false;
            }
            return mValues[10];
        }

        float DOMMatrix::m34 ( float newM34 )
        {
            mValues[11] = newM34;
            if ( newM34 != 0.0f )
            {
                mIs2D = false;
            }
            return mValues[11];
        }

        float DOMMatrix::m41 ( float newM41 )
        {
            mValues[12] = newM41;
            return mValues[12];
        }

        float DOMMatrix::m42 ( float newM42 )
        {
            mValues[13] = newM42;
            return mValues[13];
        }

        float DOMMatrix::m43 ( float newM43 )
        {
            mValues[14] = newM43;
            if ( newM43 != 0.0f )
            {
                mIs2D = false;
            }
            return mValues[14];
        }

        float DOMMatrix::m44 ( float newM44 )
        {
            mValues[15] = newM44;
            if ( newM44 != 1.0f )
            {
                mIs2D = false;
            }
            return mValues[15];
        }

        // Self-modifying transformation methods
        DOMMatrix& DOMMatrix::invertSelf()
        {
            DOMMatrixReadOnly result = inverse();
            mValues = result.toFloat32Array();
            mIs2D = result.is2D();
            return *this;
        }

        DOMMatrix& DOMMatrix::multiplySelf ( const DOMMatrixReadOnly& other )
        {
            DOMMatrixReadOnly result = multiply ( other );
            mValues = result.toFloat32Array();
            mIs2D = result.is2D();
            return *this;
        }

        DOMMatrix& DOMMatrix::preMultiplySelf ( const DOMMatrixReadOnly& other )
        {
            DOMMatrixReadOnly result = other.multiply ( *this );
            mValues = result.toFloat32Array();
            mIs2D = result.is2D();
            return *this;
        }

        DOMMatrix& DOMMatrix::translateSelf ( float tx, float ty, float tz )
        {
            DOMMatrixReadOnly result = translate ( tx, ty, tz );
            mValues = result.toFloat32Array();
            mIs2D = result.is2D();
            return *this;
        }

        DOMMatrix& DOMMatrix::scaleSelf ( float sx, float sy, float sz )
        {
            // Use base class scale method with origin at (0,0,0)
            DOMMatrixReadOnly result = scale ( sx, sy, sz, 0.0f, 0.0f, 0.0f );
            mValues = result.toFloat32Array();
            mIs2D = result.is2D();
            return *this;
        }

        DOMMatrix& DOMMatrix::scale3dSelf ( float sx, float sy, float sz )
        {
            // scale3d is equivalent to scale with same parameters
            return scaleSelf ( sx, sy, sz );
        }

        DOMMatrix& DOMMatrix::rotateSelf ( float rx, float ry, float rz )
        {
            DOMMatrixReadOnly result = rotate ( rx, ry, rz );
            mValues = result.toFloat32Array();
            mIs2D = result.is2D();
            return *this;
        }

        DOMMatrix& DOMMatrix::rotateAxisAngleSelf ( float x, float y, float z, float angle )
        {
            DOMMatrixReadOnly result = rotateAxisAngle ( x, y, z, angle );
            mValues = result.toFloat32Array();
            mIs2D = result.is2D();
            return *this;
        }

        DOMMatrix& DOMMatrix::rotateFromVectorSelf ( float rotX, float rotY )
        {
            DOMMatrixReadOnly result = rotateFromVector ( rotX, rotY );
            mValues = result.toFloat32Array();
            mIs2D = result.is2D();
            return *this;
        }

        DOMMatrix& DOMMatrix::skewXSelf ( float angle )
        {
            DOMMatrixReadOnly result = skewX ( angle );
            mValues = result.toFloat32Array();
            mIs2D = result.is2D();
            return *this;
        }

        DOMMatrix& DOMMatrix::skewYSelf ( float angle )
        {
            DOMMatrixReadOnly result = skewY ( angle );
            mValues = result.toFloat32Array();
            mIs2D = result.is2D();
            return *this;
        }
    }
}
