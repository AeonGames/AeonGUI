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
#include "aeongui/dom/DOMMatrixReadOnly.hpp"
#include "aeongui/dom/DOMException.hpp"
#include <cmath>
#include <algorithm>
namespace AeonGUI
{
    namespace DOM
    {
        DOMMatrixReadOnly::DOMMatrixReadOnly ( std::initializer_list<float> values ) : mIs2D {values.size() == 6}
        {
            if ( values.size() != 6 && values.size() != 16 )
            {
                throw DOMTypeMismatchError ( "DOMMatrixReadOnly constructor requires 6 or 16 values." );
            }
            else if ( values.size() == 16 )
            {
                std::copy ( values.begin(), values.end(), mValues.begin() );
            }
            else
            {
                mValues[0] = values.begin() [0];
                mValues[1] = values.begin() [1];
                mValues[2] = 0.0f;
                mValues[3] = 0.0f;
                mValues[4] = values.begin() [2];
                mValues[5] = values.begin() [3];
                mValues[6] = 0.0f;
                mValues[7] = 0.0f;
                mValues[8] = 0.0f;
                mValues[9] = 0.0f;
                mValues[10] = 1.0f;
                mValues[11] = 0.0f;
                mValues[12] = values.begin() [4];
                mValues[13] = values.begin() [5];
                mValues[14] = 0.0f;
                mValues[15] = 1.0f;
            }
        }

        DOMMatrixReadOnly::~DOMMatrixReadOnly() = default;

        bool DOMMatrixReadOnly::is2D() const
        {
            return mIs2D;
        }

        bool DOMMatrixReadOnly::isIdentity() const
        {
            return mValues[0] == 1.0f && mValues[1] == 0.0f && mValues[2] == 0.0f && mValues[3] == 0.0f &&
                   mValues[4] == 0.0f && mValues[5] == 1.0f && mValues[6] == 0.0f && mValues[7] == 0.0f &&
                   mValues[8] == 0.0f && mValues[9] == 0.0f && mValues[10] == 1.0f && mValues[11] == 0.0f &&
                   mValues[12] == 0.0f && mValues[13] == 0.0f && mValues[14] == 0.0f && mValues[15] == 1.0f;
        }

        // 2D Matrix accessors (legacy names)
        float DOMMatrixReadOnly::a() const
        {
            return mValues[0];    // m11
        }
        float DOMMatrixReadOnly::b() const
        {
            return mValues[1];    // m12
        }
        float DOMMatrixReadOnly::c() const
        {
            return mValues[4];    // m21
        }
        float DOMMatrixReadOnly::d() const
        {
            return mValues[5];    // m22
        }
        float DOMMatrixReadOnly::e() const
        {
            return mValues[12];    // m41
        }
        float DOMMatrixReadOnly::f() const
        {
            return mValues[13];    // m42
        }

        // 4x4 Matrix accessors
        float DOMMatrixReadOnly::m11() const
        {
            return mValues[0];
        }
        float DOMMatrixReadOnly::m12() const
        {
            return mValues[1];
        }
        float DOMMatrixReadOnly::m13() const
        {
            return mValues[2];
        }
        float DOMMatrixReadOnly::m14() const
        {
            return mValues[3];
        }
        float DOMMatrixReadOnly::m21() const
        {
            return mValues[4];
        }
        float DOMMatrixReadOnly::m22() const
        {
            return mValues[5];
        }
        float DOMMatrixReadOnly::m23() const
        {
            return mValues[6];
        }
        float DOMMatrixReadOnly::m24() const
        {
            return mValues[7];
        }
        float DOMMatrixReadOnly::m31() const
        {
            return mValues[8];
        }
        float DOMMatrixReadOnly::m32() const
        {
            return mValues[9];
        }
        float DOMMatrixReadOnly::m33() const
        {
            return mValues[10];
        }
        float DOMMatrixReadOnly::m34() const
        {
            return mValues[11];
        }
        float DOMMatrixReadOnly::m41() const
        {
            return mValues[12];
        }
        float DOMMatrixReadOnly::m42() const
        {
            return mValues[13];
        }
        float DOMMatrixReadOnly::m43() const
        {
            return mValues[14];
        }
        float DOMMatrixReadOnly::m44() const
        {
            return mValues[15];
        }

        DOMMatrixReadOnly DOMMatrixReadOnly::flipX() const
        {
            DOMMatrixReadOnly result ( *this );
            result.mValues[0] = -result.mValues[0];   // m11
            result.mValues[4] = -result.mValues[4];   // m21
            result.mValues[8] = -result.mValues[8];   // m31
            result.mValues[12] = -result.mValues[12]; // m41
            return result;
        }

        DOMMatrixReadOnly DOMMatrixReadOnly::flipY() const
        {
            DOMMatrixReadOnly result ( *this );
            result.mValues[1] = -result.mValues[1];   // m12
            result.mValues[5] = -result.mValues[5];   // m22
            result.mValues[9] = -result.mValues[9];   // m32
            result.mValues[13] = -result.mValues[13]; // m42
            return result;
        }

        DOMMatrixReadOnly DOMMatrixReadOnly::inverse() const
        {
            DOMMatrixReadOnly result;

            float det = mValues[0] * ( mValues[5] * ( mValues[10] * mValues[15] - mValues[11] * mValues[14] ) - mValues[6] * ( mValues[9] * mValues[15] - mValues[11] * mValues[13] ) + mValues[7] * ( mValues[9] * mValues[14] - mValues[10] * mValues[13] ) ) -
                        mValues[1] * ( mValues[4] * ( mValues[10] * mValues[15] - mValues[11] * mValues[14] ) - mValues[6] * ( mValues[8] * mValues[15] - mValues[11] * mValues[12] ) + mValues[7] * ( mValues[8] * mValues[14] - mValues[10] * mValues[12] ) ) +
                        mValues[2] * ( mValues[4] * ( mValues[9] * mValues[15] - mValues[11] * mValues[13] ) - mValues[5] * ( mValues[8] * mValues[15] - mValues[11] * mValues[12] ) + mValues[7] * ( mValues[8] * mValues[13] - mValues[9] * mValues[12] ) ) -
                        mValues[3] * ( mValues[4] * ( mValues[9] * mValues[14] - mValues[10] * mValues[13] ) - mValues[5] * ( mValues[8] * mValues[14] - mValues[10] * mValues[12] ) + mValues[6] * ( mValues[8] * mValues[13] - mValues[9] * mValues[12] ) );

            if ( std::abs ( det ) < 1e-10f )
            {
                throw DOMInvalidStateError ( "Matrix is not invertible" );
            }

            float invDet = 1.0f / det;

            result.mValues[0] = invDet * ( mValues[5] * ( mValues[10] * mValues[15] - mValues[11] * mValues[14] ) - mValues[6] * ( mValues[9] * mValues[15] - mValues[11] * mValues[13] ) + mValues[7] * ( mValues[9] * mValues[14] - mValues[10] * mValues[13] ) );
            result.mValues[1] = invDet * ( - ( mValues[1] * ( mValues[10] * mValues[15] - mValues[11] * mValues[14] ) - mValues[2] * ( mValues[9] * mValues[15] - mValues[11] * mValues[13] ) + mValues[3] * ( mValues[9] * mValues[14] - mValues[10] * mValues[13] ) ) );
            result.mValues[2] = invDet * ( mValues[1] * ( mValues[6] * mValues[15] - mValues[7] * mValues[14] ) - mValues[2] * ( mValues[5] * mValues[15] - mValues[7] * mValues[13] ) + mValues[3] * ( mValues[5] * mValues[14] - mValues[6] * mValues[13] ) );
            result.mValues[3] = invDet * ( - ( mValues[1] * ( mValues[6] * mValues[11] - mValues[7] * mValues[10] ) - mValues[2] * ( mValues[5] * mValues[11] - mValues[7] * mValues[9] ) + mValues[3] * ( mValues[5] * mValues[10] - mValues[6] * mValues[9] ) ) );

            result.mValues[4] = invDet * ( - ( mValues[4] * ( mValues[10] * mValues[15] - mValues[11] * mValues[14] ) - mValues[6] * ( mValues[8] * mValues[15] - mValues[11] * mValues[12] ) + mValues[7] * ( mValues[8] * mValues[14] - mValues[10] * mValues[12] ) ) );
            result.mValues[5] = invDet * ( mValues[0] * ( mValues[10] * mValues[15] - mValues[11] * mValues[14] ) - mValues[2] * ( mValues[8] * mValues[15] - mValues[11] * mValues[12] ) + mValues[3] * ( mValues[8] * mValues[14] - mValues[10] * mValues[12] ) );
            result.mValues[6] = invDet * ( - ( mValues[0] * ( mValues[6] * mValues[15] - mValues[7] * mValues[14] ) - mValues[2] * ( mValues[4] * mValues[15] - mValues[7] * mValues[12] ) + mValues[3] * ( mValues[4] * mValues[14] - mValues[6] * mValues[12] ) ) );
            result.mValues[7] = invDet * ( mValues[0] * ( mValues[6] * mValues[11] - mValues[7] * mValues[10] ) - mValues[2] * ( mValues[4] * mValues[11] - mValues[7] * mValues[8] ) + mValues[3] * ( mValues[4] * mValues[10] - mValues[6] * mValues[8] ) );

            result.mValues[8] = invDet * ( mValues[4] * ( mValues[9] * mValues[15] - mValues[11] * mValues[13] ) - mValues[5] * ( mValues[8] * mValues[15] - mValues[11] * mValues[12] ) + mValues[7] * ( mValues[8] * mValues[13] - mValues[9] * mValues[12] ) );
            result.mValues[9] = invDet * ( - ( mValues[0] * ( mValues[9] * mValues[15] - mValues[11] * mValues[13] ) - mValues[1] * ( mValues[8] * mValues[15] - mValues[11] * mValues[12] ) + mValues[3] * ( mValues[8] * mValues[13] - mValues[9] * mValues[12] ) ) );
            result.mValues[10] = invDet * ( mValues[0] * ( mValues[5] * mValues[15] - mValues[7] * mValues[13] ) - mValues[1] * ( mValues[4] * mValues[15] - mValues[7] * mValues[12] ) + mValues[3] * ( mValues[4] * mValues[13] - mValues[5] * mValues[12] ) );
            result.mValues[11] = invDet * ( - ( mValues[0] * ( mValues[5] * mValues[11] - mValues[7] * mValues[9] ) - mValues[1] * ( mValues[4] * mValues[11] - mValues[7] * mValues[8] ) + mValues[3] * ( mValues[4] * mValues[9] - mValues[5] * mValues[8] ) ) );

            result.mValues[12] = invDet * ( - ( mValues[4] * ( mValues[9] * mValues[14] - mValues[10] * mValues[13] ) - mValues[5] * ( mValues[8] * mValues[14] - mValues[10] * mValues[12] ) + mValues[6] * ( mValues[8] * mValues[13] - mValues[9] * mValues[12] ) ) );
            result.mValues[13] = invDet * ( mValues[0] * ( mValues[9] * mValues[14] - mValues[10] * mValues[13] ) - mValues[1] * ( mValues[8] * mValues[14] - mValues[10] * mValues[12] ) + mValues[2] * ( mValues[8] * mValues[13] - mValues[9] * mValues[12] ) );
            result.mValues[14] = invDet * ( - ( mValues[0] * ( mValues[5] * mValues[14] - mValues[6] * mValues[13] ) - mValues[1] * ( mValues[4] * mValues[14] - mValues[6] * mValues[12] ) + mValues[2] * ( mValues[4] * mValues[13] - mValues[5] * mValues[12] ) ) );
            result.mValues[15] = invDet * ( mValues[0] * ( mValues[5] * mValues[10] - mValues[6] * mValues[9] ) - mValues[1] * ( mValues[4] * mValues[10] - mValues[6] * mValues[8] ) + mValues[2] * ( mValues[4] * mValues[9] - mValues[5] * mValues[8] ) );

            result.mIs2D = mIs2D;
            return result;
        }

        DOMMatrixReadOnly DOMMatrixReadOnly::multiply ( const DOMMatrixReadOnly& other ) const
        {
            DOMMatrixReadOnly result;

            result.mValues[0] = mValues[0] * other.mValues[0] + mValues[1] * other.mValues[4] + mValues[2] * other.mValues[8] + mValues[3] * other.mValues[12];
            result.mValues[1] = mValues[0] * other.mValues[1] + mValues[1] * other.mValues[5] + mValues[2] * other.mValues[9] + mValues[3] * other.mValues[13];
            result.mValues[2] = mValues[0] * other.mValues[2] + mValues[1] * other.mValues[6] + mValues[2] * other.mValues[10] + mValues[3] * other.mValues[14];
            result.mValues[3] = mValues[0] * other.mValues[3] + mValues[1] * other.mValues[7] + mValues[2] * other.mValues[11] + mValues[3] * other.mValues[15];

            result.mValues[4] = mValues[4] * other.mValues[0] + mValues[5] * other.mValues[4] + mValues[6] * other.mValues[8] + mValues[7] * other.mValues[12];
            result.mValues[5] = mValues[4] * other.mValues[1] + mValues[5] * other.mValues[5] + mValues[6] * other.mValues[9] + mValues[7] * other.mValues[13];
            result.mValues[6] = mValues[4] * other.mValues[2] + mValues[5] * other.mValues[6] + mValues[6] * other.mValues[10] + mValues[7] * other.mValues[14];
            result.mValues[7] = mValues[4] * other.mValues[3] + mValues[5] * other.mValues[7] + mValues[6] * other.mValues[11] + mValues[7] * other.mValues[15];

            result.mValues[8] = mValues[8] * other.mValues[0] + mValues[9] * other.mValues[4] + mValues[10] * other.mValues[8] + mValues[11] * other.mValues[12];
            result.mValues[9] = mValues[8] * other.mValues[1] + mValues[9] * other.mValues[5] + mValues[10] * other.mValues[9] + mValues[11] * other.mValues[13];
            result.mValues[10] = mValues[8] * other.mValues[2] + mValues[9] * other.mValues[6] + mValues[10] * other.mValues[10] + mValues[11] * other.mValues[14];
            result.mValues[11] = mValues[8] * other.mValues[3] + mValues[9] * other.mValues[7] + mValues[10] * other.mValues[11] + mValues[11] * other.mValues[15];

            result.mValues[12] = mValues[12] * other.mValues[0] + mValues[13] * other.mValues[4] + mValues[14] * other.mValues[8] + mValues[15] * other.mValues[12];
            result.mValues[13] = mValues[12] * other.mValues[1] + mValues[13] * other.mValues[5] + mValues[14] * other.mValues[9] + mValues[15] * other.mValues[13];
            result.mValues[14] = mValues[12] * other.mValues[2] + mValues[13] * other.mValues[6] + mValues[14] * other.mValues[10] + mValues[15] * other.mValues[14];
            result.mValues[15] = mValues[12] * other.mValues[3] + mValues[13] * other.mValues[7] + mValues[14] * other.mValues[11] + mValues[15] * other.mValues[15];

            result.mIs2D = mIs2D && other.mIs2D;
            return result;
        }

        DOMMatrixReadOnly DOMMatrixReadOnly::rotateAxisAngle ( float x, float y, float z, float angle ) const
        {
            DOMMatrixReadOnly rotation;
            float radians = angle * 3.14159265359f / 180.0f;
            float c = std::cos ( radians );
            float s = std::sin ( radians );
            float mag = std::sqrt ( x * x + y * y + z * z );

            if ( mag == 0.0f )
            {
                return *this;
            }

            x /= mag;
            y /= mag;
            z /= mag;

            float oneMinusC = 1.0f - c;

            rotation.mValues[0] = c + x * x * oneMinusC;
            rotation.mValues[1] = x * y * oneMinusC - z * s;
            rotation.mValues[2] = x * z * oneMinusC + y * s;
            rotation.mValues[3] = 0.0f;

            rotation.mValues[4] = y * x * oneMinusC + z * s;
            rotation.mValues[5] = c + y * y * oneMinusC;
            rotation.mValues[6] = y * z * oneMinusC - x * s;
            rotation.mValues[7] = 0.0f;

            rotation.mValues[8] = z * x * oneMinusC - y * s;
            rotation.mValues[9] = z * y * oneMinusC + x * s;
            rotation.mValues[10] = c + z * z * oneMinusC;
            rotation.mValues[11] = 0.0f;

            rotation.mValues[12] = 0.0f;
            rotation.mValues[13] = 0.0f;
            rotation.mValues[14] = 0.0f;
            rotation.mValues[15] = 1.0f;

            rotation.mIs2D = false;
            return multiply ( rotation );
        }

        DOMMatrixReadOnly DOMMatrixReadOnly::rotate ( float rotX, float rotY, float rotZ ) const
        {
            if ( rotZ != 0.0f )
            {
                if ( rotY != 0.0f )
                {
                    if ( rotX != 0.0f )
                    {
                        return rotateAxisAngle ( 0, 0, 1, rotZ ).rotateAxisAngle ( 0, 1, 0, rotY ).rotateAxisAngle ( 1, 0, 0, rotX );
                    }
                    return rotateAxisAngle ( 0, 0, 1, rotZ ).rotateAxisAngle ( 0, 1, 0, rotY );
                }
                if ( rotX != 0.0f )
                {
                    return rotateAxisAngle ( 0, 0, 1, rotZ ).rotateAxisAngle ( 1, 0, 0, rotX );
                }
                return rotateAxisAngle ( 0, 0, 1, rotZ );
            }

            if ( rotY != 0.0f )
            {
                if ( rotX != 0.0f )
                {
                    return rotateAxisAngle ( 0, 1, 0, rotY ).rotateAxisAngle ( 1, 0, 0, rotX );
                }
                return rotateAxisAngle ( 0, 1, 0, rotY );
            }

            if ( rotX != 0.0f )
            {
                return rotateAxisAngle ( 1, 0, 0, rotX );
            }

            return *this;
        }

        DOMMatrixReadOnly DOMMatrixReadOnly::rotateFromVector ( float x, float y ) const
        {
            float angle = std::atan2 ( y, x ) * 180.0f / 3.14159265359f;
            return rotateAxisAngle ( 0, 0, 1, angle );
        }

        DOMMatrixReadOnly DOMMatrixReadOnly::scale ( float scaleX, float scaleY, float scaleZ, float originX, float originY, float originZ ) const
        {
            DOMMatrixReadOnly translation1 = translate ( -originX, -originY, -originZ );

            DOMMatrixReadOnly scaling;
            scaling.mValues[0] = scaleX;
            scaling.mValues[5] = scaleY;
            scaling.mValues[10] = scaleZ;
            scaling.mIs2D = ( scaleZ == 1.0f && originZ == 0.0f && mIs2D );

            DOMMatrixReadOnly translation2 = DOMMatrixReadOnly().translate ( originX, originY, originZ );

            return translation1.multiply ( scaling ).multiply ( translation2 );
        }

        DOMMatrixReadOnly DOMMatrixReadOnly::scale3d ( float scale, float originX, float originY, float originZ ) const
        {
            return this->scale ( scale, scale, scale, originX, originY, originZ );
        }

        DOMMatrixReadOnly DOMMatrixReadOnly::skewX ( float sx ) const
        {
            DOMMatrixReadOnly skew;
            float radians = sx * 3.14159265359f / 180.0f;
            skew.mValues[4] = std::tan ( radians ); // m21
            skew.mIs2D = mIs2D;
            return multiply ( skew );
        }

        DOMMatrixReadOnly DOMMatrixReadOnly::skewY ( float sy ) const
        {
            DOMMatrixReadOnly skew;
            float radians = sy * 3.14159265359f / 180.0f;
            skew.mValues[1] = std::tan ( radians ); // m12
            skew.mIs2D = mIs2D;
            return multiply ( skew );
        }

        std::array<float, 16> DOMMatrixReadOnly::toFloat32Array() const
        {
            return mValues;
        }

        std::array<double, 16> DOMMatrixReadOnly::toFloat64Array() const
        {
            std::array<double, 16> result;
            for ( size_t i = 0; i < 16; ++i )
            {
                result[i] = static_cast<double> ( mValues[i] );
            }
            return result;
        }

        DOMString DOMMatrixReadOnly::toJSON() const
        {
            auto formatFloat = [] ( float value ) -> DOMString
            {
                DOMString str = std::to_string ( value );
                // Remove trailing zeros and decimal point if needed
                str.erase ( str.find_last_not_of ( '0' ) + 1, DOMString::npos );
                str.erase ( str.find_last_not_of ( '.' ) + 1, DOMString::npos );
                return str;
            };

            if ( mIs2D )
            {
                return "{\"a\":" + formatFloat ( mValues[0] ) + ",\"b\":" + formatFloat ( mValues[1] ) +
                       ",\"c\":" + formatFloat ( mValues[4] ) + ",\"d\":" + formatFloat ( mValues[5] ) +
                       ",\"e\":" + formatFloat ( mValues[12] ) + ",\"f\":" + formatFloat ( mValues[13] ) +
                       ",\"is2D\":" + ( mIs2D ? "true" : "false" ) +
                       ",\"isIdentity\":" + ( isIdentity() ? "true" : "false" ) + "}";
            }
            else
            {
                return "{\"m11\":" + formatFloat ( mValues[0] ) + ",\"m12\":" + formatFloat ( mValues[1] ) +
                       ",\"m13\":" + formatFloat ( mValues[2] ) + ",\"m14\":" + formatFloat ( mValues[3] ) +
                       ",\"m21\":" + formatFloat ( mValues[4] ) + ",\"m22\":" + formatFloat ( mValues[5] ) +
                       ",\"m23\":" + formatFloat ( mValues[6] ) + ",\"m24\":" + formatFloat ( mValues[7] ) +
                       ",\"m31\":" + formatFloat ( mValues[8] ) + ",\"m32\":" + formatFloat ( mValues[9] ) +
                       ",\"m33\":" + formatFloat ( mValues[10] ) + ",\"m34\":" + formatFloat ( mValues[11] ) +
                       ",\"m41\":" + formatFloat ( mValues[12] ) + ",\"m42\":" + formatFloat ( mValues[13] ) +
                       ",\"m43\":" + formatFloat ( mValues[14] ) + ",\"m44\":" + formatFloat ( mValues[15] ) +
                       ",\"is2D\":" + ( mIs2D ? "true" : "false" ) +
                       ",\"isIdentity\":" + ( isIdentity() ? "true" : "false" ) + "}";
            }
        }

        DOMString DOMMatrixReadOnly::toString() const
        {
            if ( mIs2D )
            {
                return "matrix(" + std::to_string ( mValues[0] ) + ", " + std::to_string ( mValues[1] ) + ", " +
                       std::to_string ( mValues[4] ) + ", " + std::to_string ( mValues[5] ) + ", " +
                       std::to_string ( mValues[12] ) + ", " + std::to_string ( mValues[13] ) + ")";
            }
            else
            {
                return "matrix3d(" + std::to_string ( mValues[0] ) + ", " + std::to_string ( mValues[1] ) + ", " +
                       std::to_string ( mValues[2] ) + ", " + std::to_string ( mValues[3] ) + ", " +
                       std::to_string ( mValues[4] ) + ", " + std::to_string ( mValues[5] ) + ", " +
                       std::to_string ( mValues[6] ) + ", " + std::to_string ( mValues[7] ) + ", " +
                       std::to_string ( mValues[8] ) + ", " + std::to_string ( mValues[9] ) + ", " +
                       std::to_string ( mValues[10] ) + ", " + std::to_string ( mValues[11] ) + ", " +
                       std::to_string ( mValues[12] ) + ", " + std::to_string ( mValues[13] ) + ", " +
                       std::to_string ( mValues[14] ) + ", " + std::to_string ( mValues[15] ) + ")";
            }
        }

        DOMPoint DOMMatrixReadOnly::transformPoint ( const DOMPoint& point ) const
        {
            float x = point.x() * mValues[0] + point.y() * mValues[4] + point.z() * mValues[8] + point.w() * mValues[12];
            float y = point.x() * mValues[1] + point.y() * mValues[5] + point.z() * mValues[9] + point.w() * mValues[13];
            float z = point.x() * mValues[2] + point.y() * mValues[6] + point.z() * mValues[10] + point.w() * mValues[14];
            float w = point.x() * mValues[3] + point.y() * mValues[7] + point.z() * mValues[11] + point.w() * mValues[15];

            return DOMPoint ( x, y, z, w );
        }

        DOMMatrixReadOnly DOMMatrixReadOnly::translate ( float x, float y, float z ) const
        {
            DOMMatrixReadOnly translation{};
            translation.mValues[12] = x;  // m41
            translation.mValues[13] = y;  // m42
            translation.mValues[14] = z;  // m43
            translation.mIs2D = ( z == 0.0f && mIs2D );
            return multiply ( translation );
        }
    }
}
