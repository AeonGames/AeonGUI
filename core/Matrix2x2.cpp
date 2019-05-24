/*
Copyright (C) 2019 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/Matrix2x2.h"
#include "aeongui/Vector2.h"
#include <cmath>
#include <cstring>
namespace AeonGUI
{
    Matrix2x2::Matrix2x2()
    {
        mMatrix2x2[0] = mMatrix2x2[3] = 1.0f;
        mMatrix2x2[1] = mMatrix2x2[2] = 0.0f;
    }

    Matrix2x2::Matrix2x2 ( const std::array<const float, 4> aMatrixArray )
    {
        memcpy ( mMatrix2x2, aMatrixArray.data(), sizeof ( float ) * 4 );
    }

    Matrix2x2::Matrix2x2 ( float aRotation )
    {
        float radians = aRotation * 180.0f / M_PI;
        mMatrix2x2[0] = mMatrix2x2[3] = std::cos ( radians );
        mMatrix2x2[1] = -std::sin ( radians );
        mMatrix2x2[2] = std::sin ( radians );
    }
    Matrix2x2::Matrix2x2 ( const Vector2& aScale )
    {
        mMatrix2x2[0] = aScale[0];
        mMatrix2x2[3] = aScale[1];
        mMatrix2x2[1] = mMatrix2x2[2] = 0.0f;
    }
    Matrix2x2::Matrix2x2 ( const Vector2& aScale, float aRotation ) : Matrix2x2{aScale}
    {
        Matrix2x2 rotation ( aRotation );
        *this *= rotation;
    }

    Matrix2x2& Matrix2x2::operator*= ( const Matrix2x2& aRight )
    {
        const Matrix2x2 local{*this};
        mMatrix2x2[0] = local[0] * aRight[0] + local[1] * aRight[2];
        mMatrix2x2[1] = local[0] * aRight[1] + local[1] * aRight[3];
        mMatrix2x2[2] = local[2] * aRight[0] + local[3] * aRight[2];
        mMatrix2x2[3] = local[2] * aRight[1] + local[3] * aRight[3];
        return *this;
    }

    const float& Matrix2x2::operator[] ( size_t aIndex ) const
    {
        return mMatrix2x2[aIndex];
    }

    const Matrix2x2 Abs ( const Matrix2x2& aMatrix2x2 )
    {
        return Matrix2x2
        {
            {
                std::abs ( aMatrix2x2[0] ),
                std::abs ( aMatrix2x2[1] ),
                std::abs ( aMatrix2x2[2] ),
                std::abs ( aMatrix2x2[3] )
            }
        };
    }
}
