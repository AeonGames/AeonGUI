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
#include "aeongui/Vector2.h"
#include "aeongui/Matrix2x3.h"
namespace AeonGUI
{
    Vector2::Vector2()
    {
        mVector2[0] = 0.0f;
        mVector2[1] = 0.0f;
    }
    Vector2::Vector2 ( float aX, float aY )
    {
        mVector2[0] = aX;
        mVector2[1] = aY;
    }

    float Vector2::GetX() const
    {
        return mVector2[0];
    }
    float Vector2::GetY() const
    {
        return mVector2[1];
    }
    const float& Vector2::operator[] ( std::size_t aIndex ) const
    {
        return mVector2[aIndex];
    }

    Vector2& Vector2::operator+= ( const Vector2& aRight )
    {
        mVector2[0] += aRight.mVector2[0];
        mVector2[1] += aRight.mVector2[1];
        return *this;
    }

    Vector2 operator+ ( const Vector2& aLeft, const Vector2& aRight )
    {
        return Vector2 { aLeft } += aRight;
    }

    Vector2& Vector2::operator*= ( const Matrix2x3& aRight )
    {
        Vector2 local{*this};
        mVector2[0] = local[0] * aRight[0] + local[1] * aRight[2] + aRight[4];
        mVector2[1] = local[0] * aRight[1] + local[1] * aRight[3] + aRight[5];
        return *this;
    }

    Vector2 operator* ( const Vector2& aLeft, const Matrix2x3& aRight )
    {
        return Vector2 { aLeft } *= aRight;
    }
}
