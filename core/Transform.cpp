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
#include "aeongui/Transform.h"
#include "aeongui/Matrix2x3.h"
#include "aeongui/AABB.h"

namespace AeonGUI
{
    Transform::Transform() : mScale{1.0f, 1.0f}, mRotation{}, mTranslation{}
    {}
    Transform::Transform ( const Vector2& aScale, float aRotation, const Vector2 aTranslation ) :
        mScale{aScale}, mRotation{aRotation}, mTranslation{aTranslation}
    {}

    const Vector2& Transform::GetScale() const
    {
        return mScale;
    }

    float Transform::GetRotation() const
    {
        return mRotation;
    }

    const Vector2& Transform::GetTranslation() const
    {
        return mTranslation;
    }

    Matrix2x3 Transform::GetMatrix() const
    {
        return Matrix2x3{mScale, mRotation, mTranslation};
    }

    void Transform::SetScale ( const Vector2& aScale )
    {
        mScale = aScale;
    }
    void Transform::SetRotation ( float aRotation )
    {
        mRotation = aRotation;
    }
    void Transform::SetTranslation ( const Vector2& aTranslation )
    {
        mTranslation = aTranslation;
    }

    Transform& Transform::operator *= ( const Transform& aRight )
    {
        mTranslation += ( aRight.GetTranslation() * Matrix2x3 ( mRotation ) );
        mRotation += aRight.GetRotation();
        mScale *= aRight.GetScale();
        return *this;
    }

    const Transform operator* ( const Transform& aLeft, const Transform& aRight )
    {
        return Transform { aLeft } *= aRight;
    }

    const AABB operator* ( const Transform & aLeft, const AABB & aRight )
    {
        Matrix2x3 scale_rotation{aLeft.GetScale(), aLeft.GetRotation() };
        return AABB
        {
            aLeft.GetTranslation() + ( aRight.GetCenter() * scale_rotation ),
            aRight.GetRadii() * Abs ( scale_rotation )
        };
    }
}
