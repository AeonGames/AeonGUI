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
#ifndef AEONGUI_TRANSFORM_H
#define AEONGUI_TRANSFORM_H
#include <cstdint>
#include "aeongui/Platform.h"
#include "aeongui/Vector2.h"
#include "aeongui/Matrix3x3.h"

namespace AeonGUI
{
    class AABB;
    class Transform
    {
    public:
        DLL Transform();
        DLL Transform ( const Vector2& aScale, float aRotation, const Vector2 aTranslation );
        DLL const Vector2& GetScale() const;
        DLL float GetRotation() const;
        DLL const Vector2& GetTranslation() const;
        DLL Matrix3x3 GetMatrix() const;

        DLL void SetScale ( const Vector2& aScale );
        DLL void SetRotation ( float aRotation );
        DLL void SetTranslation ( const Vector2& );
        /*! \name Operators */
        //@{
        DLL Transform& operator*= ( const Transform& aTransform );
        //@}
    private:
        Vector2 mScale{1.0f, 1.0f};
        float mRotation{};
        Vector2 mTranslation{};
    };

    DLL const Transform operator* ( const Transform& aLeft, const Transform& aRight );
    DLL const AABB operator* ( const Transform& aLeft, const AABB& aRight );
}
#endif
