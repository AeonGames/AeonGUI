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
#ifndef AEONGUI_AABB_H
#define AEONGUI_AABB_H
#include "aeongui/Platform.h"
#include "aeongui/Vector2.h"
namespace AeonGUI
{
    class AABB
    {
    public:
        DLL AABB();
        DLL AABB ( const Vector2& aCenter, const Vector2& aRadii );
        DLL const Vector2& GetCenter() const;
        DLL const Vector2& GetRadii() const;
        DLL double GetX() const;
        DLL double GetY() const;
        DLL double GetWidth() const;
        DLL double GetHeight() const;
    private:
        Vector2 mCenter{};
        Vector2 mRadii{};
    };
}
#endif
