/*
Copyright (C) 2019,2025 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/AABB.hpp"
namespace AeonGUI
{
    AABB::AABB() = default;
    AABB::AABB ( const Vector2& aCenter, const Vector2& aRadii ) : mCenter{aCenter}, mRadii{aRadii} {}
    double AABB::GetX() const
    {
        return mCenter.GetX() - mRadii.GetX();
    }
    double AABB::GetY() const
    {
        return mCenter.GetY() - mRadii.GetY();
    }
    double AABB::GetWidth() const
    {
        return mRadii.GetX() * 2;
    }
    double AABB::GetHeight() const
    {
        return mRadii.GetY() * 2;
    }
    const Vector2& AABB::GetCenter() const
    {
        return mCenter;
    }
    const Vector2& AABB::GetRadii() const
    {
        return mRadii;
    }
}
