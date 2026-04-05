/*
Copyright (C) 2019,2025,2026 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/Platform.hpp"
#include "aeongui/Vector2.hpp"
namespace AeonGUI
{
    /** @brief Axis-Aligned Bounding Box.
     *
     *  Represents a 2D axis-aligned bounding box defined by a center point and radii (half-extents).
     */
    class AABB
    {
    public:
        /** @brief Default constructor. Initializes center and radii to zero. */
        AEONGUI_DLL AABB();
        /** @brief Construct an AABB from a center point and radii.
         *  @param aCenter The center of the bounding box.
         *  @param aRadii  The half-extents (radii) along each axis.
         */
        AEONGUI_DLL AABB ( const Vector2& aCenter, const Vector2& aRadii );
        /** @brief Get the center of the bounding box.
         *  @return Reference to the center vector.
         */
        AEONGUI_DLL const Vector2& GetCenter() const;
        /** @brief Get the radii (half-extents) of the bounding box.
         *  @return Reference to the radii vector.
         */
        AEONGUI_DLL const Vector2& GetRadii() const;
        /** @brief Get the X coordinate of the top-left corner.
         *  @return X position (center.x - radii.x).
         */
        AEONGUI_DLL double GetX() const;
        /** @brief Get the Y coordinate of the top-left corner.
         *  @return Y position (center.y - radii.y).
         */
        AEONGUI_DLL double GetY() const;
        /** @brief Get the width of the bounding box.
         *  @return Width (2 * radii.x).
         */
        AEONGUI_DLL double GetWidth() const;
        /** @brief Get the height of the bounding box.
         *  @return Height (2 * radii.y).
         */
        AEONGUI_DLL double GetHeight() const;
    private:
        Vector2 mCenter{};
        Vector2 mRadii{};
    };
}
#endif
