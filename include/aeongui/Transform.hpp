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
#ifndef AEONGUI_TRANSFORM_H
#define AEONGUI_TRANSFORM_H
#include <cstdint>
#include "aeongui/Platform.hpp"
#include "aeongui/Vector2.hpp"
#include "aeongui/Matrix2x3.hpp"

namespace AeonGUI
{
    class AABB;
    /** @brief 2D transform composed of scale, rotation, and translation.
     *
     *  Convenience class that decomposes a 2D affine transformation into
     *  its constituent components and can produce a Matrix2x3.
     */
    class Transform
    {
    public:
        /** @brief Default constructor. Identity transform. */
        DLL Transform();
        /** @brief Construct from scale, rotation, and translation.
         *  @param aScale       Scale factors per axis.
         *  @param aRotation    Rotation angle in radians.
         *  @param aTranslation Translation vector.
         */
        DLL Transform ( const Vector2& aScale, float aRotation, const Vector2 aTranslation );
        /** @brief Get the scale component.
         *  @return Const reference to the scale vector.
         */
        DLL const Vector2& GetScale() const;
        /** @brief Get the rotation angle.
         *  @return Rotation in radians.
         */
        DLL float GetRotation() const;
        /** @brief Get the translation component.
         *  @return Const reference to the translation vector.
         */
        DLL const Vector2& GetTranslation() const;
        /** @brief Build the equivalent 2x3 transformation matrix.
         *  @return The composed Matrix2x3.
         */
        DLL Matrix2x3 GetMatrix() const;

        /** @brief Set the scale component.
         *  @param aScale The new scale vector.
         */
        DLL void SetScale ( const Vector2& aScale );
        /** @brief Set the rotation angle.
         *  @param aRotation The new rotation in radians.
         */
        DLL void SetRotation ( float aRotation );
        /** @brief Set the translation component.
         *  @param aTranslation The new translation vector.
         */
        DLL void SetTranslation ( const Vector2& );
        /*! \name Operators */
        //@{
        /** @brief Combine this transform with another (post-multiply).
         *  @param aTransform The right-hand-side transform.
         *  @return Reference to this transform.
         */
        DLL Transform& operator*= ( const Transform& aTransform );
        //@}
    private:
        Vector2 mScale{1.0f, 1.0f};
        float mRotation{};
        Vector2 mTranslation{};
    };

    /** @brief Combine two transforms.
     *  @return The composed transform.
     */
    DLL const Transform operator* ( const Transform& aLeft, const Transform& aRight );
    /** @brief Transform an AABB.
     *  @return The transformed bounding box.
     */
    DLL const AABB operator* ( const Transform& aLeft, const AABB& aRight );
}
#endif
