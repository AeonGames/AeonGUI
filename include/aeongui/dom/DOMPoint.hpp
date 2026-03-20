/*
Copyright (C) 2025,2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_DOMPOINT_HPP
#define AEONGUI_DOMPOINT_HPP

#include "aeongui/Platform.hpp"
#include "aeongui/dom/DOMPointReadOnly.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Mutable 3D point with a perspective component.
         *
         *  Extends DOMPointReadOnly with setters for x, y, z, and w.
         *  @see https://drafts.fxtf.org/geometry/#dompoint
         */
        class DLL DOMPoint : public DOMPointReadOnly
        {
        public:
            /** @brief Construct a point.
             *  @param x X coordinate (default 0).
             *  @param y Y coordinate (default 0).
             *  @param z Z coordinate (default 0).
             *  @param w Perspective component (default 1).
             */
            DOMPoint ( float x = 0.0f, float y = 0.0f, float z = 0.0f, float w = 1.0f );
            /** @brief Destructor. */
            virtual ~DOMPoint() final;
            using DOMPointReadOnly::x;
            using DOMPointReadOnly::y;
            using DOMPointReadOnly::z;
            using DOMPointReadOnly::w;
            /** @brief Create a DOMPoint from any point-like object.
             *  @tparam T A type with x(), y(), z(), and w() accessors.
             *  @param point The source point.
             *  @return A new DOMPoint.
             */
            template <typename T>
            static DOMPoint fromPoint ( const T& point )
            {
                return DOMPoint ( point.x(), point.y(), point.z(), point.w() );
            }
            /** @brief Set the X coordinate.
             *  @param newX The new X value.
             *  @return The set value.
             */
            float x ( float newX );
            /** @brief Set the Y coordinate.
             *  @param newY The new Y value.
             *  @return The set value.
             */
            float y ( float newY );
            /** @brief Set the Z coordinate.
             *  @param newZ The new Z value.
             *  @return The set value.
             */
            float z ( float newZ );
            /** @brief Set the W component.
             *  @param newW The new W value.
             *  @return The set value.
             */
            float w ( float newW );
        };
    }
}

#endif
