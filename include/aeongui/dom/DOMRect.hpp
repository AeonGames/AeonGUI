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
#ifndef AEONGUI_DOMRECT_HPP
#define AEONGUI_DOMRECT_HPP

#include "aeongui/Platform.hpp"
#include "aeongui/dom/DOMRectReadOnly.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Mutable rectangle.
         *
         *  Extends DOMRectReadOnly with setters for all components.
         *  @see https://drafts.fxtf.org/geometry/#domrect
         */
        class AEONGUI_DLL DOMRect : public DOMRectReadOnly
        {
        public:
            /** @brief Construct a rectangle.
             *  @param x      X coordinate.
             *  @param y      Y coordinate.
             *  @param width  Width.
             *  @param height Height.
             */
            DOMRect ( float x = 0.0f, float y = 0.0f, float width = 0.0f, float height = 0.0f );
            /** @brief Destructor. */
            virtual ~DOMRect() final;

            using DOMRectReadOnly::x;
            using DOMRectReadOnly::y;
            using DOMRectReadOnly::width;
            using DOMRectReadOnly::height;
            using DOMRectReadOnly::top;
            using DOMRectReadOnly::right;
            using DOMRectReadOnly::bottom;
            using DOMRectReadOnly::left;

            /** @brief Set the X coordinate. @param newX New value. @return The set value. */
            float x ( float newX );
            /** @brief Set the Y coordinate. @param newY New value. @return The set value. */
            float y ( float newY );
            /** @brief Set the width. @param newWidth New value. @return The set value. */
            float width ( float newWidth );
            /** @brief Set the height. @param newHeight New value. @return The set value. */
            float height ( float newHeight );
            /** @brief Set the top edge. @param newTop New value. @return The set value. */
            float top ( float newTop );
            /** @brief Set the right edge. @param newRight New value. @return The set value. */
            float right ( float newRight );
            /** @brief Set the bottom edge. @param newBottom New value. @return The set value. */
            float bottom ( float newBottom );
            /** @brief Set the left edge. @param newLeft New value. @return The set value. */
            float left ( float newLeft );

            /** @brief Create a DOMRect from any rect-like object.
             *  @tparam T A type with x(), y(), width(), and height() accessors.
             *  @param rect The source rectangle.
             *  @return A new DOMRect.
             */
            template <typename T>
            static DOMRect fromRect ( const T& rect )
            {
                return DOMRect ( rect.x(), rect.y(), rect.width(), rect.height() );
            }
        };
    }
}

#endif
