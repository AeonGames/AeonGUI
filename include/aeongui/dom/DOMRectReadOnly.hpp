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
#ifndef AEONGUI_DOMRECTREADONLY_HPP
#define AEONGUI_DOMRECTREADONLY_HPP

#include "aeongui/Platform.hpp"
#include "DOMString.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Immutable rectangle.
         *
         *  Implements the DOM DOMRectReadOnly interface.
         *  @see https://drafts.fxtf.org/geometry/#domrectreadonly
         */
        class DLL DOMRectReadOnly
        {
        public:
            /** @brief Construct a rectangle.
             *  @param x      X coordinate of the origin.
             *  @param y      Y coordinate of the origin.
             *  @param width  Rectangle width.
             *  @param height Rectangle height.
             */
            DOMRectReadOnly ( float x = 0.0f, float y = 0.0f, float width = 0.0f, float height = 0.0f );
            /** @brief Destructor. */
            virtual ~DOMRectReadOnly();

            /** @brief Get the X coordinate.
             *  @return The X value. */
            float x() const;
            /** @brief Get the Y coordinate.
             *  @return The Y value. */
            float y() const;
            /** @brief Get the width.
             *  @return The width value. */
            float width() const;
            /** @brief Get the height.
             *  @return The height value. */
            float height() const;
            /** @brief Get the top edge (min of y and y+height).
             *  @return The top edge. */
            float top() const;
            /** @brief Get the right edge (max of x and x+width).
             *  @return The right edge. */
            float right() const;
            /** @brief Get the bottom edge (max of y and y+height).
             *  @return The bottom edge. */
            float bottom() const;
            /** @brief Get the left edge (min of x and x+width).
             *  @return The left edge. */
            float left() const;

            /** @brief Create a DOMRectReadOnly from any rect-like object.
             *  @tparam T A type with x(), y(), width(), and height() accessors.
             *  @param rect The source rectangle.
             *  @return A new DOMRectReadOnly.
             */
            template <typename T>
            static DOMRectReadOnly fromRect ( const T& rect )
            {
                return DOMRectReadOnly ( rect.x(), rect.y(), rect.width(), rect.height() );
            }

            /** @brief Serialize to JSON.
             *  @return A JSON string representation.
             */
            DOMString toJSON() const;

        protected:
            float mX{}; ///< X coordinate.
            float mY{}; ///< Y coordinate.
            float mWidth{}; ///< Rectangle width.
            float mHeight{}; ///< Rectangle height.
        };
    }
}

#endif
