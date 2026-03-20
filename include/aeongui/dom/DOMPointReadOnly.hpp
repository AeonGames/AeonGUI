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
#ifndef AEONGUI_DOMPOINTREADONLY_HPP
#define AEONGUI_DOMPOINTREADONLY_HPP

#include "aeongui/Platform.hpp"
#include "DOMString.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class DOMMatrixReadOnly;
        /** @brief Immutable 3D point with a perspective component.
         *
         *  Implements the DOM DOMPointReadOnly interface.
         *  @see https://drafts.fxtf.org/geometry/#dompointreadonly
         */
        class DLL DOMPointReadOnly
        {
        public:
            /** @brief Construct a point.
             *  @param x X coordinate (default 0).
             *  @param y Y coordinate (default 0).
             *  @param z Z coordinate (default 0).
             *  @param w Perspective component (default 1).
             */
            DOMPointReadOnly ( float x = 0.0f, float y = 0.0f, float z = 0.0f, float w = 1.0f );
            /** @brief Destructor. */
            virtual ~DOMPointReadOnly();
            /** @brief Get the X coordinate. */
            float x() const;
            /** @brief Get the Y coordinate. */
            float y() const;
            /** @brief Get the Z coordinate. */
            float z() const;
            /** @brief Get the W (perspective) component. */
            float w() const;

            /** @brief Create a DOMPointReadOnly from any point-like object.
             *  @tparam T A type with x(), y(), z(), and w() accessors.
             *  @param point The source point.
             *  @return A new DOMPointReadOnly.
             */
            template <typename T>
            static DOMPointReadOnly fromPoint ( const T& point )
            {
                return DOMPointReadOnly ( point.x(), point.y(), point.z(), point.w() );
            }
            /** @brief Transform this point by a matrix.
             *  @param matrix The transformation matrix.
             *  @return The transformed point.
             */
            DOMPointReadOnly matrixTransform ( const DOMMatrixReadOnly& matrix ) const;
            /** @brief Serialize to JSON.
             *  @return A JSON string representation.
             */
            DOMString toJSON() const;
        protected:
            float mX{};
            float mY{};
            float mZ{};
            float mW{};
        };
    }
}

#endif
