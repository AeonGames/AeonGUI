/*
Copyright (C) 2025 Rodrigo Jose Hernandez Cordoba

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
        class DLL DOMRect : public DOMRectReadOnly
        {
        public:
            DOMRect ( float x = 0.0f, float y = 0.0f, float width = 0.0f, float height = 0.0f );
            virtual ~DOMRect() final;

            using DOMRectReadOnly::x;
            using DOMRectReadOnly::y;
            using DOMRectReadOnly::width;
            using DOMRectReadOnly::height;
            using DOMRectReadOnly::top;
            using DOMRectReadOnly::right;
            using DOMRectReadOnly::bottom;
            using DOMRectReadOnly::left;

            float x ( float newX );
            float y ( float newY );
            float width ( float newWidth );
            float height ( float newHeight );
            float top ( float newTop );
            float right ( float newRight );
            float bottom ( float newBottom );
            float left ( float newLeft );

            template <typename T>
            static DOMRect fromRect ( const T& rect )
            {
                return DOMRect ( rect.x(), rect.y(), rect.width(), rect.height() );
            }
        };
    }
}

#endif
