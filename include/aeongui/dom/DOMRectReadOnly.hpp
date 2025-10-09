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
#ifndef AEONGUI_DOMRECTREADONLY_HPP
#define AEONGUI_DOMRECTREADONLY_HPP

#include "aeongui/Platform.hpp"
#include "DOMString.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class DLL DOMRectReadOnly
        {
        public:
            DOMRectReadOnly ( float x = 0.0f, float y = 0.0f, float width = 0.0f, float height = 0.0f );
            virtual ~DOMRectReadOnly();

            float x() const;
            float y() const;
            float width() const;
            float height() const;
            float top() const;
            float right() const;
            float bottom() const;
            float left() const;

            template <typename T>
            static DOMRectReadOnly fromRect ( const T& rect )
            {
                return DOMRectReadOnly ( rect.x(), rect.y(), rect.width(), rect.height() );
            }

            DOMString toJSON() const;

        protected:
            float mX{};
            float mY{};
            float mWidth{};
            float mHeight{};
        };
    }
}

#endif
