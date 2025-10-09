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
#include <format>
#include <algorithm>
#include "aeongui/dom/DOMRectReadOnly.hpp"
namespace AeonGUI
{
    namespace DOM
    {
        DOMRectReadOnly::DOMRectReadOnly ( float x, float y, float width, float height )
            : mX ( x ), mY ( y ), mWidth ( width ), mHeight ( height )
        {
        }
        DOMRectReadOnly::~DOMRectReadOnly() = default;
        float DOMRectReadOnly::x() const
        {
            return mX;
        }
        float DOMRectReadOnly::y() const
        {
            return mY;
        }
        float DOMRectReadOnly::width() const
        {
            return mWidth;
        }
        float DOMRectReadOnly::height() const
        {
            return mHeight;
        }
        float DOMRectReadOnly::top() const
        {
            return mY;
        }
        float DOMRectReadOnly::right() const
        {
            return mX + mWidth;
        }
        float DOMRectReadOnly::bottom() const
        {
            return mY + mHeight;
        }
        float DOMRectReadOnly::left() const
        {
            return mX;
        }
        DOMString DOMRectReadOnly::toJSON() const
        {
            return std::format ( R"({{"x": {:.10f}, "y": {:.10f}, "width": {:.10f}, "height": {:.10f}}})",
                                 mX, mY, mWidth, mHeight );
        }
    }
}
