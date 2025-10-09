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
#include "aeongui/dom/DOMPoint.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        DOMPoint::DOMPoint ( float x, float y, float z, float w ) : DOMPointReadOnly ( x, y, z, w )
        {
        }

        DOMPoint::~DOMPoint() = default;

        float DOMPoint::x ( float newX )
        {
            mX = newX;
            return mX;
        }

        float DOMPoint::y ( float newY )
        {
            mY = newY;
            return mY;
        }

        float DOMPoint::z ( float newZ )
        {
            mZ = newZ;
            return mZ;
        }

        float DOMPoint::w ( float newW )
        {
            mW = newW;
            return mW;
        }
    }
}
