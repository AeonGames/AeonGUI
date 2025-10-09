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
#ifndef AEONGUI_DOMPOINT_HPP
#define AEONGUI_DOMPOINT_HPP

#include "aeongui/Platform.hpp"
#include "aeongui/dom/DOMPointReadOnly.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class DLL DOMPoint : public DOMPointReadOnly
        {
        public:
            DOMPoint ( float x = 0.0f, float y = 0.0f, float z = 0.0f, float w = 1.0f );
            virtual ~DOMPoint() final;
            using DOMPointReadOnly::x;
            using DOMPointReadOnly::y;
            using DOMPointReadOnly::z;
            using DOMPointReadOnly::w;
            template <typename T>
            static DOMPoint fromPoint ( const T& point )
            {
                return DOMPoint ( point.x(), point.y(), point.z(), point.w() );
            }
            float x ( float newX );
            float y ( float newY );
            float z ( float newZ );
            float w ( float newW );
        };
    }
}

#endif
