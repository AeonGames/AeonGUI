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
#ifndef AEONGUI_DOMPOINTREADONLY_HPP
#define AEONGUI_DOMPOINTREADONLY_HPP

#include "aeongui/Platform.hpp"
#include "DOMString.hpp"
#include "DOMMatrixReadOnly.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class DLL DOMPointReadOnly
        {
        public:
            DOMPointReadOnly ( float x = 0.0f, float y = 0.0f, float z = 0.0f, float w = 1.0f );
            virtual ~DOMPointReadOnly();
            float x() const;
            float y() const;
            float z() const;
            float w() const;

            template <typename T>
            static DOMPointReadOnly fromPoint ( const T& point )
            {
                return DOMPointReadOnly ( point.x(), point.y(), point.z(), point.w() );
            }
            DOMPointReadOnly matrixTransform ( const DOMMatrixReadOnly& matrix ) const;
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
