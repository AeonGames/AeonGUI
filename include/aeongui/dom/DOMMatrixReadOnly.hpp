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
#ifndef AEONGUI_DOMMATRIXREADONLY_HPP
#define AEONGUI_DOMMATRIXREADONLY_HPP

#include <initializer_list>
#include <array>
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class DLL DOMMatrixReadOnly
        {
        public:
            // Constructor from initializer list
            DOMMatrixReadOnly ( std::initializer_list<float> values = {1, 0, 0, 1, 0, 0} );
            virtual ~DOMMatrixReadOnly();
            bool is2D() const;
            bool isIdentity() const;
            // Accessors for matrix values
            float a() const;
            float b() const;
            float c() const;
            float d() const;
            float e() const;
            float f() const;

            float m11() const;
            float m12() const;
            float m13() const;
            float m14() const;
            float m21() const;
            float m22() const;
            float m23() const;
            float m24() const;
            float m31() const;
            float m32() const;
            float m33() const;
            float m34() const;
            float m41() const;
            float m42() const;
            float m43() const;
            float m44() const;

        protected:
            std::array<float, 16> mValues{};
            bool mIs2D{};
        };
    }
}

#endif
