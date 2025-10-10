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
#include "aeongui/dom/DOMPointReadOnly.hpp"
#include <format>

namespace AeonGUI
{
    namespace DOM
    {
        DOMPointReadOnly::DOMPointReadOnly ( float x, float y, float z, float w )
            : mX ( x ), mY ( y ), mZ ( z ), mW ( w ) {}

        DOMPointReadOnly::~DOMPointReadOnly() = default;

        float DOMPointReadOnly::x() const
        {
            return mX;
        }
        float DOMPointReadOnly::y() const
        {
            return mY;
        }
        float DOMPointReadOnly::z() const
        {
            return mZ;
        }
        float DOMPointReadOnly::w() const
        {
            return mW;
        }

        DOMString DOMPointReadOnly::toJSON() const
        {
            // Convert the point to a JSON representation
            return std::format ( R"({{"x": {:.10f}, "y": {:.10f}, "z": {:.10f}, "w": {:.10f}}})", mX, mY, mZ, mW );
        }

        DOMPointReadOnly DOMPointReadOnly::matrixTransform ( const DOMMatrixReadOnly& matrix ) const
        {
#if 0
            // Apply the matrix transformation to the point
            float x = mX * matrix.a() + mY * matrix.c() + mZ * matrix.e() + mW * matrix.g();
            float y = mX * matrix.b() + mY * matrix.d() + mZ * matrix.f() + mW * matrix.h();
            float z = mX * matrix.i() + mY * matrix.j() + mZ * matrix.k() + mW * matrix.l();
            float w = mX * matrix.m() + mY * matrix.n() + mZ * matrix.o() + mW * matrix.p();
            return DOMPointReadOnly ( x, y, z, w );
#else
            return DOMPointReadOnly ( mX, mY, mZ, mW );
#endif
        }
    }
}
