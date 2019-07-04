/*
Copyright (C) 2019 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_VECTOR2_H
#define AEONGUI_VECTOR2_H
#include <cstddef>
#include <cstdint>
#include "aeongui/Platform.h"
namespace AeonGUI
{
    class Matrix2x3;
    class Vector2
    {
    public:
        DLL Vector2();
        DLL Vector2 ( double aX, double aY );
        DLL double GetX() const;
        DLL double GetY() const;
        DLL double Length() const;
        DLL const double& operator[] ( std::size_t aIndex ) const;
        DLL Vector2& operator+= ( const Vector2& aRight );
        DLL Vector2& operator-= ( const Vector2& aRight );
        DLL Vector2& operator*= ( const Matrix2x3& aRight );
        DLL Vector2& operator*= ( const Vector2& aRight );
        DLL Vector2& operator*= ( double aRight );
        DLL Vector2& operator/= ( double aRight );
    private:
        double mVector2[2];
    };
    DLL Vector2 operator+ ( const Vector2& aLeft, const Vector2& aRight );
    DLL Vector2 operator- ( const Vector2& aLeft, const Vector2& aRight );
    DLL Vector2 operator* ( const Vector2& aLeft, const Matrix2x3& aRight );
    DLL Vector2 operator* ( const Vector2& aLeft, const Vector2& aRight );
    DLL Vector2 operator/ ( const Vector2& aLeft, double aRight );
    DLL Vector2 operator* ( const Vector2& aLeft, double aRight );
    DLL Vector2 Abs ( const Vector2& aVector2 );
    DLL double Dot ( const Vector2& aLeft, const Vector2& aRight );
    DLL double Distance ( const Vector2& aLeft, const Vector2& aRight );
}
#endif
