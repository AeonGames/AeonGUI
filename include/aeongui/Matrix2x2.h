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
#ifndef AEONGUI_MATRIX2X2_H
#define AEONGUI_MATRIX2X2_H
#include <array>
#include <cstdint>
#include "aeongui/Platform.h"
namespace AeonGUI
{
    class Vector2;
    class Matrix2x2
    {
    public:
        DLL Matrix2x2();
        DLL Matrix2x2 ( const std::array<const float, 4> aMatrixArray );
        DLL Matrix2x2 ( float aRotation );
        DLL Matrix2x2 ( const Vector2& aScale );
        DLL Matrix2x2 ( const Vector2& aScale, float aRotation );
        DLL Matrix2x2& operator*= ( const Matrix2x2& aRight );
        DLL const float& operator[] ( size_t aIndex ) const;
    private:
        float mMatrix2x2[4];
    };
    const Matrix2x2 Abs ( const Matrix2x2& aMatrix2x2 );
}
#endif
