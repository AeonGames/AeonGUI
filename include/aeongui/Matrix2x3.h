/*
Copyright (C) 2019,2024 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_MATRIX2X3_H
#define AEONGUI_MATRIX2X3_H
#include <array>
#include <cstdint>
#include "aeongui/Platform.h"
namespace AeonGUI
{
    class Vector2;
    class Matrix2x3
    {
    public:
        DLL Matrix2x3();
        DLL Matrix2x3 (
            double xx, double yx,
            double xy, double yy,
            double x0, double y0
        );
        DLL Matrix2x3 ( const std::array<const double, 6> aMatrixArray );
        DLL Matrix2x3 ( double aRotation );
        DLL Matrix2x3 ( const Vector2& aScale );
        DLL Matrix2x3 ( const Vector2& aScale, double aRotation );
        DLL Matrix2x3 ( const Vector2& aScale, double aRotation, const Vector2& aTranslation );
        DLL Matrix2x3& operator*= ( const Matrix2x3& aRight );
        DLL const double& operator[] ( size_t aIndex ) const;
    private:
        double mMatrix2x3[6];
    };
    const Matrix2x3 Abs ( const Matrix2x3& aMatrix2x3 );
}
#endif
