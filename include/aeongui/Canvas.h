/*
Copyright (C) 2019,2020,2024 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_CANVAS_H
#define AEONGUI_CANVAS_H
#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>
#include "aeongui/Platform.h"
#include "aeongui/DrawType.h"
#include "aeongui/Color.h"
#include "aeongui/Attribute.hpp"
#include "aeongui/Matrix2x3.h"
namespace AeonGUI
{
    class Path;
    class Canvas
    {
    public:
        virtual void ResizeViewport ( uint32_t aWidth, uint32_t aHeight ) = 0;
        virtual const uint8_t* GetPixels() const = 0;
        virtual size_t GetWidth() const = 0;
        virtual size_t GetHeight() const = 0;
        virtual size_t GetStride() const = 0;
        virtual void Clear() = 0;
        virtual void SetFillColor ( const ColorAttr& aColor ) = 0;
        virtual const ColorAttr& GetFillColor() const = 0;
        virtual void SetStrokeColor ( const ColorAttr& aColor ) = 0;
        virtual const ColorAttr& GetStrokeColor() const = 0;
        virtual void SetStrokeWidth ( double aWidth ) = 0;
        virtual double GetStrokeWidth () const = 0;
        virtual void SetStrokeOpacity ( double aWidth ) = 0;
        virtual double GetStrokeOpacity () const = 0;
        virtual void SetFillOpacity ( double aWidth ) = 0;
        virtual double GetFillOpacity () const = 0;
        virtual void SetOpacity ( double aWidth ) = 0;
        virtual double GetOpacity () const = 0;
        virtual void Draw ( const Path& ) = 0;
        virtual void SetViewBox ( const ViewBox& aViewBox, const PreserveAspectRatio& aPreserveAspectRatio ) = 0;
        virtual void SetTransform ( const Matrix2x3& aMatrix ) = 0;
        virtual void Transform ( const Matrix2x3& aMatrix ) = 0;
        DLL virtual ~Canvas() = 0;
    };
}
#endif
