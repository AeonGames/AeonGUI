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
#ifndef AEONGUI_CANVAS_H
#define AEONGUI_CANVAS_H
#include <cstdint>
#include "aeongui/Platform.h"
namespace AeonGUI
{
    class Canvas
    {
    public:
        virtual void ResizeViewport ( uint32_t aWidth, uint32_t aHeight ) = 0;
        virtual const uint8_t* GetPixels() const = 0;
        virtual size_t GetWidth() const = 0;
        virtual size_t GetHeight() const = 0;
        virtual size_t GetStride() const = 0;
        virtual void Clear() = 0;
        virtual ~Canvas() = 0;
    };
}
#endif
