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
#ifndef AEONGUI_CAIROCANVAS_H
#define AEONGUI_CAIROCANVAS_H
#include <cstdint>
#include "aeongui/Canvas.h"

struct _cairo_surface;
struct _cairo;
typedef struct _cairo_surface cairo_surface_t;
typedef struct _cairo cairo_t;

namespace AeonGUI
{
    class CairoCanvas : public Canvas
    {
    public:
        CairoCanvas ( uint32_t aWidth, uint32_t aHeight );
        void ResizeViewport ( uint32_t aWidth, uint32_t aHeight ) final;
        const uint8_t* GetPixels() const final;
        size_t GetWidth() const final;
        size_t GetHeight() const final;
        size_t GetStride() const final;
        void Clear() final;
        void Draw ( const std::vector<DrawType>& aCommands ) final;
        DLL ~CairoCanvas() final;
    private:
        cairo_surface_t* mCairoSurface{};
        cairo_t* mCairoContext{};
    };
}
#endif
