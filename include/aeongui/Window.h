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
#ifndef AEONGUI_WINDOW_H
#define AEONGUI_WINDOW_H
#include <cstdint>
#include <vector>
#include <memory>
#include <algorithm>
#include <string>
#include "aeongui/Document.h"
#include "aeongui/Platform.h"
///@todo Canvas implementation should be selectable.
#include "aeongui/CairoCanvas.h"

namespace AeonGUI
{
    class Window
    {
    public:
        DLL Window ( const std::string aFilename, uint32_t aWidth, uint32_t aHeight );
        DLL void ResizeViewport ( uint32_t aWidth, uint32_t aHeight );
        DLL const uint8_t* GetPixels() const;
        DLL size_t GetWidth() const;
        DLL size_t GetHeight() const;
        DLL size_t GetStride() const;
        DLL void Draw();
    private:
        Document mDocument;
        CairoCanvas mCanvas;
    };
}
#endif
