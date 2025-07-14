/*
Copyright (C) 2019,2020,2023,2025 Rodrigo Jose Hernandez Cordoba

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
#include <string>
#include "aeongui/Platform.h"
#include "aeongui/CairoCanvas.h"
#include "aeongui/dom/EventTarget.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class Document;
        class Window : public EventTarget
        {
        public:
            DLL Window ();
            DLL Window ( uint32_t aWidth, uint32_t aHeight );
            DLL Window ( const Document* aDocument, uint32_t aWidth, uint32_t aHeight );
            DLL ~Window () override final;
            DLL void ResizeViewport ( uint32_t aWidth, uint32_t aHeight );
            DLL const uint8_t* GetPixels() const;
            DLL size_t GetWidth() const;
            DLL size_t GetHeight() const;
            DLL size_t GetStride() const;
            DLL void Draw();
        private:
            const Document* mDocument{};
            CairoCanvas mCanvas{};
        };
    }
}
#endif
