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
#ifndef AEONGUI_WORKSPACE_H
#define AEONGUI_WORKSPACE_H
#include <cstdint>
#include <vector>
#include <memory>
#include "aeongui/Platform.h"
#include "aeongui/Widget.h"

namespace AeonGUI
{
    class Workspace
    {
    public:
        DLL Workspace ( uint32_t aWidth, uint32_t aHeight );
        DLL ~Workspace();
        DLL void Resize ( uint32_t aWidth, uint32_t aHeight );
        DLL void Draw() const;
        DLL const uint8_t* GetData() const;
        DLL size_t GetWidth() const;
        DLL size_t GetHeight() const;
        DLL size_t GetStride() const;
        DLL Widget* AddWidget ( std::unique_ptr<Widget> aWidget );
        DLL std::unique_ptr<Widget> RemoveWidget ( const Widget* aWidget );
    private:
        void* mCairoSurface{};
        void* mCairoContext{};
        std::vector<std::unique_ptr<Widget>> mChildren{};
    };
}
#endif
