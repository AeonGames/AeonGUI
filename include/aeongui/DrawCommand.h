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
#ifndef AEONGUI_DRAWCOMMAND_H
#define AEONGUI_DRAWCOMMAND_H

#include <cstdint>
#include <cstddef>
#include <vector>
#include <initializer_list>
#include "aeongui/Platform.h"
#include "aeongui/Vector2.h"

namespace AeonGUI
{
    class DrawCommand
    {
    public:
        DLL DrawCommand ( uint64_t aCommand, const Vector2& aVertex );
        DLL uint64_t GetCommand() const;
        DLL const Vector2& GetVertex() const;
    private:
        uint64_t mCommand;
        Vector2  mVertex;
    };
}
#endif
