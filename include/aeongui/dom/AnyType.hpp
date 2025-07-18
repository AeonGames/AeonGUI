/*
Copyright (C) 2025 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_DOM_ANYTYPE_H
#define AEONGUI_DOM_ANYTYPE_H
#include <cstdint>
#include <variant>
#include "aeongui/Platform.hpp"
#include "DOMString.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        using AnyType = std::variant <
                        DOMString,
                        int8_t, int16_t, int32_t, int64_t,
                        uint8_t, uint16_t, uint32_t, uint64_t,
                        float, double,
                        bool >;
    }
}
#endif
