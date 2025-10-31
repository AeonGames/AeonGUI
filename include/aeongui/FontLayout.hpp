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
#ifndef AEONGUI_FONTLAYOUT_H
#define AEONGUI_FONTLAYOUT_H
#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>
#include "aeongui/Platform.hpp"
#include "aeongui/Color.hpp"
#include "aeongui/Attribute.hpp"
#include "aeongui/Matrix2x3.hpp"
namespace AeonGUI
{
    /** Font layout interface
     * This abstracts PangoLayout the same way Canvas abstracts Cairo surface in order to allow for multiple backends to be used interchangeably in the future.
    */
    class FontLayout
    {
    public:
        DLL virtual ~FontLayout() = 0;
    };
}
#endif
