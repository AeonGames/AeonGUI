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
#ifndef AEONGUI_DOM_DOMSTRING_H
#define AEONGUI_DOM_DOMSTRING_H
#include <string>
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** Represents a string in the Document Object Model (DOM).
         * The DOM specifies that the DOMString type is a sequence of UTF-16 code units,
         * but in this implementation, it is represented as a UTF-8 string which is more efficient and easier to work with.
         * If there is a need to handle UTF-16 specifically, it can be converted as needed.
         */
        using DOMString = std::string;
    }
}
#endif
