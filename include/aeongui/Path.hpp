/*
Copyright (C) 2019,2020,2025 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_PATH_H
#define AEONGUI_PATH_H
#include <cstdint>
#include <cstddef>
#include <vector>
#include "aeongui/Platform.hpp"
#include "aeongui/DrawType.hpp"

namespace AeonGUI
{
    /** Base class for cached path data. */
    class Path
    {
    public:
        virtual void Construct ( const std::vector<DrawType>& aCommands ) = 0;
        virtual void Construct ( const DrawType* aCommands, size_t aCommandCount ) = 0;
        DLL virtual ~Path() = 0;
    };
}
#endif
