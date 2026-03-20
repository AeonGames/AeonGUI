/*
Copyright (C) 2019,2020,2025,2026 Rodrigo Jose Hernandez Cordoba

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
    /** @brief Abstract base class for renderable path data.
     *
     *  A Path is constructed from a sequence of DrawType commands and can
     *  be drawn by a Canvas implementation.
     */
    class Path
    {
    public:
        /** @brief Build the path from a vector of draw commands.
         *  @param aCommands The draw command sequence.
         */
        virtual void Construct ( const std::vector<DrawType>& aCommands ) = 0;
        /** @brief Build the path from a raw array of draw commands.
         *  @param aCommands     Pointer to the command array.
         *  @param aCommandCount Number of commands.
         */
        virtual void Construct ( const DrawType* aCommands, size_t aCommandCount ) = 0;
        /** @brief Virtual destructor. */
        DLL virtual ~Path() = 0;
    };
}
#endif
