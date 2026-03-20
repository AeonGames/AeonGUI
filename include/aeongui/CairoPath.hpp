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
#ifndef AEONGUI_CAIROPATH_H
#define AEONGUI_CAIROPATH_H
#include <cairo.h>
#include "aeongui/CairoCanvas.hpp"
#include "aeongui/Path.hpp"

namespace AeonGUI
{
    /** @brief Cairo-backend path implementation.
     *
     *  Converts DrawType command sequences into a cached cairo_path_t
     *  structure for efficient repeated rendering.
     */
    class CairoPath : public Path
    {
    public:
        /** @brief Default constructor. */
        CairoPath();
        /** @brief Build the Cairo path from a vector of draw commands.
         *  @param aCommands The draw command sequence.
         */
        void Construct ( const std::vector<DrawType>& aCommands ) final;
        /** @brief Build the Cairo path from a raw array of draw commands.
         *  @param aCommands    Pointer to the command array.
         *  @param aCommandCount Number of commands.
         */
        void Construct ( const DrawType* aCommands, size_t aCommandCount ) final;
        /** @brief Destructor. Frees path data. */
        ~CairoPath();
        /** @brief Get the underlying Cairo path structure.
         *  @return Pointer to the cairo_path_t.
         */
        const cairo_path_t* GetCairoPath() const;
    private:
        cairo_path_t mPath{};
        std::vector<cairo_path_data_t> mPathData;
    };
}
#endif
