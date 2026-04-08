/*
Copyright (C) 2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_SKIAPATH_H
#define AEONGUI_SKIAPATH_H
#include "aeongui/Path.hpp"
#include <core/SkPath.h>

namespace AeonGUI
{
    /** @brief Skia-based Path implementation.
     *
     *  Wraps an @c SkPath and is constructed from AeonGUI draw commands.
     */
    class SkiaPath : public Path
    {
    public:
        SkiaPath();
        /** @brief Build the Skia path from a vector of draw commands.
         *  @param aCommands The draw command sequence.
         *  @param aPathDataHint Pre-computed size hint (unused by Skia backend).
         */
        void Construct ( const std::vector<DrawType>& aCommands, size_t aPathDataHint = 0 ) final;
        /** @brief Build the Skia path from a raw array of draw commands.
         *  @param aCommands     Pointer to the command array.
         *  @param aCommandCount Number of commands.
         *  @param aPathDataHint Pre-computed size hint (unused by Skia backend).
         */
        void Construct ( const DrawType* aCommands, size_t aCommandCount, size_t aPathDataHint = 0 ) final;
        ~SkiaPath();
        /** @brief Access the underlying Skia path object.
         *  @return Const reference to the internal @c SkPath.
         */
        const SkPath& GetSkPath() const;
        AEONGUI_DLL double GetTotalLength() const final;
        PathPoint GetPointAtLength ( double aDistance ) const final;
        bool IsClosed() const final;
    private:
        SkPath mPath;
    };
}
#endif
