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
#ifndef AEONGUI_ASCIICASE_H
#define AEONGUI_ASCIICASE_H
#include <algorithm>
#include <cctype>
#include <string_view>

namespace AeonGUI
{
    /** @brief Compare @p aValue against an already-lowercase ASCII keyword.
     *
     *  HTML keyword attributes (`type`, `step="any"`, ...) are matched
     *  ASCII case-insensitively.  Folding in place avoids the copy a
     *  lowercased temporary would cost on every parse.
     *  @param aValue   Candidate text, in any case.
     *  @param aKeyword Keyword to match, which must already be lowercase.
     */
    inline bool EqualsIgnoreCase ( std::string_view aValue, std::string_view aKeyword )
    {
        return aValue.size() == aKeyword.size() &&
               std::equal ( aValue.begin(), aValue.end(), aKeyword.begin(),
                            [] ( char aLeft, char aRight )
        {
            return std::tolower ( static_cast<unsigned char> ( aLeft ) ) == aRight;
        } );
    }
}
#endif
