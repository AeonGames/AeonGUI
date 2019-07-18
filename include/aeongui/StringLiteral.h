/*
Copyright (C) 2018,2019 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_STRINGLITERAL_H
#define AEONGUI_STRINGLITERAL_H
#include <cstdint>
#include <string>
#include <cstring>
#include <functional>

namespace AeonGUI
{
    class StringLiteral
    {
    public:
        template<size_t aStringSize>
        constexpr StringLiteral ( const char ( &aString ) [aStringSize] ) :
            mString{aString}, mStringSize{aStringSize} {}

        constexpr StringLiteral() noexcept = default;
        constexpr StringLiteral ( const StringLiteral& ) noexcept = default;
        constexpr StringLiteral ( StringLiteral&& aString ) noexcept = default;
        constexpr StringLiteral& operator= ( const StringLiteral& ) noexcept = default;
        constexpr StringLiteral& operator= ( StringLiteral&& ) noexcept = default;

        constexpr const char* GetString() const
        {
            return mString;
        }
        constexpr size_t GetStringSize() const
        {
            return mStringSize;
        }
        constexpr bool operator == ( const StringLiteral& b ) const
        {
            return ( mString == b.mString ) || ( strcmp ( mString, b.mString ) == 0 );
        }
#ifdef __GNUG__
        constexpr
#endif
        bool operator == ( const char* b ) const
        {
            return ( mString == b ) || ( strcmp ( mString, b ) == 0 );
        }
        bool operator == ( const std::string &b ) const
        {
            return b == mString;
        }
        operator std::string() const
        {
            return std::string{mString};
        }
    private:
        const char* mString{};
        size_t mStringSize{};
    };

    struct StringLiteralHash
    {
        size_t operator() ( const StringLiteral& Key ) const noexcept
        {
            return std::hash<std::string> {} ( Key );
        }
    };
}

#endif
