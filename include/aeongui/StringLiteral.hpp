/*
Copyright (C) 2018,2019,2025,2026 Rodrigo Jose Hernandez Cordoba

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
#include <iostream>

namespace AeonGUI
{
    /** @brief Compile-time string literal wrapper.
     *
     *  Stores a pointer to a string literal and its size, enabling
     *  constexpr comparisons and use as hash-map keys without allocations.
     */
    class StringLiteral
    {
    public:
        /** @brief Construct from a string literal.
         *  @tparam aStringSize Size of the character array including null terminator.
         *  @param aString The string literal.
         */
        template<size_t aStringSize>
        constexpr StringLiteral ( const char ( &aString ) [aStringSize] ) :
            mString{aString}, mStringSize{aStringSize} {}

        /** @brief Default constructor. */
        constexpr StringLiteral() noexcept = default;
        /** @brief Copy constructor. */
        constexpr StringLiteral ( const StringLiteral& ) noexcept = default;
        /** @brief Move constructor. */
        constexpr StringLiteral ( StringLiteral&& aString ) noexcept = default;
        /** @brief Copy assignment operator. */
        constexpr StringLiteral& operator= ( const StringLiteral& ) noexcept = default;
        /** @brief Move assignment operator. */
        constexpr StringLiteral& operator= ( StringLiteral&& ) noexcept = default;

        /** @brief Get the underlying C string.
         *  @return Pointer to the null-terminated string.
         */
        constexpr const char* GetString() const
        {
            return mString;
        }
        /** @brief Get the string size including the null terminator.
         *  @return Size in bytes.
         */
        constexpr size_t GetStringSize() const
        {
            return mStringSize;
        }
        /** @brief Compare with another StringLiteral for equality. */
        constexpr bool operator == ( const StringLiteral& b ) const
        {
            return ( mString == b.mString || std::char_traits<char>::compare ( mString, b.mString, mStringSize ) == 0 );
        }
        /** @brief Compare with a C string for equality. */
#ifdef __GNUG__
        constexpr
#endif
        bool operator == ( const char* b ) const
        {
            return ( mString == b ) || std::char_traits<char>::compare ( mString, b, mStringSize ) == 0;
        }
        /** @brief Compare with a std::string for equality. */
        bool operator == ( const std::string &b ) const
        {
            return b == mString;
        }
        /** @brief Implicit conversion to std::string. */
        operator std::string() const
        {
            return std::string{mString};
        }
    private:
        const char* mString{};
        size_t mStringSize{};
    };

    /** @brief Hash functor for StringLiteral.
     *
     *  Allows StringLiteral to be used as a key in std::unordered_map
     *  and similar containers.
     */
    struct StringLiteralHash
    {
        size_t operator() ( const StringLiteral& Key ) const noexcept
        {
            return std::hash<std::string> {} ( Key );
        }
    };
}

#endif
