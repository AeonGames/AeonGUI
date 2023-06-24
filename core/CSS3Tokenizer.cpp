/*
Copyright (C) 2023 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/CSS3Tokenizer.h"
namespace AeonGUI
{

    constexpr uint32_t flz8 ( uint8_t n )
    {
        constexpr uint32_t DeBruijnHash8[8] =
        {
            0, 1, 6, 2, 7, 5, 4, 3
        };
        n = ~n;
        n |= ( n >>  1 );
        n |= ( n >>  2 );
        n |= ( n >>  4 );
        n ^= ( n >> 1 );
        return DeBruijnHash8[ ( ( n * 0x1D ) & 0xFF ) >> 5];
    }

    uint32_t CSS3Tokenizer::GetCodePointFromUT8 ( const char* bytes, size_t byte_count, size_t* code_point_lenght )
    {
        if ( byte_count == 0 )
        {
            if ( code_point_lenght != nullptr )
            {
                *code_point_lenght = 0;
            }
            return 0;
        }
        const uint8_t* unsigned_bytes = reinterpret_cast<const uint8_t*> ( bytes );
        switch ( flz8 ( *unsigned_bytes ) )
        {
        case 7:
            if ( code_point_lenght != nullptr )
            {
                *code_point_lenght = 1;
            }
            return *unsigned_bytes;
        case 5: /* two byte  UTF8 */
        {
            if ( byte_count < 2 )
            {
                break;
            }
            if ( code_point_lenght != nullptr )
            {
                *code_point_lenght = 2;
            }
            return ( ( unsigned_bytes[0] ^ 0xC0 ) << 6 ) | ( unsigned_bytes[1] ^ 0x80 );
        }
        case 4: /* three byte  UTF8 */
        {
            if ( byte_count < 3 )
            {
                break;
            }
            if ( code_point_lenght != nullptr )
            {
                *code_point_lenght = 3;
            }
            return ( ( unsigned_bytes[0] ^ 0xE0 ) << 12 ) | ( ( unsigned_bytes[1] ^ 0x80 ) << 6 ) | ( unsigned_bytes[2] ^ 0x80 );
        }
        case 3: /* four byte  UTF8 */
        {
            if ( byte_count < 4 )
            {
                break;
            }
            if ( code_point_lenght != nullptr )
            {
                *code_point_lenght = 4;
            }
            return ( ( unsigned_bytes[0] ^ 0xF0 ) << 18 ) | ( ( unsigned_bytes[1] ^ 0x80 ) << 12 ) | ( ( unsigned_bytes[2] ^ 0x80 ) << 6 ) | ( unsigned_bytes[3] ^ 0x80 );
        }
        }
        if ( code_point_lenght != nullptr )
        {
            *code_point_lenght = 0;
        }
        return 0;
    }

    static_assert ( flz8 ( 0x0 ) == 7, "Find Last Zero Bit Failure." );
    static_assert ( flz8 ( 0x80 ) == 6, "Find Last Zero Bit Failure." );
    static_assert ( flz8 ( 0xC0 ) == 5, "Find Last Zero Bit Failure." );
    static_assert ( flz8 ( 0xE0 ) == 4, "Find Last Zero Bit Failure." );
    static_assert ( flz8 ( 0xF0 ) == 3, "Find Last Zero Bit Failure." );
    static_assert ( flz8 ( 0xF8 ) == 2, "Find Last Zero Bit Failure." );
    static_assert ( flz8 ( 0xFC ) == 1, "Find Last Zero Bit Failure." );
    static_assert ( flz8 ( 0xFE ) == 0, "Find Last Zero Bit Failure." );
    static_assert ( flz8 ( 0xe2 ) == 4, "Find Last Zero Bit Failure." );

    CSS3Token::CSS3Token ( CSS3TokenType id ) : mId{id} {}
    CSS3TokenType CSS3Token::GetType() const
    {
        return mId;
    }

    CSS3Token CSS3Tokenizer::Consume ( const std::string& code_points, size_t& pos )
    {
        if ( pos >= code_points.size() )
        {
            return CSS3TokenType::END_OF_FILE;
        }
        size_t code_point_lenght{};
        uint32_t code_point{ GetCodePointFromUT8 ( code_points.data() + pos, code_points.size() - pos, &code_point_lenght ) };

        /* Skip all comments */
        if ( code_point == 0x002f && GetCodePointFromUT8 ( code_points.data() + pos + 1, code_points.size() - ( pos + 1 ) ) == 0x002a )
        {
            pos += 2;
            while ( pos < code_points.size() )
            {
                code_point = GetCodePointFromUT8 ( code_points.data() + pos, code_points.size() - pos, &code_point_lenght );
                if ( code_point == 0x002a && GetCodePointFromUT8 ( code_points.data() + pos + 1, code_points.size() - ( pos + 1 ) ) == 0x002f )
                {
                    pos += 2;
                    code_point = GetCodePointFromUT8 ( code_points.data() + pos, code_points.size() - pos, &code_point_lenght );
                    break;
                }
                pos += code_point_lenght;
            }
        }

        /* Consume Whitespace */
        bool whitespace{false};
        while ( code_point == 0x000a || code_point == 0x0009 || code_point == 0x0020 )
        {
            pos += code_point_lenght;
            code_point = GetCodePointFromUT8 ( code_points.data() + pos, code_points.size() - pos, &code_point_lenght );
            whitespace = true;
        }
        if ( whitespace )
        {
            return CSS3TokenType::WHITESPACE;
        }

        return CSS3TokenType::END_OF_FILE;
    }
}
