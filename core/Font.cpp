/******************************************************************************
Copyright 2010-2012 Rodrigo Hernandez Cordoba

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/
#include "Font.h"
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cwchar>
#include "fontstructs.h"

static int glyphcompar ( const void *key, const void *element )
{
    wchar_t keycode = * ( ( wchar_t* ) key );
    wchar_t elecode = ( wchar_t ) ( ( FNTGlyph* ) element )->charcode;
    if ( keycode < elecode )
    {
        return -1;
    }
    else if ( keycode > elecode )
    {
        return 1;
    }
    return 0;
}

namespace AeonGUI
{
    Font::Font ( void* data, size_t size )
    {
        ///\todo Split for different font version loading.
        // Safe Guard, Glyph and FNTGlyph must be exactly the same
        assert ( sizeof ( Glyph ) == sizeof ( FNTGlyph ) );
        FNTHeader* header = ( FNTHeader* ) data;
        isgood = true;
        if ( std::string ( header->id ) != "AEONFNT" )
        {
            isgood = false;
            return;
        }
        glyphcount = header->glyphcount;
        map_width = header->map_width;
        map_height = header->map_height;
        nominal_width = header->nominal_width;
        nominal_height = header->nominal_height;
        ascender = header->ascender;
        descender = header->descender;
        height = header->height;
        max_advance = header->max_advance;

        glyphdata = new Glyph[glyphcount];
        memcpy ( glyphdata, ( ( ( unsigned char* ) data ) + sizeof ( FNTHeader ) ), sizeof ( Glyph ) *glyphcount );
        glyphmap = ( uint8_t* ) new uint8_t[map_width * map_height];
        memcpy ( glyphmap, ( ( ( unsigned char* ) data ) + sizeof ( FNTHeader ) + sizeof ( Glyph ) *glyphcount ), sizeof ( unsigned char ) *map_width * map_height );
#if 1
        std::cout << "ID: " << header->id << std::endl;
        std::cout << "Glyph Count: " << glyphcount << std::endl;
        std::cout << "Map Width: " << map_width << std::endl;
        std::cout << "Map Height: " << map_height << std::endl;
        std::cout << "Nominal Width: " << nominal_width << std::endl;
        std::cout << "Nominal Height: " << nominal_height << std::endl;
        std::cout << "Ascender: " << ascender << std::endl;
        std::cout << "Descender: " << descender << std::endl;
        std::cout << "Height: " << height << std::endl;
#endif
#if 0
        glyphs = ( Glyph* ) glyphdata;
        for ( uint32_t i = 0; i < glyphcount; ++i )
        {
            std::wcout << "Charcode: " << glyphs[i].charcode << std::endl;
            std::cout << "Minimum X: " << glyphs[i].min[0] << std::endl;
            std::cout << "Minimum Y: " << glyphs[i].min[1] << std::endl;
            std::cout << "Maximum X: " << glyphs[i].max[0] << std::endl;
            std::cout << "Maximum Y: " << glyphs[i].max[1] << std::endl;
            std::cout << "Normalized Minimum X: " << glyphs[i].normalized_min[0] << std::endl;
            std::cout << "Normalized Minimum Y: " << glyphs[i].normalized_min[1] << std::endl;
            std::cout << "Normalized Maximum X: " << glyphs[i].normalized_max[0] << std::endl;
            std::cout << "Normalized Maximum Y: " << glyphs[i].normalized_max[1] << std::endl;
            std::cout << "Top: " << glyphs[i].top << std::endl;
            std::cout << "Left: " << glyphs[i].left << std::endl;
            std::cout << "Horizontal Advance: " << glyphs[i].advance[0] << std::endl;
            std::cout << "Vertical Advance: " << glyphs[i].advance[1] << std::endl;
        }
#endif
    }
    Font::Font ( const char* filename ) :
        isgood ( false ),
        glyphcount ( 0 ),
        map_width ( 0 ),
        map_height ( 0 ),
        nominal_width ( 0 ),
        nominal_height ( 0 ),
        ascender ( 0 ),
        descender ( 0 ),
        height ( 0 ),
        max_advance ( 0 ),
        glyphdata ( NULL ),
        glyphmap ( NULL )
    {
        unsigned char* buffer;
        size_t length;
        std::ifstream file;
        file.open ( filename, std::fstream::in | std::fstream::binary );
        if ( !file.is_open() )
        {
            // need to catch this error
            isgood = false;
            return;
        }
        file.seekg ( 0, std::ios::end );
        length = static_cast<size_t> ( file.tellg() );
        buffer = new unsigned char[length];
        file.seekg ( 0, std::ios::beg );
        file.read ( ( char* ) buffer, length );
        file.close();
        // Data  constructor sets isgood
        //isgood = true;
        Font ( ( void* ) buffer, length );
        delete[] buffer;
    }
    Font::Glyph* Font::GetGlyph ( wchar_t charcode )
    {
        return ( Glyph* ) bsearch ( ( void* ) ( &charcode ), glyphdata, glyphcount, sizeof ( Glyph ), glyphcompar );
    }
}
