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
#include "pcx.h"
#include <iostream>
#include <fstream>

Pcx::Pcx() :
    pixels ( NULL ),
    pixels_size ( 0 )
{
    memset ( &header, 0, sizeof ( Header ) );
}

#if 0
bool Pcx::IsPatch9()
{
    return ( ( header.XStartStretch != 0 ) || ( header.XEndStretch != 0 ) || ( header.XStartPad != 0 ) || ( header.XEndPad != 0 ) ||
             ( header.YStartStretch != 0 ) || ( header.YEndStretch != 0 ) || ( header.YStartPad != 0 ) || ( header.YEndPad != 0 ) );
}
#endif

Pcx::~Pcx()
{
    Unload();
}

bool Pcx::Encode ( uint32_t width, uint32_t height, void* buffer, uint32_t buffer_size )
{
    header.Identifier = 0x0A;
    header.Version = 5;
    header.Encoding = 1;
    header.BitsPerPixel = 8;
    header.XStart = 0;
    header.YStart = 0;
    header.XEnd = width - 1;
    header.YEnd = height - 1;
    header.HorzRes = 300;
    header.VertRes = 300;
    header.NumBitPlanes = 3;
    header.BytesPerLine = width;
    header.PaletteType = 1;
    pixels_size = PadPixels ( width, height, buffer, buffer_size );
    pixels = new uint8_t[pixels_size];
    PadPixels ( width, height, buffer, buffer_size );
    return true;
}

uint32_t Pcx::PadPixels ( uint32_t width, uint32_t height, void* buffer, uint32_t buffer_size )
{
    // This function is untested
    uint32_t datasize = 0;
    uint8_t counter = 0;
    uint8_t* scanline = ( uint8_t* ) buffer;
    uint8_t* encoded_pixel = ( uint8_t* ) pixels;
    uint8_t* encoded_scanline = new uint8_t[width * 2]; // worst case scenario all bytes are different
    uint32_t scanline_count = 0;
    uint8_t byte;

    for ( uint32_t y = 0; y < height; ++y )
    {
        byte = scanline[0];
        counter = 1;
        scanline_count = 0;
        for ( uint32_t x = 1; x < width; ++x )
        {
            if ( ( byte == scanline[x] ) && ( counter < 63 ) )
            {
                counter++;
            }
            else
            {
                if ( encoded_pixel != NULL )
                {
                    encoded_scanline[scanline_count++] = counter | 0xC0;
                    encoded_scanline[scanline_count++] = byte;
                }
                datasize += 2;
                byte = scanline[x];
                counter = 1;
            }
        }
        // we're done a scanline, write the remnant and advance to the next.
        if ( encoded_pixel != NULL )
        {
            encoded_scanline[scanline_count++] = counter | 0xC0;
            encoded_scanline[scanline_count++] = byte;
            memcpy ( encoded_pixel, encoded_scanline, scanline_count );
            encoded_pixel += scanline_count;
            memcpy ( encoded_pixel, encoded_scanline, scanline_count );
            encoded_pixel += scanline_count;
            memcpy ( encoded_pixel, encoded_scanline, scanline_count );
            encoded_pixel += scanline_count;
        }
        datasize += 2;
        scanline += width;
    }
    delete[] encoded_scanline;
    return datasize * 3;
}

bool Pcx::Save ( const char* filename )
{
    std::ofstream pcx;
    pcx.open ( filename, std::ios_base::out | std::ios_base::binary );
    if ( !pcx.is_open() )
    {
        std::cerr << "Problem opening " << filename << " for writting." << std::endl;
        return false;
    }
    pcx.write ( ( const char* ) &header, sizeof ( Header ) );
    pcx.write ( ( const char* ) pixels, sizeof ( char ) *pixels_size );
    pcx.close();
    return true;
}

uint32_t Pcx::GetWidth()
{
    return header.XEnd - header.XStart + 1;
}

uint32_t Pcx::GetHeight()
{
    return header.YEnd - header.YStart + 1;
}

const uint8_t* Pcx::GetPixels()
{
    return pixels;
}

uint8_t Pcx::GetNumBitPlanes()
{
    return header.NumBitPlanes;
}

uint16_t Pcx::GetStretchX()
{
    return header.StretchX;
}

uint16_t Pcx::GetStretchWidth()
{
    return header.StretchWidth;
}
uint16_t Pcx::GetPadX()
{
    return header.PadX;
}
uint16_t Pcx::GetPadWidth()
{
    return header.PadWidth;
}

uint16_t Pcx::GetStretchY()
{
    return header.StretchY;
}
uint16_t Pcx::GetStretchHeight()
{
    return header.StretchHeight;
}
uint16_t Pcx::GetPadY()
{
    return header.PadY;
}
uint16_t Pcx::GetPadHeight()
{
    return header.PadHeight;
}

bool Pcx::Decode ( uint32_t buffer_size, void* buffer )
{
    memcpy ( &header, buffer, sizeof ( Header ) );
    if ( ( header.Version != 5 ) && ( header.Encoding != 1 ) && ( header.BitsPerPixel != 8 ) )
    {
        // Support only file format version 5 for now.
        Unload();
        return false;
    }

    uint32_t scanline_length = header.NumBitPlanes * header.BytesPerLine;
    uint8_t* byte = reinterpret_cast<uint8_t*> ( buffer ) + sizeof ( Header );
    pixels = new uint8_t[ scanline_length * GetHeight() ];

    for ( uint32_t i = 0; i < GetHeight(); ++i )
    {
        for ( uint8_t j = 0; j < header.NumBitPlanes; ++j )
        {
            uint8_t* pixel = pixels + ( scanline_length * i ) + j;
            for ( uint16_t k = 0; k < header.BytesPerLine; )
            {
                if ( ( byte[0] & 0xC0 ) == 0xC0 )
                {
                    uint8_t count = byte[0] & 0x3F;
                    for ( uint8_t l = 0; l < count; ++l )
                    {
                        *pixel = byte[1];
                        pixel += header.NumBitPlanes;
                    }
                    byte += 2;
                    k += count;
                }
                else
                {
                    *pixel = byte[0];
                    pixel += header.NumBitPlanes;
                    ++byte;
                    ++k;
                }
            }
        }
    }
    return true;
}

bool Pcx::Load ( const char* filename )
{
    uint8_t* buffer = NULL;
    uint32_t buffer_size = 0;
    bool retval;
    std::ifstream pcx;
    pcx.open ( filename, std::ios_base::in | std::ios_base::binary );
    if ( !pcx.is_open() )
    {
        std::cerr << "Problem opening " << filename << " for reading." << std::endl;
        return false;
    }

    pcx.seekg ( 0, std::ios_base::end );
    buffer_size = static_cast<uint32_t> ( pcx.tellg() );
    pcx.seekg ( 0, std::ios_base::beg );
    buffer = new uint8_t[buffer_size];
    pcx.read ( reinterpret_cast<char*> ( buffer ), buffer_size );
    pcx.close();

    retval = Decode ( buffer_size, buffer );
    delete[] buffer;
    return retval;
}

void Pcx::Unload ( )
{
    if ( pixels != NULL )
    {
        delete[] pixels;
        pixels = NULL;
        pixels_size = 0;
        memset ( &header, 0, sizeof ( Header ) );
    }
}
