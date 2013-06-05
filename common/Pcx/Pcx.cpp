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

bool Pcx::Encode ( unsigned int width, unsigned int height, void* buffer, unsigned int buffer_size )
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
    pixels_size = FillPixels ( width, height, buffer, buffer_size );
    pixels = new unsigned char[pixels_size];
    FillPixels ( width, height, buffer, buffer_size );
    return true;
}

unsigned int Pcx::FillPixels ( unsigned int width, unsigned int height, void* buffer, unsigned int buffer_size )
{
    // This function is untested
    unsigned int datasize = 0;
    unsigned char counter = 0;
    unsigned char* scanline = ( unsigned char* ) buffer;
    unsigned char* encoded_pixel = ( unsigned char* ) pixels;
    unsigned char* encoded_scanline = new unsigned char[width * 2]; // worst case scenario all bytes are different
    unsigned int scanline_count = 0;
    unsigned char byte;

    for ( unsigned int y = 0; y < height; ++y )
    {
        byte = scanline[0];
        counter = 1;
        scanline_count = 0;
        for ( unsigned int x = 1; x < width; ++x )
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

bool Pcx::Decode ( void* buffer, unsigned int buffer_size )
{
    return false;
}

bool Pcx::Load ( const char* filename )
{
    unsigned char* buffer = NULL;
    unsigned int buffer_size = 0;
    bool retval;
    std::ifstream pcx;
    pcx.open ( filename, std::ios_base::in | std::ios_base::binary );
    if ( !pcx.is_open() )
    {
        std::cerr << "Problem opening " << filename << " for reading." << std::endl;
        return false;
    }

    pcx.seekg ( 0, std::ios_base::end );
    buffer_size = pcx.tellg();
    pcx.seekg ( 0, std::ios_base::beg );
    buffer = new unsigned char[buffer_size];
    pcx.read ( reinterpret_cast<char*> ( buffer ), buffer_size );
    pcx.close();

    retval = Decode ( buffer, buffer_size );
    delete[] buffer;
    return retval;
}
