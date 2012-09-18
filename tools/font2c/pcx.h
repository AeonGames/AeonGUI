#ifndef PCX_H
#define PCX_H
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
#include <cstring>
class Pcx
{
public:
    Pcx()
    {
        pixels = NULL;
        memset ( &header, 0, sizeof ( Header ) );
        pixels_size = 0;
    }
    ~Pcx()
    {
        if ( pixels != NULL )
        {
            delete[] ( unsigned char* ) pixels;
        }
    }
    bool Encode ( unsigned int width, unsigned int height, void* buffer, unsigned int buffer_size );
    bool Save ( const char* filename );
private:
    unsigned int FillPixels ( unsigned int width, unsigned int height, void* buffer, unsigned int buffer_size );
    struct Header
    {
        unsigned char    Identifier;        // PCX Id Number (Always 0x0A)
        unsigned char    Version;           // Version Number
        unsigned char    Encoding;          // Encoding Format
        unsigned char    BitsPerPixel;      // Bits per Pixel
        unsigned short XStart;            // Left of image
        unsigned short YStart;            // Top of Image
        unsigned short XEnd;              // Right of Image
        unsigned short YEnd;              // Bottom of image
        unsigned short HorzRes;           // Horizontal Resolution
        unsigned short VertRes;           // Vertical Resolution
        unsigned char    Palette[48];       // 16-Color EGA Palette
        unsigned char    Reserved1;         // Reserved (Always 0)
        unsigned char    NumBitPlanes;      // Number of Bit Planes
        unsigned short BytesPerLine;      // Bytes per Scan-line
        unsigned short PaletteType;       // Palette Type
        unsigned short HorzScreenSize;    // Horizontal Screen Size
        unsigned short VertScreenSize;    // Vertical Screen Size
        unsigned char    Reserved2[54];     // Reserved (Always 0)
    };
    Header header;
    void* pixels;
    unsigned int pixels_size;
};
#endif
