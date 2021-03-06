#ifndef PCX_H
#define PCX_H
/******************************************************************************
Copyright (C) 2010-2012,2019 Rodrigo Hernandez Cordoba

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
#include <string.h>
#include <stdint.h>
class Pcx
{
public:
    Pcx();
    ~Pcx();
    bool Encode ( uint32_t width, uint32_t height, void* buffer, uint32_t buffer_size );
    bool Save ( const char* filename );
    bool Decode ( uint32_t buffer_size, void* buffer );
    bool Load ( const char* filename );
    void Unload ( );
    uint32_t GetWidth();
    uint32_t GetHeight();
    const uint8_t* GetPixels();
    uint8_t GetNumBitPlanes();
    uint16_t GetXStretchStart();
    uint16_t GetXStretchEnd();
    uint16_t GetStretchWidth();
    uint16_t GetXPadStart();
    uint16_t GetXPadEnd();
    uint16_t GetPadWidth();
    uint16_t GetYStretchStart();
    uint16_t GetYStretchEnd();
    uint16_t GetStretchHeight();
    uint16_t GetYPadStart();
    uint16_t GetYPadEnd();
    uint16_t GetPadHeight();
private:
    uint32_t PadPixels ( uint32_t width, uint32_t height, void* buffer, uint32_t buffer_size );
    struct Header
    {
        uint8_t  Identifier;        // PCX Id Number (Always 0x0A)
        uint8_t  Version;           // Version Number
        uint8_t  Encoding;          // Encoding Format
        uint8_t  BitsPerPixel;      // Bits per Pixel
        uint16_t XStart;            // Left of image
        uint16_t YStart;            // Top of Image
        uint16_t XEnd;              // Right of Image
        uint16_t YEnd;              // Bottom of image
        uint16_t HorzRes;           // Horizontal Resolution
        uint16_t VertRes;           // Vertical Resolution
        uint8_t  Palette[48];       // 16-Color EGA Palette
        uint8_t  Reserved1;         // Reserved (Always 0)
        uint8_t  NumBitPlanes;      // Number of Bit Planes
        uint16_t BytesPerLine;      // Bytes per Scan-line
        uint16_t PaletteType;       // Palette Type
        uint16_t HorzScreenSize;    // Horizontal Screen Size
        uint16_t VertScreenSize;    // Vertical Screen Size
        uint16_t XStretchStart;      // Patch 9 start stretch coordinate (Unofficial feature)
        uint16_t XStretchEnd;        // Patch 9 end stretch coordinate (Unofficial feature)
        uint16_t XPadStart;         // Patch 9 start Pad coordinate (Unofficial feature)
        uint16_t XPadEnd;           // Patch 9 end Pad coordinate (Unofficial feature)
        uint16_t YStretchStart;      // Patch 9 start stretch coordinate (Unofficial feature)
        uint16_t YStretchEnd;        // Patch 9 end stretch coordinate (Unofficial feature)
        uint16_t YPadStart;         // Patch 9 start Pad coordinate (Unofficial feature)
        uint16_t YPadEnd;           // Patch 9 end Pad coordinate (Unofficial feature)
        uint8_t  Reserved2[38];     // Reserved (Always 0, should be 54, but 16 bytes are taken by the unofficial patch 9 support)
    };
    Header header;
    uint8_t* pixels;
    uint32_t pixels_size;
};
#endif
