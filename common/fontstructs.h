#ifndef FONTSTRUCTS_H
#define FONTSTRUCTS_H
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
#include "Integer.h"
struct FNTHeader
{
    char id[8];
    uint16_t version[2];
    uint32_t glyphcount;
    uint32_t map_width;
    uint32_t map_height;
    uint16_t nominal_width;
    uint16_t nominal_height;
    int16_t ascender;
    int16_t descender;
    uint16_t height;
    int16_t max_advance;
};

struct FNTGlyph
{
    uint32_t charcode;
    uint16_t width;
    uint16_t height;
    uint16_t min[2];
    uint16_t max[2];
    // save normalized uvs
    // so the calculation and storage
    // is not needed in the renderer
    float normalized_min[2];
    float normalized_max[2];
    int16_t top;
    int16_t left;
    int16_t advance[2];
};
#endif
