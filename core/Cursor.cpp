/******************************************************************************
Copyright 2013 Rodrigo Hernandez Cordoba

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

#include "Cursor.h"

namespace AeonGUI
{
    Cursor::Cursor() :
        cursor_image ( NULL ),
        x ( 0 ),
        y ( 0 ),
        xoffset ( 0 ),
        yoffset ( 0 )
    {
    }
    Cursor::~Cursor()
    {
    }
    void Cursor::SetCursorImage ( Image* image )
    {
        cursor_image = image;
    }
    Image* Cursor::GetCursorImage()
    {
        return cursor_image;
    }

    void Cursor::SetOffsets ( int32_t xoff, int32_t yoff )
    {
        xoffset = xoff;
        yoffset = yoff;
    }
    void Cursor::SetPosition ( int32_t xpos, int32_t ypos )
    {
        x = xpos;
        y = ypos;
    }

    void Cursor::Render ( Renderer* renderer )
    {
        if ( cursor_image != NULL )
        {
            renderer->DrawImage ( cursor_image, x - xoffset, y - yoffset );
        }
    }
}
