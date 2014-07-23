#ifndef AEONGUI_CURSOR_H
#define AEONGUI_CURSOR_H
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
#include "Platform.h"
#include "Integer.h"
#include "Image.h"
#include "Renderer.h"
namespace AeonGUI
{
    class Cursor
    {
    public:
        Cursor();
        virtual ~Cursor();
        void SetCursorImage ( Image* image );
        void SetOffsets ( int32_t xoff, int32_t yoff );
        void SetPosition ( int32_t xpos, int32_t ypos );
        Image* GetCursorImage();
    protected:
        friend class Renderer;
        void Render ( Renderer* renderer );
    private:
        Image* cursor_image;
        int32_t x;
        int32_t y;
        int32_t xoffset;
        int32_t yoffset;
    };
}
#endif