#ifndef AEONGUI_BUTTON_H
#define AEONGUI_BUTTON_H
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
#include "Widget.h"
#include "Image.h"
#include <string>

namespace AeonGUI
{
    /*! \brief Basic push button. */
    class Button : public Widget
    {
    public:
        enum BUTTONSTATE
        {
            DEFAULT = 0,
            HIGHLIGHTED,
            FOCUSED,
            PRESSED
        };
        Button() :
            normal ( NULL ),
            highlighted ( NULL ),
            focused ( NULL ),
            pressed ( NULL ),
            state ( DEFAULT )
        {}
        virtual ~Button()
        {}
        inline void SetNormalImage ( Image* image )
        {
            normal = image;
        }
        inline void SetFocusedImage ( Image* image )
        {
            focused = image;
        }
        inline void SetPressedImage ( Image* image )
        {
            pressed = image;
        }
        inline void SetCaption ( std::wstring& newcaption )
        {
            caption = newcaption;
        }
        inline Image* GetNormalImage()
        {
            return normal;
        }
        inline Image* GetFocusedImage()
        {
            return focused;
        }
        inline Image* GetPressedImage()
        {
            return pressed;
        }
        inline std::wstring& GetCaption()
        {
            return caption;
        }
        virtual void OnMouseButtonDown ( uint8_t button, uint16_t X, uint16_t Y );
        virtual void OnMouseButtonUp ( uint8_t button, uint16_t X, uint16_t Y );
    protected:
        virtual void OnRender ( Renderer* renderer );
        std::wstring caption;
        Image* normal;
        Image* highlighted;
        Image* focused;
        Image* pressed;
        BUTTONSTATE state;
    };
}
#endif
