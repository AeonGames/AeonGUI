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

#include "Button.h"
#include <iostream>

namespace AeonGUI
{
    Button::Button() :
        caption ( NULL ),
        normal ( NULL ),
        highlighted ( NULL ),
        focused ( NULL ),
        pressed ( NULL ),
        state ( DEFAULT ) {}

    Button::~Button()
    {
        if ( caption != NULL )
        {
            delete[] caption;
        }
    }

    void Button::SetNormalImage ( Image* image )
    {
        normal = image;
    }

    void Button::SetFocusedImage ( Image* image )
    {
        focused = image;
    }

    void Button::SetPressedImage ( Image* image )
    {
        pressed = image;
    }

    void Button::SetCaption ( wchar_t* newcaption )
    {
        size_t len = wcslen ( newcaption );
        if ( caption != NULL )
        {
            delete[] caption;
        }
        if ( len == 0 )
        {
            return;
        }
        caption = new wchar_t[len + 1];
        wcscpy ( caption, newcaption );
    }

    Image* Button::GetNormalImage()
    {
        return normal;
    }

    Image* Button::GetFocusedImage()
    {
        return focused;
    }

    Image* Button::GetPressedImage()
    {
        return pressed;
    }

    const wchar_t* Button::GetCaption()
    {
        return caption;
    }

    void Button::OnRender ( Renderer* renderer )
    {
        Widget::OnRender ( renderer );
        Rect screenrect;
        GetScreenRect ( &screenrect );
        switch ( state )
        {
        case DEFAULT:
            if ( normal != NULL )
            {
                renderer->DrawImage ( normal, screenrect.GetX(), screenrect.GetY() );
            }
            break;
        case HIGHLIGHTED:
            if ( highlighted != NULL )
            {
                renderer->DrawImage ( highlighted, screenrect.GetX(), screenrect.GetY() );
            }
            break;
        case FOCUSED:
            if ( focused != NULL )
            {
                renderer->DrawImage ( focused, screenrect.GetX(), screenrect.GetY() );
            }
            break;
        case PRESSED:
            if ( pressed != NULL )
            {
                renderer->DrawImage ( pressed, screenrect.GetX(), screenrect.GetY() );
            }
            break;
        }
        if ( caption != NULL )
        {
            DrawString ( renderer, textcolor, screenrect.GetX(), screenrect.GetY(), caption );
        }
    }

    bool Button::OnMouseButtonDown ( uint8_t button, uint32_t X, uint32_t Y, Widget* widget )
    {
        //std::cout << "On Mouse Button Down " << X << " " << Y << std::endl;
        state = PRESSED;
        CaptureMouse();
        return false;
    }

    bool Button::OnMouseButtonUp ( uint8_t button, uint32_t X, uint32_t Y, Widget* widget )
    {
        //std::cout << "On Mouse Button Up " << X << " " << Y << std::endl;
        state = DEFAULT;
        ReleaseMouse();
        return false;
    }

    bool Button::OnMouseClick ( uint8_t button, uint32_t x, uint32_t y, Widget* widget )
    {
        /* Cascade event to parent */
        return false;
    }
}
