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
    void Button::OnRender ( Renderer* renderer )
    {
        Widget::OnRender ( renderer );
        switch ( state )
        {
        case DEFAULT:
            if ( normal != NULL )
            {
                DrawImage ( renderer, backgroundcolor, 0, 0, normal );
            }
            break;
        case HIGHLIGHTED:
            if ( highlighted != NULL )
            {
                DrawImage ( renderer, backgroundcolor, 0, 0, highlighted );
            }
            break;
        case FOCUSED:
            if ( focused != NULL )
            {
                DrawImage ( renderer, backgroundcolor, 0, 0, focused );
            }
            break;
        case PRESSED:
            if ( pressed != NULL )
            {
                DrawImage ( renderer, backgroundcolor, 0, 0, pressed );
            }
            break;
        }
        if ( !caption.empty() )
        {
            DrawString ( renderer, textcolor, 0, 0, caption.c_str() );
        }
    }

    void Button::OnMouseButtonDown ( uint8_t button, uint16_t X, uint16_t Y )
    {
        //std::cout << "On Mouse Button Down " << X << " " << Y << std::endl;
        state = PRESSED;
        CaptureMouse();
    }

    void Button::OnMouseButtonUp ( uint8_t button, uint16_t X, uint16_t Y )
    {
        //std::cout << "On Mouse Button Up " << X << " " << Y << std::endl;
        state = DEFAULT;
        ReleaseMouse();
    }
}
