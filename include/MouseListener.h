#ifndef AEONGUI_MOUSE_LISTENER_H
#define AEONGUI_MOUSE_LISTENER_H
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

namespace AeonGUI
{
    class Widget;
    /// Mouse Listener interface.
    class MouseListener
    {
    public:
        virtual ~MouseListener() {};
        virtual void OnMouseMove ( Widget* widget, uint32_t X, uint32_t Y ) = 0;
        virtual void OnMouseButtonDown ( Widget* widget, uint8_t button, uint32_t X, uint32_t Y ) = 0;
        virtual void OnMouseButtonUp ( Widget* widget, uint8_t button, uint32_t X, uint32_t Y ) = 0;
        virtual void OnMouseClick ( Widget* widget, uint8_t button, uint32_t X, uint32_t Y ) = 0;
    };
}
#endif
