/*
Copyright (C) 2026 Rodrigo Jose Hernandez Cordoba

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "aeongui/dom/WheelEvent.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        WheelEvent::WheelEvent ( const DOMString& type, const WheelEventInit& eventInitDict ) :
            MouseEvent ( type, MouseEventInit
                         {
                             EventModifierInit{
                                 UIEventInit{EventInit{eventInitDict.bubbles, eventInitDict.cancelable, eventInitDict.composed}, eventInitDict.view, eventInitDict.detail},
                                 eventInitDict.ctrlKey, eventInitDict.shiftKey, eventInitDict.altKey, eventInitDict.metaKey,
                                 eventInitDict.modifierAltGraph, eventInitDict.modifierCapsLock, eventInitDict.modifierFn, eventInitDict.modifierFnLock,
                                 eventInitDict.modifierHyper, eventInitDict.modifierNumLock, eventInitDict.modifierScrollLock, eventInitDict.modifierSuper,
                                 eventInitDict.modifierSymbol, eventInitDict.modifierSymbolLock
                             },
                             eventInitDict.screenX, eventInitDict.screenY, eventInitDict.clientX, eventInitDict.clientY,
                             eventInitDict.button, eventInitDict.buttons, eventInitDict.relatedTarget
                         } ),
            m_deltaX{eventInitDict.deltaX},
            m_deltaY{eventInitDict.deltaY},
            m_deltaZ{eventInitDict.deltaZ},
            m_deltaMode{eventInitDict.deltaMode}
        {
        }

        double WheelEvent::deltaX() const
        {
            return m_deltaX;
        }
        double WheelEvent::deltaY() const
        {
            return m_deltaY;
        }
        double WheelEvent::deltaZ() const
        {
            return m_deltaZ;
        }
        unsigned long WheelEvent::deltaMode() const
        {
            return m_deltaMode;
        }
    }
}
