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
#include "aeongui/dom/MouseEvent.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        MouseEvent::MouseEvent ( const DOMString& type, const MouseEventInit& eventInitDict ) :
            UIEvent ( type, UIEventInit{EventInit{eventInitDict.bubbles, eventInitDict.cancelable, eventInitDict.composed}, eventInitDict.view, eventInitDict.detail} ),
            m_screenX{eventInitDict.screenX},
            m_screenY{eventInitDict.screenY},
            m_clientX{eventInitDict.clientX},
            m_clientY{eventInitDict.clientY},
            m_ctrlKey{eventInitDict.ctrlKey},
            m_shiftKey{eventInitDict.shiftKey},
            m_altKey{eventInitDict.altKey},
            m_metaKey{eventInitDict.metaKey},
            m_button{eventInitDict.button},
            m_buttons{eventInitDict.buttons},
            m_relatedTarget{eventInitDict.relatedTarget},
            m_modifierAltGraph{eventInitDict.modifierAltGraph},
            m_modifierCapsLock{eventInitDict.modifierCapsLock},
            m_modifierFn{eventInitDict.modifierFn},
            m_modifierFnLock{eventInitDict.modifierFnLock},
            m_modifierHyper{eventInitDict.modifierHyper},
            m_modifierNumLock{eventInitDict.modifierNumLock},
            m_modifierScrollLock{eventInitDict.modifierScrollLock},
            m_modifierSuper{eventInitDict.modifierSuper},
            m_modifierSymbol{eventInitDict.modifierSymbol},
            m_modifierSymbolLock{eventInitDict.modifierSymbolLock}
        {
        }

        double MouseEvent::screenX() const
        {
            return m_screenX;
        }
        double MouseEvent::screenY() const
        {
            return m_screenY;
        }
        double MouseEvent::clientX() const
        {
            return m_clientX;
        }
        double MouseEvent::clientY() const
        {
            return m_clientY;
        }
        bool MouseEvent::ctrlKey() const
        {
            return m_ctrlKey;
        }
        bool MouseEvent::shiftKey() const
        {
            return m_shiftKey;
        }
        bool MouseEvent::altKey() const
        {
            return m_altKey;
        }
        bool MouseEvent::metaKey() const
        {
            return m_metaKey;
        }
        short MouseEvent::button() const
        {
            return m_button;
        }
        unsigned short MouseEvent::buttons() const
        {
            return m_buttons;
        }
        EventTarget* MouseEvent::relatedTarget() const
        {
            return m_relatedTarget;
        }

        bool MouseEvent::getModifierState ( const DOMString& keyArg ) const
        {
            if ( keyArg == "Control" )
            {
                return m_ctrlKey;
            }
            if ( keyArg == "Shift" )
            {
                return m_shiftKey;
            }
            if ( keyArg == "Alt" )
            {
                return m_altKey;
            }
            if ( keyArg == "Meta" )
            {
                return m_metaKey;
            }
            if ( keyArg == "AltGraph" )
            {
                return m_modifierAltGraph;
            }
            if ( keyArg == "CapsLock" )
            {
                return m_modifierCapsLock;
            }
            if ( keyArg == "Fn" )
            {
                return m_modifierFn;
            }
            if ( keyArg == "FnLock" )
            {
                return m_modifierFnLock;
            }
            if ( keyArg == "Hyper" )
            {
                return m_modifierHyper;
            }
            if ( keyArg == "NumLock" )
            {
                return m_modifierNumLock;
            }
            if ( keyArg == "ScrollLock" )
            {
                return m_modifierScrollLock;
            }
            if ( keyArg == "Super" )
            {
                return m_modifierSuper;
            }
            if ( keyArg == "Symbol" )
            {
                return m_modifierSymbol;
            }
            if ( keyArg == "SymbolLock" )
            {
                return m_modifierSymbolLock;
            }
            return false;
        }
    }
}
