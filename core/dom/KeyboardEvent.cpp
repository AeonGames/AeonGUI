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
#include "aeongui/dom/KeyboardEvent.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        KeyboardEvent::KeyboardEvent ( const DOMString& type, const KeyboardEventInit& eventInitDict ) :
            UIEvent ( type, UIEventInit{EventInit{eventInitDict.bubbles, eventInitDict.cancelable, eventInitDict.composed}, eventInitDict.view, eventInitDict.detail} ),
            m_key{eventInitDict.key},
            m_code{eventInitDict.code},
            m_location{eventInitDict.location},
            m_ctrlKey{eventInitDict.ctrlKey},
            m_shiftKey{eventInitDict.shiftKey},
            m_altKey{eventInitDict.altKey},
            m_metaKey{eventInitDict.metaKey},
            m_repeat{eventInitDict.repeat},
            m_isComposing{eventInitDict.isComposing},
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

        const DOMString& KeyboardEvent::key() const
        {
            return m_key;
        }
        const DOMString& KeyboardEvent::code() const
        {
            return m_code;
        }
        unsigned long KeyboardEvent::location() const
        {
            return m_location;
        }
        bool KeyboardEvent::ctrlKey() const
        {
            return m_ctrlKey;
        }
        bool KeyboardEvent::shiftKey() const
        {
            return m_shiftKey;
        }
        bool KeyboardEvent::altKey() const
        {
            return m_altKey;
        }
        bool KeyboardEvent::metaKey() const
        {
            return m_metaKey;
        }
        bool KeyboardEvent::repeat() const
        {
            return m_repeat;
        }
        bool KeyboardEvent::isComposing() const
        {
            return m_isComposing;
        }

        bool KeyboardEvent::getModifierState ( const DOMString& keyArg ) const
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
