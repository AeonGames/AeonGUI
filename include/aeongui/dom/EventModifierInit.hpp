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
#ifndef AEONGUI_DOM_EVENTMODIFIERINIT_H
#define AEONGUI_DOM_EVENTMODIFIERINIT_H
#include "aeongui/dom/UIEvent.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Shared modifier key initialization dictionary.
         *
         *  Used by both KeyboardEvent and MouseEvent to initialize
         *  keyboard modifier attributes and extended modifier states.
         *  @see https://www.w3.org/TR/uievents/#event-modifier-initializers
         */
        struct EventModifierInit : UIEventInit
        {
            bool ctrlKey{false};           ///< Control key modifier active.
            bool shiftKey{false};          ///< Shift key modifier active.
            bool altKey{false};            ///< Alt (Option) key modifier active.
            bool metaKey{false};           ///< Meta (Command/Windows) key modifier active.
            bool modifierAltGraph{false};  ///< AltGraph modifier active.
            bool modifierCapsLock{false};  ///< CapsLock modifier active.
            bool modifierFn{false};        ///< Fn modifier active.
            bool modifierFnLock{false};    ///< FnLock modifier active.
            bool modifierHyper{false};     ///< Hyper modifier active.
            bool modifierNumLock{false};   ///< NumLock modifier active.
            bool modifierScrollLock{false};///< ScrollLock modifier active.
            bool modifierSuper{false};     ///< Super modifier active.
            bool modifierSymbol{false};    ///< Symbol modifier active.
            bool modifierSymbolLock{false};///< SymbolLock modifier active.
        };
    }
}
#endif
