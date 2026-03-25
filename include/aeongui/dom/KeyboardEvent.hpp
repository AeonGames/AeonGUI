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
#ifndef AEONGUI_DOM_KEYBOARDEVENT_H
#define AEONGUI_DOM_KEYBOARDEVENT_H
#include "aeongui/dom/UIEvent.hpp"
#include "aeongui/dom/EventModifierInit.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        /** @brief Initialization dictionary for KeyboardEvent construction.
         *  @see https://www.w3.org/TR/uievents/#idl-keyboardeventinit
         */
        struct KeyboardEventInit : EventModifierInit
        {
            DOMString key{""};       ///< The key value of the key pressed.
            DOMString code{""};      ///< The physical key code.
            unsigned long location{0}; ///< Key location on the device.
            bool repeat{false};      ///< Whether the key is held down (repeating).
            bool isComposing{false}; ///< Whether part of a composition session.
        };

        /** @brief Keyboard event providing key-related information.
         *
         *  Provides specific contextual information associated with
         *  keyboard devices.
         *  @see https://www.w3.org/TR/uievents/#idl-keyboardevent
         */
        class KeyboardEvent : public UIEvent
        {
        public:
            static constexpr unsigned long DOM_KEY_LOCATION_STANDARD = 0x00; ///< Standard key location.
            static constexpr unsigned long DOM_KEY_LOCATION_LEFT     = 0x01; ///< Left key location.
            static constexpr unsigned long DOM_KEY_LOCATION_RIGHT    = 0x02; ///< Right key location.
            static constexpr unsigned long DOM_KEY_LOCATION_NUMPAD   = 0x03; ///< Numpad key location.

            /** @brief Construct a KeyboardEvent.
             *  @param type The event type name (e.g., "keydown", "keyup").
             *  @param eventInitDict Optional initialization dictionary.
             */
            DLL KeyboardEvent ( const DOMString& type, const KeyboardEventInit& eventInitDict = {} );

            /** @brief Get the key value.
             *  @return The key value string. */
            DLL const DOMString& key() const;
            /** @brief Get the physical key code.
             *  @return The code value string. */
            DLL const DOMString& code() const;
            /** @brief Get the key location.
             *  @return One of the DOM_KEY_LOCATION constants. */
            DLL unsigned long location() const;
            /** @brief Check if Control modifier was active.
             *  @return true if active. */
            DLL bool ctrlKey() const;
            /** @brief Check if Shift modifier was active.
             *  @return true if active. */
            DLL bool shiftKey() const;
            /** @brief Check if Alt modifier was active.
             *  @return true if active. */
            DLL bool altKey() const;
            /** @brief Check if Meta modifier was active.
             *  @return true if active. */
            DLL bool metaKey() const;
            /** @brief Check if the key is repeating.
             *  @return true if repeating. */
            DLL bool repeat() const;
            /** @brief Check if part of a composition session.
             *  @return true if composing. */
            DLL bool isComposing() const;
            /** @brief Query the state of a modifier key.
             *  @param keyArg The modifier key name (e.g., "Control", "Shift", "CapsLock").
             *  @return true if the modifier is active. */
            DLL bool getModifierState ( const DOMString& keyArg ) const;

        private:
            DOMString m_key;
            DOMString m_code;
            unsigned long m_location{0};
            bool m_ctrlKey{false};
            bool m_shiftKey{false};
            bool m_altKey{false};
            bool m_metaKey{false};
            bool m_repeat{false};
            bool m_isComposing{false};
            bool m_modifierAltGraph{false};
            bool m_modifierCapsLock{false};
            bool m_modifierFn{false};
            bool m_modifierFnLock{false};
            bool m_modifierHyper{false};
            bool m_modifierNumLock{false};
            bool m_modifierScrollLock{false};
            bool m_modifierSuper{false};
            bool m_modifierSymbol{false};
            bool m_modifierSymbolLock{false};
        };
    }
}
#endif
