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
#ifndef AEONGUI_DOM_MOUSEEVENT_H
#define AEONGUI_DOM_MOUSEEVENT_H
#include "aeongui/dom/UIEvent.hpp"
#include "aeongui/dom/EventModifierInit.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class EventTarget;

        /** @brief Initialization dictionary for MouseEvent construction.
         *  @see https://www.w3.org/TR/uievents/#idl-mouseeventinit
         */
        struct MouseEventInit : EventModifierInit
        {
            double screenX{0.0};   ///< Horizontal coordinate relative to screen origin.
            double screenY{0.0};   ///< Vertical coordinate relative to screen origin.
            double clientX{0.0};   ///< Horizontal coordinate relative to viewport.
            double clientY{0.0};   ///< Vertical coordinate relative to viewport.
            short button{0};       ///< Button number that changed state.
            unsigned short buttons{0}; ///< Bitmask of currently pressed buttons.
            EventTarget* relatedTarget{nullptr}; ///< Related target for enter/leave events.
        };

        /** @brief Mouse event providing pointer-related information.
         *
         *  Provides specific contextual information associated with
         *  mouse devices.
         *  @see https://www.w3.org/TR/uievents/#idl-mouseevent
         */
        class MouseEvent : public UIEvent
        {
        public:
            /** @brief Construct a MouseEvent.
             *  @param type The event type name (e.g., "click", "mousedown", "mouseup", "mousemove").
             *  @param eventInitDict Optional initialization dictionary.
             */
            AEONGUI_DLL MouseEvent ( const DOMString& type, const MouseEventInit& eventInitDict = {} );

            /** @brief Horizontal coordinate relative to screen origin, as a double.
             *  @return Screen X coordinate. */
            AEONGUI_DLL double screenX() const;
            /** @brief Vertical coordinate relative to screen origin, as a double.
             *  @return Screen Y coordinate. */
            AEONGUI_DLL double screenY() const;
            /** @brief Horizontal coordinate relative to viewport, as a double.
             *  @return Client X coordinate. */
            AEONGUI_DLL double clientX() const;
            /** @brief Vertical coordinate relative to viewport, as a double.
             *  @return Client Y coordinate. */
            AEONGUI_DLL double clientY() const;
            /** @brief Check if Control modifier was active.
             *  @return true if active. */
            AEONGUI_DLL bool ctrlKey() const;
            /** @brief Check if Shift modifier was active.
             *  @return true if active. */
            AEONGUI_DLL bool shiftKey() const;
            /** @brief Check if Alt modifier was active.
             *  @return true if active. */
            AEONGUI_DLL bool altKey() const;
            /** @brief Check if Meta modifier was active.
             *  @return true if active. */
            AEONGUI_DLL bool metaKey() const;
            /** @brief Get the button number that changed state.
             *  @return The button number (0=primary, 1=auxiliary, 2=secondary). */
            AEONGUI_DLL short button() const;
            /** @brief Get the bitmask of currently pressed buttons.
             *  @return The buttons bitmask. */
            AEONGUI_DLL unsigned short buttons() const;
            /** @brief Get the related target.
             *  @return Pointer to the related EventTarget, or nullptr. */
            AEONGUI_DLL EventTarget* relatedTarget() const;
            /** @brief Query the state of a modifier key.
             *  @param keyArg The modifier key name (e.g., "Control", "Shift").
             *  @return true if the modifier is active. */
            AEONGUI_DLL bool getModifierState ( const DOMString& keyArg ) const;

        protected:
            double m_screenX{0.0}; ///< Horizontal coordinate in screen space.
            double m_screenY{0.0}; ///< Vertical coordinate in screen space.
            double m_clientX{0.0}; ///< Horizontal coordinate in viewport space.
            double m_clientY{0.0}; ///< Vertical coordinate in viewport space.
            bool m_ctrlKey{false}; ///< State of Control modifier.
            bool m_shiftKey{false}; ///< State of Shift modifier.
            bool m_altKey{false}; ///< State of Alt modifier.
            bool m_metaKey{false}; ///< State of Meta modifier.
            short m_button{0}; ///< Button that changed state.
            unsigned short m_buttons{0}; ///< Bitmask of currently pressed buttons.
            EventTarget* m_relatedTarget{nullptr}; ///< Related target for enter/leave style events.
            bool m_modifierAltGraph{false}; ///< State of AltGraph modifier.
            bool m_modifierCapsLock{false}; ///< State of CapsLock modifier.
            bool m_modifierFn{false}; ///< State of Fn modifier.
            bool m_modifierFnLock{false}; ///< State of FnLock modifier.
            bool m_modifierHyper{false}; ///< State of Hyper modifier.
            bool m_modifierNumLock{false}; ///< State of NumLock modifier.
            bool m_modifierScrollLock{false}; ///< State of ScrollLock modifier.
            bool m_modifierSuper{false}; ///< State of Super modifier.
            bool m_modifierSymbol{false}; ///< State of Symbol modifier.
            bool m_modifierSymbolLock{false}; ///< State of SymbolLock modifier.
        };
    }
}
#endif
