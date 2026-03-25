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
#ifndef AEONGUI_DOM_FOCUSEVENT_H
#define AEONGUI_DOM_FOCUSEVENT_H
#include "aeongui/dom/UIEvent.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class EventTarget;

        /** @brief Initialization dictionary for FocusEvent construction.
         *  @see https://www.w3.org/TR/uievents/#idl-focuseventinit
         */
        struct FocusEventInit : UIEventInit
        {
            EventTarget* relatedTarget{nullptr}; ///< The secondary target losing or gaining focus.
        };

        /** @brief Focus event indicating a change in focus.
         *
         *  Provides specific contextual information associated with
         *  focus events: focus, blur, focusin, focusout.
         *  @see https://www.w3.org/TR/uievents/#idl-focusevent
         */
        class FocusEvent : public UIEvent
        {
        public:
            /** @brief Construct a FocusEvent.
             *  @param type The event type name (e.g., "focus", "blur", "focusin", "focusout").
             *  @param eventInitDict Optional initialization dictionary.
             */
            DLL FocusEvent ( const DOMString& type, const FocusEventInit& eventInitDict = {} );

            /** @brief Get the related target.
             *  @return Pointer to the related EventTarget, or nullptr. */
            DLL EventTarget* relatedTarget() const;

        private:
            EventTarget* m_relatedTarget{nullptr};
        };
    }
}
#endif
