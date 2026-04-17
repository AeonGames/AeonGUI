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
#ifndef AEONGUI_DOM_UIEVENT_H
#define AEONGUI_DOM_UIEVENT_H
#include "aeongui/dom/Event.hpp"
#include <cstdint>

namespace AeonGUI
{
    namespace DOM
    {
        class Window;
        /** @brief Initialization dictionary for UIEvent construction.
         *  @see https://www.w3.org/TR/uievents/#idl-uieventinit
         */
        struct UIEventInit : EventInit
        {
            Window* view{nullptr}; ///< The Window from which the event was generated.
            int32_t detail{0};        ///< Application-specific detail information.
        };

        /** @brief UI Event base class.
         *
         *  Provides specific contextual information associated with
         *  User Interface events.
         *  @see https://www.w3.org/TR/uievents/#idl-uievent
         */
        class UIEvent : public Event
        {
        public:
            /** @brief Construct a UIEvent.
             *  @param type The event type name.
             *  @param eventInitDict Optional initialization dictionary.
             */
            AEONGUI_DLL UIEvent ( const DOMString& type, const UIEventInit& eventInitDict = {} );
            /** @brief Get the Window associated with this event.
             *  @return Pointer to the Window, or nullptr. */
            AEONGUI_DLL Window* view() const;
            /** @brief Get the detail value.
             *  @return Application-specific detail. */
            AEONGUI_DLL int32_t detail() const;
        private:
            Window* m_view{nullptr};
            int32_t m_detail{0};
        };
    }
}
#endif
