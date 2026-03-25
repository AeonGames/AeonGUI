/*
Copyright (C) 2025,2026 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_DOM_EVENTLISTENER_H
#define AEONGUI_DOM_EVENTLISTENER_H
#include <cstdint>
#include "aeongui/Platform.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class Event;
        /** @brief Interface for objects that handle DOM events.
         *
         *  Implement the handleEvent method to receive event notifications
         *  when registered via EventTarget::addEventListener.
         */
        class EventListener
        {
        public:
            /** @brief Virtual destructor. */
            virtual DLL ~EventListener() = 0;
            /** @brief Called when an event is dispatched to this listener.
             *  @param event The dispatched event.
             */
            virtual void handleEvent ( Event& event ) = 0;
        };
    }
}
#endif
