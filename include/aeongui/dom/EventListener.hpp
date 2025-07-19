/*
Copyright (C) 2025 Rodrigo Jose Hernandez Cordoba

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
        class EventListener
        {
        public:
            virtual ~EventListener() = 0;
            virtual void handleEvent ( Event& event ) = 0;
        };
    }
}
#endif
