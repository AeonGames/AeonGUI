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
#ifndef AEONGUI_DOM_EVENTTARGET_H
#define AEONGUI_DOM_EVENTTARGET_H
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>
#include <functional>
#include <unordered_map>
#include "aeongui/Platform.h"
#include "AnyType.hpp"
#include "DOMString.hpp"
#include "EventListener.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class AbortSignal;
        class Event;
        class EventTarget;
        using EventHandler = std::function<void ( Event& ) >;
        struct EventListenerOptions
        {
            bool capture{false};
        };

        struct AddEventListenerOptions : EventListenerOptions
        {
            bool passive;
            bool once{false};
            AbortSignal* signal{nullptr};
        };

        class EventTarget
        {
        public:
            virtual ~EventTarget() = 0;
            void addEventListener ( const DOMString& type, EventListener* callback, const std::variant<std::monostate, AddEventListenerOptions, bool>& options = {} );
            void removeEventListener ( const DOMString& type, EventListener* callback, const std::variant<std::monostate, EventListenerOptions, bool>& options = {} );
            virtual bool dispatchEvent ( Event& event );
        private:
            std::unordered_map<DOMString, std::vector<EventListener* >> mEventListeners{};
        };

        class AbortSignal :
            EventTarget
        {
        public:
            static AbortSignal abort ( const std::optional<AnyType>& reason = std::nullopt );
            static AbortSignal timeout ( uint64_t milliseconds );
            static AbortSignal any ( std::vector<AbortSignal> signals );

            bool aborted() const;
            const AnyType & reason() const;
            void throwIfAborted();

            EventHandler onabort;
            virtual ~AbortSignal() = default;
        private:
            AbortSignal() = default;
            bool m_aborted;
            AnyType m_reason;
        };
    }
}
#endif
