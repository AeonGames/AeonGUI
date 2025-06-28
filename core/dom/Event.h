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
#ifndef AEONGUI_DOM_EVENT_H
#define AEONGUI_DOM_EVENT_H
#include <cstdint>
#include <optional>
#include <vector>
#include <chrono>
#include "aeongui/Platform.h"
#include "DOMString.h"
#include "EventTarget.h"

namespace AeonGUI
{
    namespace DOM
    {
        using DOMHighResTimeStamp = std::chrono::time_point<std::chrono::high_resolution_clock>;
        struct EventInit
        {
            bool bubbles{false};
            bool cancelable{false};
            bool composed{false};
        };

        class Event
        {
        public:
            Event ( const DOMString& type, const std::optional<EventInit>& eventInitDict = {} );
            std::vector<EventTarget> composedPath() const;
            void stopPropagation();
            void stopImmediatePropagation();
            void preventDefault();

            const uint16_t NONE{0};
            const uint16_t CAPTURING_PHASE{1};
            const uint16_t AT_TARGET{2};
            const uint16_t BUBBLING_PHASE{3};

            constexpr const DOMString& type() const
            {
                return m_type;
            }
            constexpr const EventTarget* const target() const
            {
                return m_target;
            }
            constexpr const EventTarget* const currentTarget() const
            {
                return m_currentTarget;
            }
            constexpr uint16_t eventPhase() const
            {
                return m_eventPhase;
            }
            constexpr bool bubbles() const
            {
                return m_bubbles;
            }
            constexpr bool cancelable() const
            {
                return m_cancelable;
            }
            constexpr bool defaultPrevented() const
            {
                return m_defaultPrevented;
            }
            constexpr bool composed() const
            {
                return m_composed;
            }
            constexpr bool isTrusted() const
            {
                return m_isTrusted;
            }
            constexpr const DOMHighResTimeStamp& timeStamp() const
            {
                return m_timeStamp;
            }

        private:
            DOMString m_type;
            EventTarget* m_target{};
            EventTarget* m_currentTarget{};
            uint16_t m_eventPhase{NONE};
            bool m_bubbles{};
            bool m_cancelable{};
            bool m_defaultPrevented{};
            bool m_composed{};
            bool m_isTrusted{};
            DOMHighResTimeStamp m_timeStamp{};
        };
    }
}
#endif
