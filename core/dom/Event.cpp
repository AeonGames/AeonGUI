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
#include "aeongui/dom/Event.hpp"
namespace AeonGUI
{
    namespace DOM
    {
        Event::Event ( const DOMString& type, const std::optional<EventInit>& eventInitDict ) : m_type ( type ),
            m_bubbles ( eventInitDict.has_value() ? eventInitDict->bubbles : false ),
            m_cancelable ( eventInitDict.has_value() ? eventInitDict->cancelable : false ),
            m_composed ( eventInitDict.has_value() ? eventInitDict->composed : false ),
            m_isTrusted ( false ),
            m_timeStamp ( std::chrono::high_resolution_clock::now() )
        {
        }
        const std::vector<EventTarget*>& Event::composedPath() const
        {
            return m_composedPath;
        }
        void Event::stopPropagation()
        {
            m_stopPropagation = true;
        }
        void Event::stopImmediatePropagation()
        {
            m_stopPropagation = true;
            m_stopImmediatePropagation = true;
        }
        void Event::preventDefault()
        {
            if ( m_cancelable )
            {
                m_defaultPrevented = true;
            }
        }
        void Event::setTrusted ( bool trusted )
        {
            m_isTrusted = trusted;
        }
    }
}
