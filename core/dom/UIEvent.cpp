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
#include "aeongui/dom/UIEvent.hpp"
namespace AeonGUI
{
    namespace DOM
    {
        UIEvent::UIEvent ( const DOMString& type, const UIEventInit& eventInitDict ) :
            Event ( type, EventInit{eventInitDict.bubbles, eventInitDict.cancelable, eventInitDict.composed} ),
            m_view ( eventInitDict.view ),
            m_detail ( eventInitDict.detail )
        {
        }
        Window* UIEvent::view() const
        {
            return m_view;
        }
        long UIEvent::detail() const
        {
            return m_detail;
        }
    }
}
