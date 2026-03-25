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
#include "aeongui/dom/FocusEvent.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        FocusEvent::FocusEvent ( const DOMString& type, const FocusEventInit& eventInitDict ) :
            UIEvent ( type, UIEventInit{EventInit{eventInitDict.bubbles, eventInitDict.cancelable, eventInitDict.composed}, eventInitDict.view, eventInitDict.detail} ),
            m_relatedTarget{eventInitDict.relatedTarget}
        {
        }

        EventTarget* FocusEvent::relatedTarget() const
        {
            return m_relatedTarget;
        }
    }
}
