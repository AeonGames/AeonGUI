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
#include "aeongui/CompiledDocument.hpp"
#include "aeongui/dom/Document.hpp"
#include "aeongui/dom/Event.hpp"

namespace AeonGUI
{
    CompiledDocument::CompiledDocument() = default;
    CompiledDocument::~CompiledDocument() = default;

    void CompiledDocument::SetCallback ( const std::string& aName, Callback aCallback )
    {
        mCallbacks[aName] = std::move ( aCallback );
    }

    void CompiledDocument::SetProperty ( const std::string& aKey, const std::string& aValue )
    {
        mProperties[aKey] = aValue;
    }

    std::string CompiledDocument::GetProperty ( const std::string& aKey ) const
    {
        auto it = mProperties.find ( aKey );
        return ( it != mProperties.end() ) ? it->second : std::string{};
    }

    DOM::Window* CompiledDocument::window() const
    {
        return mWindow;
    }

    DOM::Document* CompiledDocument::document() const
    {
        return mDocument;
    }

    void CompiledDocument::Emit ( const std::string& aName, const std::string& aDetail )
    {
        auto it = mCallbacks.find ( aName );
        if ( it != mCallbacks.end() && it->second )
        {
            it->second ( aDetail );
        }
        if ( mDocument != nullptr )
        {
            DOM::Event event ( aName, DOM::EventInit{true, true, false} );
            mDocument->dispatchEvent ( event );
        }
    }
}
