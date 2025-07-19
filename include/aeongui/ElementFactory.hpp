/*
Copyright (C) 2019,2020,2023-2025 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_ELEMENTFACTORY_H
#define AEONGUI_ELEMENTFACTORY_H
#include <memory>
#include <functional>
#include <string>
#include "aeongui/Platform.hpp"
#include "aeongui/StringLiteral.hpp"
#include "aeongui/AttributeMap.hpp"

namespace AeonGUI
{
    namespace DOM
    {
        class Node;
        class Element;
    }
    DLL DOM::Element* Construct ( const char* aIdentifier, const AttributeMap& aAttributeMap, DOM::Node* aParent );
    DLL void Destroy ( const char* aIdentifier, DOM::Element* aElement );
    DLL bool RegisterConstructor ( const StringLiteral& aIdentifier,
                                   const std::function < DOM::Element* ( const AttributeMap&, DOM::Node* ) > & aConstructor,
                                   const std::function < void ( DOM::Element* ) > & aDestructor );
    DLL bool UnregisterConstructor ( const StringLiteral& aIdentifier );
    DLL void EnumerateConstructors ( const std::function<bool ( const StringLiteral& ) >& aEnumerator );
    DLL void Initialize();
    DLL void Finalize();
    DLL bool AddInitializer (
        const std::function < void ( DOM::Element* ) > & aInitializer,
        const std::function < void ( DOM::Element* ) > & aFinalizer );
    DLL bool RemoveInitializer (
        const std::function < void ( DOM::Element* ) > & aInitializer,
        const std::function < void ( DOM::Element* ) > & aFinalizer );
}
#endif
