/*
Copyright (C) 2019,2020 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/Platform.h"
#include "aeongui/StringLiteral.h"
#include "aeongui/AttributeMap.h"

namespace AeonGUI
{
    class Element;
    DLL Element* Construct ( const char* aIdentifier, const AttributeMap& aAttributeMap );
    DLL void Destroy ( const char* aIdentifier, Element* aElement );
    DLL bool RegisterConstructor ( const StringLiteral& aIdentifier,
                                   const std::function < Element* ( const AttributeMap& ) > & aConstructor,
                                   const std::function < void ( Element* ) > & aDestructor );
    DLL bool UnregisterConstructor ( const StringLiteral& aIdentifier );
    DLL void EnumerateConstructors ( const std::function<bool ( const StringLiteral& ) >& aEnumerator );
    DLL void Initialize ( v8::Isolate* aIsolate );
    DLL void Finalize ( v8::Isolate* aIsolate );
    DLL bool AddInitializer (
        const std::function < void ( Element* ) > & aInitializer,
        const std::function < void ( Element* ) > & aFinalizer );
    DLL bool RemoveInitializer (
        const std::function < void ( Element* ) > & aInitializer,
        const std::function < void ( Element* ) > & aFinalizer );
}
#endif
