/*
Copyright (C) 2019,2020,2023-2025,2026 Rodrigo Jose Hernandez Cordoba

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
    /** @brief Construct a DOM Element by tag name.
     *  @param aIdentifier The element tag name (e.g. "rect", "circle").
     *  @param aAttributeMap The element's attributes.
     *  @param aParent The parent node.
     *  @return A unique_ptr to the newly created Element, or nullptr on failure.
     */
    AEONGUI_DLL std::unique_ptr<DOM::Element> Construct ( const char* aIdentifier, AttributeMap&& aAttributeMap, DOM::Node* aParent );
    /** @brief Destroy a DOM Element by tag name.
     *  @param aIdentifier The element tag name.
     *  @param aElement The element to destroy.
     */
    AEONGUI_DLL void Destroy ( const char* aIdentifier, DOM::Element* aElement );
    /** @brief Register a constructor/destructor pair for a given element tag.
     *  @param aIdentifier The element tag name.
     *  @param aConstructor Factory function that creates the element.
     *  @param aDestructor  Cleanup function called when the element is destroyed.
     *  @return true if registration succeeded.
     */
    AEONGUI_DLL bool RegisterConstructor ( const StringLiteral& aIdentifier,
                                           const std::function < std::unique_ptr<DOM::Element> ( AttributeMap&&, DOM::Node* ) > & aConstructor,
                                           const std::function < void ( DOM::Element* ) > & aDestructor );
    /** @brief Unregister a previously registered element constructor.
     *  @param aIdentifier The element tag name to unregister.
     *  @return true if unregistration succeeded.
     */
    AEONGUI_DLL bool UnregisterConstructor ( const StringLiteral& aIdentifier );
    /** @brief Enumerate all registered element constructors.
     *  @param aEnumerator Callback invoked for each registered tag; return false to stop.
     */
    AEONGUI_DLL void EnumerateConstructors ( const std::function<bool ( const StringLiteral& ) >& aEnumerator );
    /** @brief Initialize the element factory. Call once at startup. */
    AEONGUI_DLL void Initialize();
    /** @brief Finalize the element factory. Call once at shutdown. */
    AEONGUI_DLL void Finalize();
    /** @brief Add an initializer/finalizer pair for newly created elements.
     *  @param aInitializer Called when an element is created.
     *  @param aFinalizer   Called when an element is destroyed.
     *  @return true if the initializer was added.
     */
    AEONGUI_DLL bool AddInitializer (
        const std::function < void ( DOM::Element* ) > & aInitializer,
        const std::function < void ( DOM::Element* ) > & aFinalizer );
    /** @brief Remove a previously added initializer/finalizer pair.
     *  @param aInitializer The initializer function to remove.
     *  @param aFinalizer   The finalizer function to remove.
     *  @return true if the initializer pair was found and removed.
     */
    AEONGUI_DLL bool RemoveInitializer (
        const std::function < void ( DOM::Element* ) > & aInitializer,
        const std::function < void ( DOM::Element* ) > & aFinalizer );
}
#endif
