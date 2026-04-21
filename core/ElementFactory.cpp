/*
Copyright (C) 2019,2020,2023-2026 Rodrigo Jose Hernandez Cordoba

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

#include <vector>
#include <functional>
#include <memory>
#include <utility>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <mutex>
#include "aeongui/StringLiteral.hpp"
#include "aeongui/LogLevel.hpp"
#include "aeongui/ElementFactory.hpp"
#include "aeongui/dom/Element.hpp"
#include "aeongui/dom/SVGSVGElement.hpp"
#include "aeongui/dom/SVGGElement.hpp"
#include "aeongui/dom/SVGLinearGradientElement.hpp"
#include "aeongui/dom/SVGStopElement.hpp"
#include "aeongui/dom/SVGDefsElement.hpp"
#include "aeongui/dom/SVGUseElement.hpp"
#include "aeongui/dom/SVGPathElement.hpp"
#include "aeongui/dom/SVGRectElement.hpp"
#include "aeongui/dom/SVGLineElement.hpp"
#include "aeongui/dom/SVGPolylineElement.hpp"
#include "aeongui/dom/SVGPolygonElement.hpp"
#include "aeongui/dom/SVGCircleElement.hpp"
#include "aeongui/dom/SVGEllipseElement.hpp"
#include "aeongui/dom/SVGImageElement.hpp"
#include "aeongui/dom/SVGTextElement.hpp"
#include "aeongui/dom/SVGTSpanElement.hpp"
#include "aeongui/dom/SVGTextPathElement.hpp"
#include "aeongui/dom/SVGAnimateElement.hpp"
#include "aeongui/dom/SVGSetElement.hpp"
#include "aeongui/dom/SVGAnimateTransformElement.hpp"
#include "aeongui/dom/SVGAnimateMotionElement.hpp"
#include "aeongui/dom/SVGScriptElement.hpp"
#include "aeongui/dom/SVGFilterElement.hpp"
#include "aeongui/dom/SVGFEDropShadowElement.hpp"

#ifdef AEONGUI_ENABLE_HTML
#include "aeongui/dom/HTMLHtmlElement.hpp"
#include "aeongui/dom/HTMLBodyElement.hpp"
#include "aeongui/dom/HTMLDivElement.hpp"
#include "aeongui/dom/HTMLSpanElement.hpp"
#include "aeongui/dom/HTMLImageElement.hpp"
#include "aeongui/dom/HTMLParagraphElement.hpp"
#endif

namespace AeonGUI
{
    using ConstructorTuple = std::tuple <
                             std::function < std::unique_ptr<DOM::Element> ( const DOM::DOMString& aTagName, AttributeMap && aAttributeMap, DOM::Node* aParent ) >,
                             std::function < void ( DOM::Element* ) >>;

    /// (namespace URI, local name) key. Empty namespace is the wildcard tier.
    using ConstructorKey = std::tuple<StringLiteral, StringLiteral>;
    using Constructor = std::tuple<ConstructorKey, ConstructorTuple>;

    template<class T> Constructor MakeConstructor ( StringLiteral aNamespaceURI, StringLiteral aId )
    {
        std::cerr << LogLevel::Info << "Registering constructor for: {" << aNamespaceURI.GetString() << "} " << aId.GetString() << std::endl;
        return
            Constructor
        {
            ConstructorKey { aNamespaceURI, aId },
            ConstructorTuple {
                [] ( const DOM::DOMString & aTagName, AttributeMap && aAttributeMap, DOM::Node * aParent )
                {
                    return std::make_unique<T> ( aTagName, std::move ( aAttributeMap ), aParent );
                },
                [] ( DOM::Element * aElement ) -> void
                {
                    delete reinterpret_cast<T*> ( aElement );
                }
            }
        };
    }

    template<class T> Constructor MakeConstructor ( StringLiteral aId )
    {
        return MakeConstructor<T> ( StringLiteral{""}, aId );
    }

    static std::vector<Constructor>& GetConstructors()
    {
        // Meyers singleton: guarantees first-use initialization, sidestepping
        // static-init-order issues with HTML registrations that may live in
        // a separate translation unit (HTMLElementFactory.cpp).
        static std::vector<Constructor> sConstructors
        {
            // SVG built-ins are registered under the wildcard namespace so they
            // continue to match SVG documents regardless of whether they declare
            // an xmlns attribute. They also serve as the fallback for inline
            // <svg> inside an XHTML document.
            MakeConstructor<DOM::SVGSVGElement> ( "svg" ),
            MakeConstructor<DOM::SVGGElement> ( "g" ),
            MakeConstructor<DOM::SVGPathElement> ( "path" ),
            MakeConstructor<DOM::SVGRectElement> ( "rect" ),
            MakeConstructor<DOM::SVGLineElement> ( "line" ),
            MakeConstructor<DOM::SVGPolylineElement> ( "polyline" ),
            MakeConstructor<DOM::SVGPolygonElement> ( "polygon" ),
            MakeConstructor<DOM::SVGCircleElement> ( "circle" ),
            MakeConstructor<DOM::SVGEllipseElement> ( "ellipse" ),
            MakeConstructor<DOM::SVGImageElement> ( "image" ),
            MakeConstructor<DOM::SVGDefsElement> ( "defs" ),
            MakeConstructor<DOM::SVGUseElement> ( "use" ),
            MakeConstructor<DOM::SVGLinearGradientElement> ( "linearGradient" ),
            MakeConstructor<DOM::SVGStopElement> ( "stop" ),
            MakeConstructor<DOM::SVGTextElement> ( "text" ),
            MakeConstructor<DOM::SVGTSpanElement> ( "tspan" ),
            MakeConstructor<DOM::SVGTextPathElement> ( "textPath" ),
            MakeConstructor<DOM::SVGAnimateElement> ( "animate" ),
            MakeConstructor<DOM::SVGSetElement> ( "set" ),
            MakeConstructor<DOM::SVGAnimateTransformElement> ( "animateTransform" ),
            MakeConstructor<DOM::SVGAnimateMotionElement> ( "animateMotion" ),
            MakeConstructor<DOM::SVGScriptElement> ( "script" ),
            MakeConstructor<DOM::SVGFilterElement> ( "filter" ),
            MakeConstructor<DOM::SVGFEDropShadowElement> ( "feDropShadow" ),
#ifdef AEONGUI_ENABLE_HTML
            // HTML built-ins are registered explicitly under the XHTML namespace.
            MakeConstructor<DOM::HTMLHtmlElement>      ( StringLiteral{"http://www.w3.org/1999/xhtml"}, StringLiteral{"html"} ),
            MakeConstructor<DOM::HTMLBodyElement>      ( StringLiteral{"http://www.w3.org/1999/xhtml"}, StringLiteral{"body"} ),
            MakeConstructor<DOM::HTMLDivElement>       ( StringLiteral{"http://www.w3.org/1999/xhtml"}, StringLiteral{"div"} ),
            MakeConstructor<DOM::HTMLSpanElement>      ( StringLiteral{"http://www.w3.org/1999/xhtml"}, StringLiteral{"span"} ),
            MakeConstructor<DOM::HTMLImageElement>     ( StringLiteral{"http://www.w3.org/1999/xhtml"}, StringLiteral{"img"} ),
            MakeConstructor<DOM::HTMLParagraphElement> ( StringLiteral{"http://www.w3.org/1999/xhtml"}, StringLiteral{"p"} ),
#endif
        };
        return sConstructors;
    }

    static std::mutex& GetConstructorsMutex()
    {
        static std::mutex sMutex;
        return sMutex;
    }

    /// Find a constructor matching the given (namespace, name). Returns
    /// end() if not found. Does NOT perform fallback — caller decides.
    static std::vector<Constructor>::iterator FindConstructor ( const char* aNamespaceURI, const char* aIdentifier )
    {
        auto& constructors = GetConstructors();
        return std::find_if ( constructors.begin(), constructors.end(),
                              [aNamespaceURI, aIdentifier] ( const Constructor & aConstructor )
        {
            const ConstructorKey& key = std::get<0> ( aConstructor );
            return std::get<0> ( key ) == aNamespaceURI && std::get<1> ( key ) == aIdentifier;
        } );
    }

    std::unique_ptr<DOM::Element> Construct ( const char* aNamespaceURI, const char* aIdentifier, AttributeMap&& aAttributeMap, DOM::Node* aParent )
    {
        std::lock_guard<std::mutex> lock ( GetConstructorsMutex() );
        const char* effective_ns = aNamespaceURI ? aNamespaceURI : "";
        auto it = FindConstructor ( effective_ns, aIdentifier );
        if ( it == GetConstructors().end() && effective_ns[0] != '\0' )
        {
            // Fallback to wildcard namespace.
            it = FindConstructor ( "", aIdentifier );
        }
        if ( it != GetConstructors().end() )
        {
            return std::get<0> ( std::get<1> ( *it ) ) ( aIdentifier, std::move ( aAttributeMap ), aParent );
        }
        std::cerr << LogLevel::Warning << "No constructor registered for identifier: {" << effective_ns << "} " << aIdentifier << std::endl;
        return std::make_unique<DOM::Element> ( aIdentifier, std::move ( aAttributeMap ), aParent );
    }

    std::unique_ptr<DOM::Element> Construct ( const char* aIdentifier, AttributeMap&& aAttributeMap, DOM::Node* aParent )
    {
        return Construct ( "", aIdentifier, std::move ( aAttributeMap ), aParent );
    }

    void Destroy ( const char* aIdentifier, DOM::Element* aElement )
    {
        std::lock_guard<std::mutex> lock ( GetConstructorsMutex() );
        auto it = FindConstructor ( "", aIdentifier );
        if ( it != GetConstructors().end() )
        {
            ( std::get<1> ( std::get<1> ( *it ) ) ) ( aElement );
        }
        else
        {
            delete aElement;
        }
    }

    bool RegisterConstructor ( const StringLiteral& aNamespaceURI,
                               const StringLiteral& aIdentifier,
                               const std::function < std::unique_ptr<DOM::Element> ( const DOM::DOMString& aTagName, AttributeMap&& aAttributeMap, DOM::Node* aParent ) > & aConstructor,
                               const std::function < void ( DOM::Element* ) > & aDestructor
                             )
    {
        std::lock_guard<std::mutex> lock ( GetConstructorsMutex() );
        auto& constructors = GetConstructors();
        auto it = std::find_if ( constructors.begin(), constructors.end(),
                                 [&aNamespaceURI, &aIdentifier] ( const Constructor & aConstructor )
        {
            const ConstructorKey& key = std::get<0> ( aConstructor );
            return aNamespaceURI == std::get<0> ( key ) && aIdentifier == std::get<1> ( key );
        } );
        if ( it == constructors.end() )
        {
            constructors.emplace_back ( ConstructorKey{aNamespaceURI, aIdentifier}, ConstructorTuple{aConstructor, aDestructor} );
            return true;
        }
        return false;
    }

    bool RegisterConstructor ( const StringLiteral& aIdentifier,
                               const std::function < std::unique_ptr<DOM::Element> ( const DOM::DOMString& aTagName, AttributeMap&& aAttributeMap, DOM::Node* aParent ) > & aConstructor,
                               const std::function < void ( DOM::Element* ) > & aDestructor
                             )
    {
        return RegisterConstructor ( StringLiteral{""}, aIdentifier, aConstructor, aDestructor );
    }

    bool UnregisterConstructor ( const StringLiteral& aNamespaceURI, const StringLiteral& aIdentifier )
    {
        std::lock_guard<std::mutex> lock ( GetConstructorsMutex() );
        auto& constructors = GetConstructors();
        auto it = std::find_if ( constructors.begin(), constructors.end(),
                                 [&aNamespaceURI, &aIdentifier] ( const Constructor & aConstructor )
        {
            const ConstructorKey& key = std::get<0> ( aConstructor );
            return aNamespaceURI == std::get<0> ( key ) && aIdentifier == std::get<1> ( key );
        } );
        if ( it != constructors.end() )
        {
            constructors.erase ( it );
            return true;
        }
        return false;
    }

    bool UnregisterConstructor ( const StringLiteral& aIdentifier )
    {
        return UnregisterConstructor ( StringLiteral{""}, aIdentifier );
    }

    void EnumerateConstructors ( const std::function<bool ( const StringLiteral& ) >& aEnumerator )
    {
        std::lock_guard<std::mutex> lock ( GetConstructorsMutex() );
        for ( auto& i : GetConstructors() )
        {
            if ( !aEnumerator ( std::get<1> ( std::get<0> ( i ) ) ) )
            {
                return;
            }
        }
    }
}
