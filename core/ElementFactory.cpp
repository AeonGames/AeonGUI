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

namespace AeonGUI
{
    using ConstructorTuple = std::tuple <
                             std::function < std::unique_ptr<DOM::Element> ( const DOM::DOMString& aTagName, AttributeMap && aAttributeMap, DOM::Node* aParent ) >,
                             std::function < void ( DOM::Element* ) >>;

    using Constructor = std::tuple<StringLiteral, ConstructorTuple>;

    template<class T> Constructor MakeConstructor ( StringLiteral aId )
    {
        std::cerr << LogLevel::Info << "Registering constructor for: " << aId.GetString() << std::endl;
        return
            Constructor
        {
            aId,
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

    static std::vector<Constructor> Constructors
    {
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
    };

    static std::mutex& GetConstructorsMutex()
    {
        static std::mutex sMutex;
        return sMutex;
    }

    std::unique_ptr<DOM::Element> Construct ( const char* aIdentifier, AttributeMap&& aAttributeMap, DOM::Node* aParent )
    {
        std::lock_guard<std::mutex> lock ( GetConstructorsMutex() );
        auto it = std::find_if ( Constructors.begin(), Constructors.end(),
                                 [aIdentifier] ( const Constructor & aConstructor )
        {
            return std::get<0> ( aConstructor ) == aIdentifier ;
        } );
        if ( it != Constructors.end() )
        {
            return std::get<0> ( std::get<1> ( *it ) ) ( aIdentifier, std::move ( aAttributeMap ), aParent );
        }
        std::cerr << LogLevel::Warning << "No constructor registered for identifier: " << aIdentifier << std::endl;
        return std::make_unique<DOM::Element> (  aIdentifier, std::move ( aAttributeMap ), aParent );
    }

    void Destroy ( const char* aIdentifier, DOM::Element* aElement )
    {
        std::lock_guard<std::mutex> lock ( GetConstructorsMutex() );
        auto it = std::find_if ( Constructors.begin(), Constructors.end(),
                                 [aIdentifier] ( const Constructor & aConstructor )
        {
            return std::get<0> ( aConstructor ) == aIdentifier ;
        } );
        if ( it != Constructors.end() )
        {
            ( std::get<1> ( std::get<1> ( *it ) ) ) ( aElement );
        }
        else
        {
            delete aElement;
        }
    }

    bool RegisterConstructor ( const StringLiteral& aIdentifier,
                               const std::function < std::unique_ptr<DOM::Element> ( const DOM::DOMString& aTagName, AttributeMap&& aAttributeMap, DOM::Node* aParent ) > & aConstructor,
                               const std::function < void ( DOM::Element* ) > & aDestructor
                             )
    {
        std::lock_guard<std::mutex> lock ( GetConstructorsMutex() );
        auto it = std::find_if ( Constructors.begin(), Constructors.end(),
                                 [aIdentifier] ( const Constructor & aConstructor )
        {
            return aIdentifier == std::get<0> ( aConstructor );
        } );
        if ( it == Constructors.end() )
        {
            Constructors.emplace_back ( aIdentifier, ConstructorTuple{aConstructor, aDestructor} );
            return true;
        }
        return false;
    }
    bool UnregisterConstructor ( const StringLiteral& aIdentifier )
    {
        std::lock_guard<std::mutex> lock ( GetConstructorsMutex() );
        auto it = std::find_if ( Constructors.begin(), Constructors.end(),
                                 [aIdentifier] ( const Constructor & aConstructor )
        {
            return aIdentifier == std::get<0> ( aConstructor );
        } );
        if ( it != Constructors.end() )
        {
            Constructors.erase ( it );
            return true;
        }
        return false;
    }
    void EnumerateConstructors ( const std::function<bool ( const StringLiteral& ) >& aEnumerator )
    {
        std::lock_guard<std::mutex> lock ( GetConstructorsMutex() );
        for ( auto& i : Constructors )
        {
            if ( !aEnumerator ( std::get<0> ( i ) ) )
            {
                return;
            }
        }
    }
}
