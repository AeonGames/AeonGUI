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

#include <vector>
#include <functional>
#include <memory>
#include <utility>
#include <tuple>
#include <algorithm>
#include <iostream>
#include "aeongui/StringLiteral.h"
#include "aeongui/ElementFactory.h"
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

namespace AeonGUI
{
    using ConstructorTuple = std::tuple <
                             std::function < DOM::Element* ( const std::u8string& aTagName, const AttributeMap& aAttributeMap, DOM::Node* aParent ) >,
                             std::function < void ( DOM::Element* ) >>;

    using Constructor = std::tuple<StringLiteral, ConstructorTuple>;

    template<class T> Constructor MakeConstructor ( StringLiteral aId )
    {
        return
            Constructor
        {
            aId,
            ConstructorTuple {
                [] ( const std::u8string & aTagName, const AttributeMap & aAttributeMap, DOM::Node * aParent )
                {
                    return new T{ aTagName, aAttributeMap, aParent };
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
        MakeConstructor<DOM::SVGSVGElement> ( u8"svg" ),
        MakeConstructor<DOM::SVGGElement> ( u8"g" ),
        MakeConstructor<DOM::SVGPathElement> ( u8"path" ),
        MakeConstructor<DOM::SVGRectElement> ( u8"rect" ),
        MakeConstructor<DOM::SVGLineElement> ( u8"line" ),
        MakeConstructor<DOM::SVGPolylineElement> ( u8"polyline" ),
        MakeConstructor<DOM::SVGPolygonElement> ( u8"polygon" ),
        MakeConstructor<DOM::SVGCircleElement> ( u8"circle" ),
        MakeConstructor<DOM::SVGEllipseElement> ( u8"ellipse" ),
        MakeConstructor<DOM::SVGDefsElement> ( u8"defs" ),
        MakeConstructor<DOM::SVGUseElement> ( u8"use" ),
        MakeConstructor<DOM::SVGLinearGradientElement> ( u8"linearGradient" ),
        MakeConstructor<DOM::SVGStopElement> ( u8"stop" ),
    };

    DOM::Element* Construct ( const char8_t* aIdentifier, const AttributeMap& aAttributeMap, DOM::Node* aParent )
    {
        auto it = std::find_if ( Constructors.begin(), Constructors.end(),
                                 [aIdentifier] ( const Constructor & aConstructor )
        {
            return std::get<0> ( aConstructor ) == aIdentifier ;
        } );
        if ( it != Constructors.end() )
        {
            return std::get<0> ( std::get<1> ( *it ) ) ( aIdentifier, aAttributeMap, aParent );
        }
        return new DOM::Element { reinterpret_cast<const char8_t*> ( aIdentifier ), aAttributeMap, aParent };
    }

    void Destroy ( const char8_t* aIdentifier, DOM::Element* aElement )
    {
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
                               const std::function < DOM::Element* ( const std::u8string& aTagName, const AttributeMap& aAttributeMap, DOM::Node* aParent ) > & aConstructor,
                               const std::function < void ( DOM::Element* ) > & aDestructor
                             )
    {
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
        for ( auto& i : Constructors )
        {
            if ( !aEnumerator ( std::get<0> ( i ) ) )
            {
                return;
            }
        }
    }
}
