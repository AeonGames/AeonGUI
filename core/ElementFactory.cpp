/*
Copyright (C) 2019,2020,2023 Rodrigo Jose Hernandez Cordoba

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
#include "dom/Element.h"
#include "dom/SVGSVGElement.h"
#include "dom/SVGGElement.h"
#include "dom/SVGLinearGradientElement.h"
#include "dom/SVGStopElement.h"
#include "dom/SVGDefsElement.h"
#include "dom/SVGUseElement.h"
#include "dom/SVGPathElement.h"
#include "dom/SVGRectElement.h"
#include "dom/SVGLineElement.h"
#include "dom/SVGPolylineElement.h"
#include "dom/SVGPolygonElement.h"
#include "dom/SVGCircleElement.h"
#include "dom/SVGEllipseElement.h"
#include "dom/Script.h"

namespace AeonGUI
{
    using ConstructorTuple = std::tuple <
                             std::function < Element* ( const std::string& aTagName, const AttributeMap& aAttributeMap ) >,
                             std::function < void ( Element* ) >>;

    using Constructor = std::tuple<StringLiteral, ConstructorTuple>;

    template<class T> Constructor MakeConstructor ( StringLiteral aId )
    {
        return
            Constructor
        {
            aId,
            ConstructorTuple {
                [] ( const std::string & aTagName, const AttributeMap & aAttributeMap )
                {
                    return new T{ aTagName, aAttributeMap };
                },
                [] ( Element * aElement ) -> void
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
        MakeConstructor<DOM::Script> ( "script" ),
        MakeConstructor<DOM::SVGDefsElement> ( "defs" ),
        MakeConstructor<DOM::SVGUseElement> ( "use" ),
        MakeConstructor<DOM::SVGLinearGradientElement> ( "linearGradient" ),
        MakeConstructor<DOM::SVGStopElement> ( "stop" ),
    };

    Element* Construct ( const char* aIdentifier, const AttributeMap& aAttributeMap )
    {
        auto it = std::find_if ( Constructors.begin(), Constructors.end(),
                                 [aIdentifier] ( const Constructor & aConstructor )
        {
            return std::get<0> ( aConstructor ) == aIdentifier ;
        } );
        if ( it != Constructors.end() )
        {
            return std::get<0> ( std::get<1> ( *it ) ) ( aIdentifier, aAttributeMap );
        }
        return new Element { aIdentifier, aAttributeMap };
    }

    void Destroy ( const char* aIdentifier, Element* aElement )
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
                               const std::function < Element* ( const std::string& aTagName, const AttributeMap& aAttributeMap ) > & aConstructor,
                               const std::function < void ( Element* ) > & aDestructor
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
