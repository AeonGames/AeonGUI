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

#include <vector>
#include <functional>
#include <memory>
#include <utility>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <libxml/tree.h>
#include "aeongui/StringLiteral.h"
#include "aeongui/Element.h"
#include "aeongui/ElementFactory.h"
#include "elements/SVGSVGElement.h"
#include "elements/SVGGElement.h"
#include "elements/SVGLinearGradientElement.h"
#include "elements/SVGStopElement.h"
#include "elements/SVGDefsElement.h"
#include "elements/SVGUseElement.h"
#include "elements/SVGPathElement.h"
#include "elements/SVGRectElement.h"
#include "elements/SVGLineElement.h"
#include "elements/SVGPolylineElement.h"
#include "elements/SVGPolygonElement.h"
#include "elements/SVGCircleElement.h"
#include "elements/SVGEllipseElement.h"
#include "elements/Script.h"

namespace AeonGUI
{
    using Constructor = std::tuple<StringLiteral, std::function < std::unique_ptr<Element> ( const AttributeMap& aAttributeMap ) >>;
    static std::vector<Constructor> Constructors
    {
        {
            "svg",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<Elements::SVGSVGElement> ( aAttributeMap );
            }
        },
        {
            "g",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<Elements::SVGGElement> ( aAttributeMap );
            }
        },
        {
            "path",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<Elements::SVGPathElement> ( aAttributeMap );
            }
        },
        {
            "rect",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<Elements::SVGRectElement> ( aAttributeMap );
            }
        },
        {
            "line",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<Elements::SVGLineElement> ( aAttributeMap );
            }
        },
        {
            "polyline",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<Elements::SVGPolylineElement> ( aAttributeMap );
            }
        },
        {
            "polygon",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<Elements::SVGPolygonElement> ( aAttributeMap );
            }
        },
        {
            "circle",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<Elements::SVGCircleElement> ( aAttributeMap );
            }
        },
        {
            "ellipse",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<Elements::SVGEllipseElement> ( aAttributeMap );
            }
        },
        {
            "script",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<Elements::Script> ( aAttributeMap );
            }
        },
        {
            "defs",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<Elements::SVGDefsElement> ( aAttributeMap );
            }
        },
        {
            "use",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<Elements::SVGUseElement> ( aAttributeMap );
            }
        },
        {
            "linearGradient",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<Elements::SVGLinearGradientElement> ( aAttributeMap );
            }
        },
        {
            "stop",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<Elements::SVGStopElement> ( aAttributeMap );
            }
        },
    };

    std::unique_ptr<Element> Construct ( const char* aIdentifier, const AttributeMap& aAttributeMap )
    {
        auto it = std::find_if ( Constructors.begin(), Constructors.end(),
                                 [&aAttributeMap, aIdentifier] ( const Constructor & aConstructor )
        {
            return std::get<0> ( aConstructor ) == aIdentifier ;
        } );
        if ( it != Constructors.end() )
        {
            return std::get<1> ( *it ) ( aAttributeMap );
        }
        return std::make_unique<Element> ( aAttributeMap );
    }
    bool RegisterConstructor ( const StringLiteral& aIdentifier, const std::function < std::unique_ptr<Element> ( const AttributeMap& aAttributeMap ) > & aConstructor )
    {
        auto it = std::find_if ( Constructors.begin(), Constructors.end(),
                                 [aIdentifier] ( const Constructor & aConstructor )
        {
            return aIdentifier == std::get<0> ( aConstructor );
        } );
        if ( it == Constructors.end() )
        {
            Constructors.emplace_back ( aIdentifier, aConstructor );
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
