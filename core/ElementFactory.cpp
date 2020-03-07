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
    using Constructor = std::tuple<StringLiteral, std::function < std::unique_ptr<Element> ( const AttributeMap& aAttributeMap ) >>;
    static std::vector<Constructor> Constructors
    {
        {
            "svg",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<DOM::SVGSVGElement> ( aAttributeMap );
            }
        },
        {
            "g",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<DOM::SVGGElement> ( aAttributeMap );
            }
        },
        {
            "path",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<DOM::SVGPathElement> ( aAttributeMap );
            }
        },
        {
            "rect",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<DOM::SVGRectElement> ( aAttributeMap );
            }
        },
        {
            "line",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<DOM::SVGLineElement> ( aAttributeMap );
            }
        },
        {
            "polyline",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<DOM::SVGPolylineElement> ( aAttributeMap );
            }
        },
        {
            "polygon",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<DOM::SVGPolygonElement> ( aAttributeMap );
            }
        },
        {
            "circle",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<DOM::SVGCircleElement> ( aAttributeMap );
            }
        },
        {
            "ellipse",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<DOM::SVGEllipseElement> ( aAttributeMap );
            }
        },
        {
            "script",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<DOM::Script> ( aAttributeMap );
            }
        },
        {
            "defs",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<DOM::SVGDefsElement> ( aAttributeMap );
            }
        },
        {
            "use",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<DOM::SVGUseElement> ( aAttributeMap );
            }
        },
        {
            "linearGradient",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<DOM::SVGLinearGradientElement> ( aAttributeMap );
            }
        },
        {
            "stop",
            [] ( const AttributeMap & aAttributeMap )
            {
                return std::make_unique<DOM::SVGStopElement> ( aAttributeMap );
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
