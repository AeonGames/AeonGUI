/*
Copyright (C) 2019 Rodrigo Jose Hernandez Cordoba

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
#include "elements/SVG.h"
#include "elements/G.h"
#include "elements/Path.h"
#include "elements/Rect.h"
#include "elements/Line.h"
#include "elements/Polyline.h"
#include "elements/Polygon.h"
#include "elements/Circle.h"
#include "elements/Ellipse.h"

namespace AeonGUI
{
    using Constructor = std::tuple<StringLiteral, std::function < std::unique_ptr<Element> ( xmlElementPtr aXmlElementPtr ) >>;
    static std::vector<Constructor> Constructors
    {
        {
            "svg",
            [] ( xmlElementPtr aXmlElementPtr )
            {
                return std::make_unique<Elements::SVG> ( aXmlElementPtr );
            }
        },
        {
            "g",
            [] ( xmlElementPtr aXmlElementPtr )
            {
                return std::make_unique<Elements::G> ( aXmlElementPtr );
            }
        },
        {
            "path",
            [] ( xmlElementPtr aXmlElementPtr )
            {
                return std::make_unique<Elements::Path> ( aXmlElementPtr );
            }
        },
        {
            "rect",
            [] ( xmlElementPtr aXmlElementPtr )
            {
                return std::make_unique<Elements::Rect> ( aXmlElementPtr );
            }
        },
        {
            "line",
            [] ( xmlElementPtr aXmlElementPtr )
            {
                return std::make_unique<Elements::Line> ( aXmlElementPtr );
            }
        },
        {
            "polyline",
            [] ( xmlElementPtr aXmlElementPtr )
            {
                return std::make_unique<Elements::Polyline> ( aXmlElementPtr );
            }
        },
        {
            "polygon",
            [] ( xmlElementPtr aXmlElementPtr )
            {
                return std::make_unique<Elements::Polygon> ( aXmlElementPtr );
            }
        },
        {
            "circle",
            [] ( xmlElementPtr aXmlElementPtr )
            {
                return std::make_unique<Elements::Circle> ( aXmlElementPtr );
            }
        },
        {
            "ellipse",
            [] ( xmlElementPtr aXmlElementPtr )
            {
                return std::make_unique<Elements::Ellipse> ( aXmlElementPtr );
            }
        },
    };

    std::unique_ptr<Element> Construct ( xmlElementPtr aXmlElementPtr )
    {
        auto it = std::find_if ( Constructors.begin(), Constructors.end(),
                                 [aXmlElementPtr] ( const Constructor & aConstructor )
        {
            return std::get<0> ( aConstructor ) == reinterpret_cast<const char*> ( aXmlElementPtr->name );
        } );
        if ( it != Constructors.end() )
        {
            return std::get<1> ( *it ) ( aXmlElementPtr );
        }
        return std::make_unique<Element> ( aXmlElementPtr );
    }
    bool RegisterConstructor ( const StringLiteral& aIdentifier, const std::function < std::unique_ptr<Element> ( xmlElementPtr aXmlElementPtr ) > & aConstructor )
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
