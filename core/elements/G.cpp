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
#include <iostream>
#include "G.h"
#include "aeongui/Canvas.h"
#include "aeongui/AttributeMap.h"

namespace AeonGUI
{
    namespace Elements
    {
        G::G ( xmlElementPtr aXmlElementPtr ) : Element {aXmlElementPtr}
        {
        }
        G::~G()
        {
        }
        void G::DrawStart ( Canvas& aCanvas ) const
        {
            for ( auto& i : mAttributeMap )
            {
                if ( i.first == "fill" )
                {
                    aCanvas.SetFillColor ( std::get<Color> ( i.second ) );
                }
                else if ( i.first == "stroke" )
                {
                    aCanvas.SetStrokeColor ( std::get<Color> ( i.second ) );
                }
                else if ( i.first == "stroke-width" )
                {
                    aCanvas.SetStrokeWidth ( std::get<double> ( i.second ) );
                }
            }
        }
        void G::DrawFinish ( Canvas& aCanvas ) const
        {
        }
    }
}