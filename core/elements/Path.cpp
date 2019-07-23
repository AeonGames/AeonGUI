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
#include "Path.h"
#include "aeongui/Canvas.h"

namespace AeonGUI
{
    namespace Elements
    {
        int ParsePathData ( std::vector<DrawType>& aPath, const char* s );
        Path::Path ( xmlElementPtr aXmlElementPtr ) : Element ( aXmlElementPtr ), mPath{}
        {
            if ( HasAttr ( "d" ) )
            {
                std::vector<DrawType> path;
                if ( int error = ParsePathData ( path, GetAttr ( "d" ) ) )
                {
                    std::cerr << error << std::endl;
                }
                mPath.Construct ( path );
            }
        }

        Path::~Path() = default;

        void Path::DrawStart ( Canvas& aCanvas ) const
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
                else if ( i.first == "stroke-opacity" )
                {
                    aCanvas.SetStrokeOpacity ( std::get<double> ( i.second ) );
                }
                else if ( i.first == "fill-opacity" )
                {
                    aCanvas.SetFillOpacity ( std::get<double> ( i.second ) );
                }
            }
            aCanvas.Draw ( mPath );
        }
    }
}
