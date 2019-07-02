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

namespace AeonGUI
{
    namespace Elements
    {

        int ParsePathData ( std::vector<DrawCommand>& aPath, const char* s );
        Path::Path ( xmlElementPtr aXmlElementPtr ) : Element ( aXmlElementPtr ), mPath{}
        {
            if ( HasAttr ( "d" ) )
            {
                if ( int error = ParsePathData ( mPath, GetAttr ( "d" ) ) )
                {
                    std::cerr << error << std::endl;
                }
                for ( auto& i : mPath )
                {
                    std::cout << static_cast<char> ( i.GetCommand() ) << " " << i.GetVertex() [0] << " " << i.GetVertex() [1] << std::endl;
                }
            }
        }
        Path::~Path()
        {
        }
        void Path::Render ( Canvas& aCanvas ) const
        {
        }
    }
}
