/******************************************************************************
Copyright 2010-2012 Rodrigo Hernandez Cordoba

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/
#include <algorithm>
#include "Renderer.h"

#error "Unused file."

namespace AeonGUI
{
#if 0
    Image* Renderer::GetImage ( std::string id, uint32_t width, uint32_t height, Image::Format format, Image::Type type, void* data )
    {
        assert ( !id.empty() );
        std::map<std::string, Image*>::iterator image;
        if ( Images.find ( id ) == Images.end() )
        {
            Images[id] = NewImage ( width, height, format, type, data );
            Images[id]->id = id;
        }
        Images[id]->IncRefCount();
        return Images[id];
    }

    void Renderer::ReleaseImage ( Image* image )
    {
        image->DecRefCount();
        if ( image->GetRefCount() == 0 )
        {
            Images.erase ( image->id );
            DeleteImage ( image );
        }
    }
#endif
}