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

#include <stdexcept>
#include "aeongui/Duktape.h"
#include "duktape.h"

namespace AeonGUI
{
    Duktape::Duktape() : mDukContext{duk_create_heap_default() }
    {
        if ( !mDukContext )
        {
            throw std::runtime_error ( "Failed to create default duktape heap." );
        }
    }
    Duktape::~Duktape()
    {
        if ( mDukContext )
        {
            duk_destroy_heap ( mDukContext );
            mDukContext = nullptr;
        }
    }
}
