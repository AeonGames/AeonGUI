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

#include <stdexcept>
#include <iostream>
#include "aeongui/Duktape.h"
#include "duktape.h"
#include "duk_console.h"

namespace AeonGUI
{
    Duktape::Duktape() : mDukContext{duk_create_heap_default() }
    {
        std::cout << "Duktape" << std::endl;
        if ( !mDukContext )
        {
            throw std::runtime_error ( "Failed to create default duktape heap." );
        }
        // Register window as an alias for the global object
        duk_eval_string_noresult ( mDukContext,
                                   R"(
Object.defineProperty(new Function('return this')(), 'window', {
    value: new Function('return this')(),
    writable: false, enumerable: true, configurable: false});
)");
        // Register Console
        duk_console_init ( mDukContext, 0 );
    }
    Duktape::~Duktape()
    {
        if ( mDukContext )
        {
            duk_destroy_heap ( mDukContext );
            mDukContext = nullptr;
        }
    }
    void Duktape::Eval ( const std::string& aString )
    {
        duk_eval_string_noresult ( mDukContext, aString.c_str() );
    }
}
