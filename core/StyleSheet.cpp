/*
Copyright (C) 2023,2024 Rodrigo Jose Hernandez Cordoba

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
#include "aeongui/StyleSheet.h"
#include <iostream>
#include <libcss/libcss.h>

namespace AeonGUI
{
    void css_stylesheet_deleter::operator() ( css_stylesheet* p )
    {
        css_error code{ css_stylesheet_destroy ( p ) };
        if ( code != CSS_OK )
        {
            std::cerr << "css_stylesheet_destroy failed with code: " << code << std::endl;
        }
    }

    void css_select_ctx_deleter::operator() ( css_select_ctx* p )
    {
        css_error code{ css_select_ctx_destroy ( p ) };
        if ( code != CSS_OK )
        {
            std::cerr << "css_select_ctx_destroy failed with code: " << code << std::endl;
        }
    }

    void css_select_results_deleter::operator() ( css_select_results* p )
    {
        css_error code{ css_select_results_destroy ( p ) };
        if ( code != CSS_OK )
        {
            std::cerr << "css_select_results_destroy failed with code: " << code << std::endl;
        }
    }
}