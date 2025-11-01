/*
Copyright (C) 2025 Rodrigo Jose Hernandez Cordoba

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

#include "aeongui/PangoTextLayout.hpp"
#include <pango/pango.h>
#include <pango/pangocairo.h>
#include <stdexcept>
#include <iostream>
namespace AeonGUI
{
    PangoTextLayout::PangoTextLayout ( const CairoCanvas& aCanvas ) :
        mCairoContext{cairo_create ( reinterpret_cast<cairo_surface_t*> ( aCanvas.GetNativeSurface() ) ), cairo_destroy},
        mLayout{pango_cairo_create_layout ( mCairoContext.get() ), g_object_unref}
    {
        if ( mCairoContext == nullptr )
        {
            throw std::runtime_error ( "Failed to create Cairo Context" );
        }
        else if ( mLayout == nullptr )
        {
            throw std::runtime_error ( "Failed to create PangoLayout" );
        }
        std::cout << pango_cairo_font_map_get_default() <<  std::endl;
    }
    PangoTextLayout::~PangoTextLayout () = default;
}