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
#include <cairo.h>
#include "aeongui/Workspace.h"

namespace AeonGUI
{
    Workspace::Workspace ( uint32_t aWidth, uint32_t aHeight ) :
        mCairoSurface{cairo_image_surface_create ( CAIRO_FORMAT_ARGB32, aWidth, aHeight ) },
        mCairoContext{cairo_create ( reinterpret_cast<cairo_surface_t*> ( mCairoSurface ) ) }
    {
        cairo_select_font_face ( reinterpret_cast<cairo_t*> ( mCairoContext ), "serif", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD );
        cairo_set_font_size ( reinterpret_cast<cairo_t*> ( mCairoContext ), 32.0 );
        cairo_set_source_rgb ( reinterpret_cast<cairo_t*> ( mCairoContext ), 0.0, 0.0, 1.0 );
        cairo_move_to ( reinterpret_cast<cairo_t*> ( mCairoContext ), 10.0, 50.0 );
        cairo_show_text ( reinterpret_cast<cairo_t*> ( mCairoContext ), "Hello, world" );
    }
    Workspace::~Workspace()
    {
        if ( mCairoContext )
        {
            cairo_destroy ( reinterpret_cast<cairo_t*> ( mCairoContext ) );
        }
        if ( mCairoSurface )
        {
            cairo_surface_destroy ( reinterpret_cast<cairo_surface_t*> ( mCairoSurface ) );
        }
    }

    void Workspace::Resize ( uint32_t aWidth, uint32_t aHeight )
    {
        if ( mCairoContext )
        {
            cairo_destroy ( reinterpret_cast<cairo_t*> ( mCairoContext ) );
        }
        if ( mCairoSurface )
        {
            cairo_surface_destroy ( reinterpret_cast<cairo_surface_t*> ( mCairoSurface ) );
        }
        mCairoSurface = cairo_image_surface_create ( CAIRO_FORMAT_ARGB32, aWidth, aHeight );
        mCairoContext = cairo_create ( reinterpret_cast<cairo_surface_t*> ( mCairoSurface ) );
        cairo_select_font_face ( reinterpret_cast<cairo_t*> ( mCairoContext ), "serif", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD );
        cairo_set_font_size ( reinterpret_cast<cairo_t*> ( mCairoContext ), 32.0 );
        cairo_set_source_rgb ( reinterpret_cast<cairo_t*> ( mCairoContext ), 0.0, 0.0, 1.0 );
        cairo_move_to ( reinterpret_cast<cairo_t*> ( mCairoContext ), 10.0, 50.0 );
        cairo_show_text ( reinterpret_cast<cairo_t*> ( mCairoContext ), "Hello, world" );
    }

    const uint8_t* Workspace::GetData() const
    {
        return cairo_image_surface_get_data ( reinterpret_cast<cairo_surface_t*> ( mCairoSurface ) );
    }
    size_t Workspace::GetWidth() const
    {
        return static_cast<size_t> ( cairo_image_surface_get_width ( reinterpret_cast<cairo_surface_t*> ( mCairoSurface ) ) );
    }
    size_t Workspace::GetHeight() const
    {
        return static_cast<size_t> ( cairo_image_surface_get_height ( reinterpret_cast<cairo_surface_t*> ( mCairoSurface ) ) );
    }
    size_t Workspace::GetStride() const
    {
        return static_cast<size_t> ( cairo_image_surface_get_stride ( reinterpret_cast<cairo_surface_t*> ( mCairoSurface ) ) );
    }
}
