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
#include <algorithm>
#include <cairo.h>
#include "aeongui/Workspace.h"

namespace AeonGUI
{
    Workspace::Workspace ( uint32_t aWidth, uint32_t aHeight ) :
        mCairoSurface{cairo_image_surface_create ( CAIRO_FORMAT_ARGB32, aWidth, aHeight ) },
        mCairoContext{cairo_create ( reinterpret_cast<cairo_surface_t*> ( mCairoSurface ) ) }
    {
        Draw();
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
        Draw();
    }

    const uint8_t* Workspace::GetData() const
    {
        return cairo_image_surface_get_data ( reinterpret_cast<cairo_surface_t*> ( mCairoSurface ) );
    }

    void Workspace::Draw() const
    {
        cairo_select_font_face ( reinterpret_cast<cairo_t*> ( mCairoContext ), "serif", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD );
        cairo_set_font_size ( reinterpret_cast<cairo_t*> ( mCairoContext ), 32.0 );
        cairo_set_source_rgb ( reinterpret_cast<cairo_t*> ( mCairoContext ), 0.0, 0.0, 1.0 );
        cairo_move_to ( reinterpret_cast<cairo_t*> ( mCairoContext ), 10.0, 50.0 );
        cairo_show_text ( reinterpret_cast<cairo_t*> ( mCairoContext ), "Hello, world" );
        cairo_set_source_rgb ( reinterpret_cast<cairo_t*> ( mCairoContext ), 1.0, 1.0, 1.0 );
        for ( auto& i : mChildren )
        {
            const auto& rect = i->GetRect();
            cairo_rectangle ( reinterpret_cast<cairo_t*> ( mCairoContext ), rect.GetX(), rect.GetY(), rect.GetWidth(), rect.GetHeight() );
            cairo_fill ( reinterpret_cast<cairo_t*> ( mCairoContext ) );
        }
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

    Widget* Workspace::AddWidget ( std::unique_ptr<Widget> aWidget )
    {
        return mChildren.emplace_back ( std::move ( aWidget ) ).get();
    }

    std::unique_ptr<Widget> Workspace::RemoveWidget ( const Widget* aWidget )
    {
        std::unique_ptr<Widget> result{};
        auto i = std::find_if ( mChildren.begin(), mChildren.end(), [aWidget] ( const std::unique_ptr<Widget>& widget )
        {
            return aWidget == widget.get();
        } );
        if ( i != mChildren.end() )
        {
            result = std::move ( *i );
            mChildren.erase ( std::remove ( i, mChildren.end(), *i ), mChildren.end() );
        }
        return result;
    }

    void Workspace::TraverseDepthFirstPreOrder ( const std::function<void ( Widget& ) >& aAction )
    {
        for ( auto & mRootWidget : mChildren )
        {
            mRootWidget->TraverseDepthFirstPreOrder ( aAction );
        }
    }

    void Workspace::TraverseDepthFirstPreOrder ( const std::function<void ( const Widget& ) >& aAction ) const
    {
        for ( const auto& mRootWidget : mChildren )
        {
            static_cast<const Widget*> ( mRootWidget.get() )->TraverseDepthFirstPreOrder ( aAction );
        }
    }

    void Workspace::TraverseDepthFirstPostOrder ( const std::function<void ( Widget& ) >& aAction )
    {
        for ( auto & mRootWidget : mChildren )
        {
            mRootWidget->TraverseDepthFirstPostOrder ( aAction );
        }
    }

    void Workspace::TraverseDepthFirstPostOrder ( const std::function<void ( const Widget& ) >& aAction ) const
    {
        for ( const auto& mRootWidget : mChildren )
        {
            static_cast<const Widget*> ( mRootWidget.get() )->TraverseDepthFirstPostOrder ( aAction );
        }
    }
}
