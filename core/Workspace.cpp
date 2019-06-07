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
#include <cmath>
#include <cairo.h>
#include "aeongui/Workspace.h"

namespace AeonGUI
{
    Workspace::Workspace ( uint32_t aWidth, uint32_t aHeight ) :
        mCairoSurface{cairo_image_surface_create ( CAIRO_FORMAT_ARGB32, aWidth, aHeight ) },
        mCairoContext{cairo_create ( reinterpret_cast<cairo_surface_t*> ( mCairoSurface ) ) }
    {
        //cairo_rotate(reinterpret_cast<cairo_t*> ( mCairoContext ),45.0*(M_PI/180.0));
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
        //cairo_rotate(reinterpret_cast<cairo_t*> ( mCairoContext ),45.0*(M_PI/180.0));
        Draw();
    }

    const uint8_t* Workspace::GetData() const
    {
        return cairo_image_surface_get_data ( reinterpret_cast<cairo_surface_t*> ( mCairoSurface ) );
    }

    void Workspace::Draw() const
    {
#if 0
        cairo_select_font_face ( reinterpret_cast<cairo_t*> ( mCairoContext ), "serif", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD );
        cairo_set_font_size ( reinterpret_cast<cairo_t*> ( mCairoContext ), 32.0 );
        cairo_set_source_rgb ( reinterpret_cast<cairo_t*> ( mCairoContext ), 0.0, 0.0, 1.0 );
        cairo_move_to ( reinterpret_cast<cairo_t*> ( mCairoContext ), 10.0, 50.0 );
        cairo_show_text ( reinterpret_cast<cairo_t*> ( mCairoContext ), "Hello, world" );
        cairo_set_source_rgb ( reinterpret_cast<cairo_t*> ( mCairoContext ), 1.0, 1.0, 1.0 );
#endif
        cairo_save ( reinterpret_cast<cairo_t*> ( mCairoContext ) );
        //cairo_set_source_rgba (reinterpret_cast<cairo_t*> ( mCairoContext ), r, g, b, a);
        cairo_set_operator ( reinterpret_cast<cairo_t*> ( mCairoContext ), CAIRO_OPERATOR_CLEAR );
        cairo_paint ( reinterpret_cast<cairo_t*> ( mCairoContext ) );
        cairo_restore ( reinterpret_cast<cairo_t*> ( mCairoContext ) );
        TraverseDepthFirstPreOrder ( [this] ( const Widget & widget )
        {
            const auto matrix = widget.GetGlobalTransform().GetMatrix();
            cairo_matrix_t transform
            {
                matrix[0], matrix[1],
                matrix[2], matrix[3],
                matrix[4], matrix[5],
            };
            cairo_set_matrix ( reinterpret_cast<cairo_t*> ( mCairoContext ), &transform );
            widget.Draw ( mCairoContext );
        } );
#if 0
        const auto& aabb = /* i->GetTransform() **/ i->GetAABB();
        const auto matrix = i->GetTransform().GetMatrix();
        cairo_matrix_t transform
        {
            matrix[0], matrix[1],
            matrix[2], matrix[3],
            matrix[4], matrix[5],
        };
        //cairo_get_matrix(reinterpret_cast<cairo_t*> ( mCairoContext ),&transform);
        cairo_set_matrix ( reinterpret_cast<cairo_t*> ( mCairoContext ), &transform );
        cairo_rectangle ( reinterpret_cast<cairo_t*> ( mCairoContext ),
                          aabb.GetCenter() [0] - aabb.GetRadii() [0],
                          aabb.GetCenter() [1] - aabb.GetRadii() [1],
                          aabb.GetRadii() [0] * 2,
                          aabb.GetRadii() [1] * 2 );
        cairo_fill ( reinterpret_cast<cairo_t*> ( mCairoContext ) );
#endif
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
