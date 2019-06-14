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
#include "aeongui/Workspace.h"

namespace AeonGUI
{
    Workspace::Workspace ( uint32_t aWidth, uint32_t aHeight ) :
        mCanvas{aWidth, aHeight}
    {
        Draw();
    }
    Workspace::~Workspace() = default;

    void Workspace::Resize ( uint32_t aWidth, uint32_t aHeight )
    {
        mCanvas.ResizeViewport ( aWidth, aHeight );
        Draw();
    }

    const uint8_t* Workspace::GetData() const
    {
        return mCanvas.GetPixels();
    }

    void Workspace::Draw()
    {
        mCanvas.Clear();
#if 0
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
#endif
    }

    size_t Workspace::GetWidth() const
    {
        return mCanvas.GetWidth();
    }
    size_t Workspace::GetHeight() const
    {
        return mCanvas.GetHeight();
    }
    size_t Workspace::GetStride() const
    {
        return mCanvas.GetStride();
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
