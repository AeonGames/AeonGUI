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
#include "Widget.h"

namespace AeonGUI
{
    Widget* Widget::focusedWidget = NULL;
    bool Widget::mouseCaptured = false;

    void Widget::DrawRect ( Renderer* renderer, Color color, const Rect* draw_rect )
    {
        Rect wrect = *draw_rect;
        ClientToScreenRect ( &wrect );
        renderer->DrawRect ( color, &wrect );
    }

    void Widget::DrawRectOutline ( Renderer* renderer, Color color, const Rect* draw_rect )
    {
        Rect wrect = *draw_rect;
        ClientToScreenRect ( &wrect );
        renderer->DrawRectOutline ( color, &wrect );
    }

    void Widget::DrawImage ( Renderer* renderer, Color color, int32_t x, int32_t y, Image* image )
    {
        Rect world_rect;
        GetScreenRect ( &world_rect );
        renderer->DrawImage ( color, world_rect.GetLeft(), world_rect.GetTop(), image );
    }

    void Widget::DrawString ( Renderer* renderer, Color color, int32_t x, int32_t y, const wchar_t* text )
    {
        ClientToScreenCoords ( x, y );
        renderer->DrawString ( color, x, y, text );
    }

    void Widget::MouseButtonDown ( uint8_t button, uint32_t x, uint32_t y )
    {
        Widget* child = children;
        while(child!=NULL)
        {
            if ( child->IsPointInside ( x, y ) )
            {
                child->MouseButtonDown ( button, x, y );
                return;
            }
            child=child->next;
        }
        Widget* handler = ( mouseCaptured ) ? focusedWidget : this;
        assert ( handler != NULL );
        handler->OnMouseButtonDown ( button, x, y );
        if ( handler->mouseListener != NULL )
        {
            handler->mouseListener->OnMouseButtonDown ( this, button, x, y );
        }
        // keep or get focus.
        handler->GetFocus();
    }

    void Widget::MouseButtonUp ( uint8_t button, uint32_t x, uint32_t y )
    {
        Widget* child = children;
        while(child!=NULL)
        {
            if ( child->IsPointInside ( x, y ) )
            {
                child->MouseButtonUp ( button, x, y );
                return;
            }
            child=child->next;
        }
        Widget* handler = ( mouseCaptured ) ? focusedWidget : this;
        assert ( handler != NULL );
        handler->OnMouseButtonUp ( button, x, y );
        if ( handler->mouseListener != NULL )
        {
            handler->mouseListener->OnMouseButtonUp ( this, button, x, y );
        }
        if ( HasFocus() )
        {
            handler->OnMouseClick ( button, x, y );
            if ( handler->mouseListener != NULL )
            {
                handler->mouseListener->OnMouseClick ( this, button, x, y );
            }
        }
    }

    void Widget::MouseMove ( uint32_t x, uint32_t y )
    {
        /// \todo This currently just broadcasts the mouse movement, need to make it more specific.
        OnMouseMove ( x, y);
        if ( mouseListener != NULL )
        {
            mouseListener->OnMouseMove ( this, x, y);
        }
        Widget* child = children;
        while(child!=NULL)
        {
            child->MouseMove ( x, y);
            child=child->next;
        }
    }

    bool Widget::KeyDown ( uint32_t charcode )
    {
        Widget* child = children;
        while(child!=NULL)
        {
            if(child->next=NULL)
            {
                if ( child->KeyDown ( charcode ) )
                {
                    return true;
                }
            }
            child=child->next;
        }
        if ( keyListener != NULL )
        {
            return keyListener->OnKeyDown ( charcode );
        }
        return false;
    }

    bool Widget::KeyUp ( uint32_t charcode )
    {
        Widget* child = children;
        while(child!=NULL)
        {
            if(child->next=NULL)
            {
                if ( child->KeyUp ( charcode ) )
                {
                    return true;
                }
            }
            child=child->next;
        }
        if ( keyListener != NULL )
        {
            return keyListener->OnKeyUp ( charcode );
        }
        return false;
    }

    void Widget::Render ( Renderer* renderer )
    {
        OnRender ( renderer );
        Widget* child = children;
        while(child!=NULL)
        {
            child->Render ( renderer );
            child = child->next;
        }
    }

    void Widget::OnRender ( Renderer* renderer )
    {
        if ( hidden )
        {
            return;
        }
        Rect local_rect ( 0, 0, rect.GetWidth(), rect.GetHeight() );
        assert ( renderer != NULL );
        if ( drawfilled )
        {
            DrawRect ( renderer, backgroundcolor, &local_rect );
        }
#if 1
        if ( hasborder )
        {
            for ( uint32_t i = 0; i < bordersize; ++i )
            {
                DrawRectOutline ( renderer, bordercolor, &local_rect );
                local_rect.Scale ( -1 );
            }
        }
#endif
    }

}
