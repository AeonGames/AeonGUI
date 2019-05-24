/******************************************************************************
Copyright (C) 2010-2013,2019 Rodrigo Hernandez Cordoba

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
#include "aeongui/Widget.h"

namespace AeonGUI
{
    Widget::Widget ( const Transform& aTransform, const AABB& aAABB ) :
        mTransform{aTransform}, mAABB{aAABB}
    {}

    const Transform& Widget::GetTransform() const
    {
        return mTransform;
    }

    const AABB& Widget::GetAABB() const
    {
        return mAABB;
    }

    /*  This is ugly, but it is only way to use the same code for the const and the non const version
        without having to add template or friend members to the class declaration. */
#define TraverseDepthFirstPreOrder(...) \
    void Widget::TraverseDepthFirstPreOrder ( const std::function<void ( __VA_ARGS__ Widget& ) >& aAction ) __VA_ARGS__ \
    {\
        /** @todo (EC++ Item 3) This code is the same as the constant overload,\
        but can't easily be implemented in terms of that because of aAction's widget parameter\
        need to also be const.\
        */\
        auto widget{this};\
        aAction ( *widget );\
        auto parent = mParent;\
        while ( widget != parent )\
        {\
            if ( widget->mIterator < widget->mChildren.size() )\
            {\
                auto prev = widget;\
                widget = widget->mChildren[widget->mIterator].get();\
                aAction ( *widget );\
                prev->mIterator++;\
            }\
            else\
            {\
                widget->mIterator = 0; /* Reset counter for next traversal.*/\
                widget = widget->mParent;\
            }\
        }\
    }

    TraverseDepthFirstPreOrder ( const )
    TraverseDepthFirstPreOrder( )
#undef TraverseDepthFirstPreOrder

#define TraverseDepthFirstPostOrder(...) \
    void Widget::TraverseDepthFirstPostOrder ( const std::function<void ( __VA_ARGS__ Widget& ) >& aAction ) __VA_ARGS__ \
    { \
        /* \
        This code implements a similar solution to this stackoverflow answer: \
        http://stackoverflow.com/questions/5987867/traversing-a-n-ary-tree-without-using-recurrsion/5988138#5988138 \
        */ \
        auto node = this; \
        auto parent = mParent; \
        while ( node != parent ) \
        { \
            if ( node->mIterator < node->mChildren.size() ) \
            { \
                auto prev = node; \
                node = node->mChildren[node->mIterator].get(); \
                ++prev->mIterator; \
            } \
            else \
            { \
                aAction ( *node ); \
                node->mIterator = 0; /* Reset counter for next traversal. */ \
                node = node->mParent; \
            } \
        } \
    }

    TraverseDepthFirstPostOrder ( const )
    TraverseDepthFirstPostOrder( )
#undef TraverseDepthFirstPostOrder


#if 0
    Widget* Widget::focusedWidget = NULL;
    bool Widget::mouseCaptured = false;

    Widget::Widget () :
        keyListener ( NULL ),
        parent ( NULL ),
        next ( NULL ),
        children ( NULL ),
        // this is temporary, should be 0
        rect ( 0, 0, 320, 200 ),
        // Window members
        backgroundcolor ( 0xffffffff ),
        textcolor ( 0xffffffff ),
        bordercolor ( 255u, 128u, 128u, 128u ),
        bordersize ( 2 ),
        wantsupdate ( true ),
        hasborder ( false ),
        hidden ( false ),
        drawfilled ( false )
    {}

    Widget::Widget ( int32_t X, int32_t Y, uint32_t width, uint32_t height ) :
        keyListener ( NULL ),
        parent ( NULL ),
        next ( NULL ),
        children ( NULL ),
        // this is temporary, should be 0
        rect ( X, Y, X + width, Y + height ),
        // Window members
        backgroundcolor ( 0xffffffff ),
        textcolor ( 0xffffffff ),
        bordercolor ( 255, 128, 128, 128 ),
        bordersize ( 2 ),
        wantsupdate ( true ),
        hasborder ( false ),
        hidden ( false ),
        drawfilled ( false )
    {}

    Widget::~Widget()
    {
        SetParent ( NULL );
    }

    bool Widget::OnMouseButtonDown ( uint8_t button, uint32_t x, uint32_t y, Widget* widget )
    {
        return true;
    }

    bool Widget::OnMouseButtonUp ( uint8_t button, uint32_t x, uint32_t y, Widget* widget )
    {
        return true;
    }

    bool Widget::OnMouseClick ( uint8_t button, uint32_t x, uint32_t y, Widget* widget )
    {
        return true;
    }

    void Widget::SetKeyListener ( KeyListener* listener )
    {
        // Should Use local event handling functions for handling own events.
        assert ( static_cast<void*> ( listener ) != static_cast<void*> ( this ) );
        keyListener = listener;
    }

    void Widget::SetParent ( Widget* newparent )
    {
        Widget* child;
        if ( parent != NULL )
        {
            child = parent->children;
            while ( child != NULL )
            {
                if ( child->next == this )
                {
                    child->next = child->next->next;
                    next = NULL;
                    child = NULL;
                }
                else
                {
                    child = child->next;
                }
            }
        }

        parent = newparent;
        if ( parent != NULL )
        {
            if ( parent->children == NULL )
            {
                parent->children = this;
            }
            else
            {
                child = parent->children;
                while ( child != NULL )
                {
                    if ( child->next == NULL )
                    {
                        child->next = this;
                        child = NULL;
                    }
                    else
                    {
                        child = child->next;
                    }
                }
            }
        }
    }

    bool Widget::HasFocus()
    {
        return focusedWidget == this;
    }

    void Widget::GetFocus()
    {
        focusedWidget = this;
    }

    void Widget::CaptureMouse()
    {
        GetFocus();
        mouseCaptured = true;
    }

    void Widget::ReleaseMouse()
    {
        mouseCaptured = false;
    }

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

    void Widget::DrawString ( Renderer* renderer, Color color, int32_t x, int32_t y, const wchar_t* text )
    {
        ClientToScreenCoords ( x, y );
        renderer->DrawString ( color, x, y, text );
    }

    void Widget::MouseButtonDown ( uint8_t button, uint32_t x, uint32_t y )
    {
        Widget* child = children;
        while ( child != NULL )
        {
            if ( child->IsPointInside ( x, y ) )
            {
                child->MouseButtonDown ( button, x, y );
                return;
            }
            child = child->next;
        }
        if ( mouseCaptured )
        {
            focusedWidget->OnMouseButtonDown ( button, x, y, focusedWidget );
        }
        else
        {
            child = this;
            while ( child != NULL && !child->OnMouseButtonDown ( button, x, y, this ) )
            {
                child = child->parent;
            }
        }
        // keep or get focus.
        this->GetFocus();
    }

    void Widget::MouseButtonUp ( uint8_t button, uint32_t x, uint32_t y )
    {
        Widget* child = children;
        while ( child != NULL )
        {
            if ( child->IsPointInside ( x, y ) )
            {
                child->MouseButtonUp ( button, x, y );
                return;
            }
            child = child->next;
        }

        if ( mouseCaptured )
        {
            focusedWidget->OnMouseButtonUp ( button, x, y, this );
        }
        else
        {
            child = this;
            while ( child != NULL && !child->OnMouseButtonUp ( button, x, y, this ) )
            {
                child = child->parent;
            }
        }

        if ( HasFocus() )
        {
            if ( mouseCaptured )
            {
                focusedWidget->OnMouseClick ( button, x, y, this );
            }
            else
            {
                child = this;
                while ( child != NULL && !child->OnMouseClick ( button, x, y, this ) )
                {
                    child = child->parent;
                }
            }
        }
    }

    void Widget::MouseMove ( uint32_t x, uint32_t y )
    {
        /// \todo This currently just broadcasts the mouse movement, need to make it more specific.
        OnMouseMove ( x, y );
        Widget* child = children;
        while ( child != NULL )
        {
            child->MouseMove ( x, y );
            child = child->next;
        }
    }

    bool Widget::KeyDown ( uint32_t charcode )
    {
        Widget* child = children;
        while ( child != NULL )
        {
            if ( NULL == child->next )
            {
                if ( child->KeyDown ( charcode ) )
                {
                    return true;
                }
            }
            child = child->next;
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
        while ( child != NULL )
        {
            if ( NULL == child->next )
            {
                if ( child->KeyUp ( charcode ) )
                {
                    return true;
                }
            }
            child = child->next;
        }
        if ( keyListener != NULL )
        {
            return keyListener->OnKeyUp ( charcode );
        }
        return false;
    }

    int32_t Widget::GetX ( )
    {
        return rect.GetX();
    }

    void Widget::SetX ( int X )
    {
        rect.SetX ( X );
        OnMove();
    }

    int32_t Widget::GetY ( )
    {
        return rect.GetY();
    }

    void Widget::SetY ( int Y )
    {
        rect.SetY ( Y );
        OnMove();
    }

    void Widget::SetPosition ( int X, int Y )
    {
        rect.SetPosition ( X, Y );
        OnMove();
    }

    void Widget::Move ( int X, int Y )
    {
        rect.Move ( X, Y );
        OnMove();
    }

    void Widget::SetDimensions ( int width, int height )
    {
        rect.SetDimensions ( width, height );
        OnSize();
    }

    void Widget::GetRect ( Rect& outrect )
    {
        outrect = rect;
    }

    void Widget::GetClientRect ( Rect& outrect )
    {
        outrect.Set ( 0, 0, rect.GetWidth(), rect.GetHeight() );
    }

    void Widget::GetScreenRect ( Rect* outrect ) const
    {
        Widget* current_parent = this->parent;
        int x = rect.GetLeft();
        int y = rect.GetTop();
        while ( current_parent != NULL )
        {
            x += current_parent->rect.GetLeft();
            y += current_parent->rect.GetTop();
            current_parent = current_parent->parent;
        }
        outrect->Set ( x, y, x + rect.GetWidth(), y + rect.GetHeight() );
    }

    void Widget::ClientToScreenRect ( Rect* inoutrect ) const
    {
        int x = inoutrect->GetLeft();
        int y = inoutrect->GetTop();
        ClientToScreenCoords ( x, y );
        inoutrect->Set ( x, y, x + inoutrect->GetWidth(), y + inoutrect->GetHeight() );
    }

    void Widget::ClientToScreenCoords ( int& x, int& y ) const
    {
        Widget* current_parent = const_cast<Widget*> ( this );
        while ( current_parent != NULL )
        {
            x += current_parent->rect.GetLeft();
            y += current_parent->rect.GetTop();
            current_parent = current_parent->parent;
        }
    }

    void Widget::ScreenToClientRect ( Rect* inoutrect ) const
    {
        int x = inoutrect->GetLeft();
        int y = inoutrect->GetTop();
        ScreenToClientCoords ( x, y );
        inoutrect->Set ( x, y, x + inoutrect->GetWidth(), y + inoutrect->GetHeight() );
    }

    void Widget::ScreenToClientCoords ( int32_t& x, int32_t& y ) const
    {
        Widget* current_parent = const_cast<Widget*> ( this );
        while ( current_parent != NULL )
        {
            x -= current_parent->rect.GetLeft();
            y -= current_parent->rect.GetTop();
            current_parent = current_parent->parent;
        }
    }

    bool Widget::IsPointInside ( int x, int y )
    {
        Rect screen_rect;
        GetScreenRect ( &screen_rect );
        return screen_rect.IsPointInside ( x, y );
    }

    void Widget::Render ( Renderer* renderer )
    {
        if ( !hidden )
        {
            OnRender ( renderer );
            Widget* child = children;
            while ( child != NULL )
            {
                child->Render ( renderer );
                child = child->next;
            }
        }
    }

    void Widget::OnRender ( Renderer* renderer )
    {
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

    void Widget::Hide ( bool hide )
    {
        hidden =  hide;
    }
    void Widget::HasBorder ( bool drawborder )
    {
        hasborder = drawborder;
    }
    void Widget::DrawFilled ( bool isfilled )
    {
        drawfilled = isfilled;
    }
    void Widget::SetBorderSize ( uint32_t newsize )
    {
        bordersize = newsize;
    }
    void Widget::SetBackgroundColor ( uint8_t R, uint8_t G, uint8_t B, uint8_t A )
    {
        backgroundcolor.r = R;
        backgroundcolor.g = G;
        backgroundcolor.b = B;
        backgroundcolor.a = A;
    }
    void Widget::SetBorderColor ( uint8_t R, uint8_t G, uint8_t B, uint8_t A )
    {
        bordercolor.r = R;
        bordercolor.g = G;
        bordercolor.b = B;
        bordercolor.a = A;
    }
#endif
}
