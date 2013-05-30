#ifndef AEONGUI_WIDGET_H
#define AEONGUI_WIDGET_H
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
#include <cstddef>
#include <climits>
#include <cassert>
#include "Integer.h"
#include "Renderer.h"
#include "Rect.h"
#include "KeyListener.h"
#include "MouseListener.h"
#include <algorithm>
#include <string>

namespace AeonGUI
{
    /*! \brief Base widget class. */
    class Widget
    {
    public:
        Widget () :
            keyListener ( NULL ),
            mouseListener ( NULL ),
            parent ( NULL ),
            next ( NULL ),
            children ( NULL ),
            // this is temporary, should be 0
            rect ( 0, 0, 320, 200 ),
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

        Widget ( int32_t X, int32_t Y, uint32_t width, uint32_t height ) :
            keyListener ( NULL ),
            mouseListener ( NULL ),
            parent ( NULL ),
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

        virtual ~Widget()
        {
            SetParent ( NULL );
        }
        inline void SetKeyListener ( KeyListener* listener )
        {
            // Should Use local event handling functions for handling own events.
            assert ( static_cast<void*> ( listener ) != static_cast<void*> ( this ) );
            keyListener = listener;
        }
        inline void SetMouseListener ( MouseListener* listener )
        {
            // Should Use local event handling functions for handling own events.
            assert ( static_cast<void*> ( listener ) != static_cast<void*> ( this ) );
            mouseListener = listener;
        }

        inline void SetParent ( Widget* newparent )
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
                child = parent->children;
                while ( child != NULL )
                {
                    if ( child->next = NULL )
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

        /*!
            \brief Checks if the widget has input focus.
            \return true if the widget has focus, false if not
         */
        inline bool HasFocus()
        {
            return focusedWidget == this;
        }

        /*!\brief Grabs input focus */
        inline void GetFocus()
        {
            focusedWidget = this;
        }

        /*!\brief Captures mouse input. */
        inline void CaptureMouse()
        {
            GetFocus();
            mouseCaptured = true;
        }

        /*!\brief Releases mouse input. */
        inline void ReleaseMouse()
        {
            mouseCaptured = false;
        }

        /*!
            \name Event emission functions
            \brief These functions generate an event when called.

            These functions should be called on the root of the widget tree
            or the lowest node desired to potentially handle the event.
         */
        /* @{ */
        /*! \brief Pass down the key down event to the currently focused widget.
            \param charcode [in] Unicode charcode for the key.
            \return true if handled, false if not.
        */
        bool KeyDown ( uint32_t charcode );
        /*! \brief Pass down the key up event to the currently focused widget.
            \param charcode [in] Unicode charcode for the key.
            \return true if handled, false if not.
        */
        bool KeyUp ( uint32_t charcode );

        /*! \brief Signal a mouse move event to widget under the mouse cursor.
            \param x [in] absolute x coordinate for the event.
            \param y [in] absolute y coordinate for the event.
        */
        void MouseMove ( uint32_t x, uint32_t y );

        /*! \brief Signal a mouse button down event to widget under the mouse cursor.
            \param x [in] absolute x coordinate for the event.
            \param y [in] absolute y coordinate for the event.*/
        void MouseButtonDown ( uint8_t button, uint32_t x, uint32_t y );

        /*! \brief Signal a mouse button up event to widget under the mouse cursor.
            \param x [in] absolute x coordinate for the event.
            \param y [in] absolute y coordinate for the event.*/
        void MouseButtonUp ( uint8_t button, uint32_t x, uint32_t y );

        /*! \brief Trigger widget tree rendering.
            \param renderer [in] Renderer to use.*/
        void Render ( Renderer* renderer );

        /* @} */
        /*!\name Event handling functions */
        /* @{ */
        /*!\brief Handles own Mouse Move event.
            \param x [in] absolute x coordinate for the event.
            \param y [in] absolute y coordinate for the event.
        */
        virtual void OnMouseMove ( uint32_t x, uint32_t y ) {};
        /*!\brief Handles own Mouse Button Down event.
           \param x [in] absolute x coordinate for the event.
           \param y [in] absolute y coordinate for the event.
        */
        virtual void OnMouseButtonDown ( uint8_t button, uint32_t x, uint32_t y ) {};
        /*!\brief Handles own Mouse Button Up event.
           \param x [in] absolute x coordinate for the event.
           \param y [in] absolute y coordinate for the event.
        */
        virtual void OnMouseButtonUp ( uint8_t button, uint32_t x, uint32_t y ) {};
        /*!\brief Handles own Mouse Click event.
           \param x [in] absolute x coordinate for the event.
           \param y [in] absolute y coordinate for the event.
        */
        virtual void OnMouseClick ( uint8_t button, uint32_t x, uint32_t y ) {};
        /* @} */
        inline int32_t GetX ( )
        {
            return rect.GetX();
        }
        inline void SetX ( int X )
        {
            rect.SetX ( X );
            OnMove();
        }
        inline int32_t GetY ( )
        {
            return rect.GetY();
        }
        inline void SetY ( int Y )
        {
            rect.SetY ( Y );
            OnMove();
        }
        inline void SetPosition ( int X, int Y )
        {
            rect.SetPosition ( X, Y );
            OnMove();
        }
        inline void Move ( int X, int Y )
        {
            rect.Move ( X, Y );
            OnMove();
        }
        inline void SetDimensions ( int width, int height )
        {
            rect.SetDimensions ( width, height );
            OnSize();
        }
        inline void GetRect ( Rect& outrect )
        {
            outrect = rect;
        }
        inline void GetClientRect ( Rect& outrect )
        {
            outrect.Set ( 0, 0, rect.GetWidth(), rect.GetHeight() );
        }
        inline void GetScreenRect ( Rect* outrect ) const
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
        inline void ClientToScreenRect ( Rect* inoutrect ) const
        {
            int x = inoutrect->GetLeft();
            int y = inoutrect->GetTop();
            ClientToScreenCoords ( x, y );
            inoutrect->Set ( x, y, x + inoutrect->GetWidth(), y + inoutrect->GetHeight() );
        }

        inline void ClientToScreenCoords ( int& x, int& y ) const
        {
            Widget* current_parent = const_cast<Widget*> ( this );
            while ( current_parent != NULL )
            {
                x += current_parent->rect.GetLeft();
                y += current_parent->rect.GetTop();
                current_parent = current_parent->parent;
            }
        }
        inline void ScreenToClientRect ( Rect* inoutrect ) const
        {
            int x = inoutrect->GetLeft();
            int y = inoutrect->GetTop();
            ScreenToClientCoords ( x, y );
            inoutrect->Set ( x, y, x + inoutrect->GetWidth(), y + inoutrect->GetHeight() );
        }

        inline void ScreenToClientCoords ( int32_t& x, int32_t& y ) const
        {
            Widget* current_parent = const_cast<Widget*> ( this );
            while ( current_parent != NULL )
            {
                x -= current_parent->rect.GetLeft();
                y -= current_parent->rect.GetTop();
                current_parent = current_parent->parent;
            }
        }
        /*! \brief Determines if a point (x,y coordinate) is inside a widget rect.
            \param x [in] x coordinate in screen space.
            \param y [in] y coordinate in screen space.
            \return true if the point is inside the rect, false if not.
        */
        inline bool IsPointInside ( int x, int y )
        {
            Rect screen_rect;
            GetScreenRect ( &screen_rect );
            return screen_rect.IsPointInside ( x, y );
        }
        /*! \name Primitive Drawing Functions */
        /* @{ */
        /*! \brief Draws a Rect in widget space.
            \param color [in] Pointer to the color to use for drawing.
            \param rect [in] Pointer to the rect.
        */
        void DrawRect ( Renderer* renderer, Color color, const Rect* draw_rect );
        /*! \brief Draws a Rect outline in widget space.
            \param color [in] Pointer to the color to use for drawing.
            \param rect [in] Pointer to the rect.
        */
        void DrawRectOutline ( Renderer* renderer, Color color, const Rect* draw_rect );
        /*! \brief Draw an Image in widget space.
            \param color [in] Pointer to the color to use for drawing.
            \param points [in] Pointer to a buffer containing x,y point coordinates for the polygon.
            \param pointcount [in] Number of points contained in the points parameter.
            \sa NewImage,DeleteImage
        */
        void DrawImage ( Renderer* renderer, Color color, int32_t x, int32_t y, Image* image );
        /*! \brief Draw text in widget space.
            \param color [in] Pointer to the color to use for drawing.
            \param x [in] x coordinate.
            \param y [in] y coordinate.
            \param text [in] text to render.
            \sa NewFont,DeleteFont.
        */
        void DrawString ( Renderer* renderer, Color color, int32_t x, int32_t y, const wchar_t* text );
        /* @} */

        // From Window: ------------------------------------------------
        inline void Hide ( bool hide )
        {
            hidden =  hide;
        }
        inline void HasBorder ( bool drawborder )
        {
            hasborder = drawborder;
        }
        inline void DrawFilled ( bool isfilled )
        {
            drawfilled = isfilled;
        }
        inline void SetBorderSize ( uint32_t newsize )
        {
            bordersize = newsize;
        }
        inline void SetBackgroundColor ( uint8_t R, uint8_t G, uint8_t B, uint8_t A )
        {
            backgroundcolor.r = R;
            backgroundcolor.g = G;
            backgroundcolor.b = B;
            backgroundcolor.a = A;
        }
        inline void SetBorderColor ( uint8_t R, uint8_t G, uint8_t B, uint8_t A )
        {
            bordercolor.r = R;
            bordercolor.g = G;
            bordercolor.b = B;
            bordercolor.a = A;
        }

    protected:

        virtual void OnRender ( Renderer* renderer );

        virtual void OnSize() {}

        virtual void OnMove() {}

        KeyListener* keyListener;
        MouseListener* mouseListener;
        Widget* parent;
        Widget* next;
        Widget* children;
        Rect rect;
        // From Window:
        Rect clientrect;
        Color backgroundcolor;
        Color textcolor;
        Color bordercolor;
        uint32_t bordersize;
        bool wantsupdate;
        bool hasborder;
        bool hidden;
        bool drawfilled;
        static Widget* focusedWidget;
        static bool mouseCaptured;
    };
}
#endif
