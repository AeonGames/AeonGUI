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
        Widget ();

        Widget ( int32_t X, int32_t Y, uint32_t width, uint32_t height );

        virtual ~Widget();

        void SetKeyListener ( KeyListener* listener );

        void SetMouseListener ( MouseListener* listener );

        void SetParent ( Widget* newparent );
        /*!
            \brief Checks if the widget has input focus.
            \return true if the widget has focus, false if not
         */
        bool HasFocus();

        /*!\brief Grabs input focus */
        void GetFocus();

        /*!\brief Captures mouse input. */
        void CaptureMouse();

        /*!\brief Releases mouse input. */
        void ReleaseMouse();

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

        int32_t GetX ( );

        void SetX ( int X );

        int32_t GetY ( );

        void SetY ( int Y );

        void SetPosition ( int X, int Y );

        void Move ( int X, int Y );

        void SetDimensions ( int width, int height );

        void GetRect ( Rect& outrect );

        void GetClientRect ( Rect& outrect );

        void GetScreenRect ( Rect* outrect ) const;

        void ClientToScreenRect ( Rect* inoutrect ) const;

        void ClientToScreenCoords ( int& x, int& y ) const;

        void ScreenToClientRect ( Rect* inoutrect ) const;

        void ScreenToClientCoords ( int32_t& x, int32_t& y ) const;

        /*! \brief Determines if a point (x,y coordinate) is inside a widget rect.
            \param x [in] x coordinate in screen space.
            \param y [in] y coordinate in screen space.
            \return true if the point is inside the rect, false if not.
        */
        bool IsPointInside ( int x, int y );

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
        void Hide ( bool hide );

        void HasBorder ( bool drawborder );

        void DrawFilled ( bool isfilled );

        void SetBorderSize ( uint32_t newsize );

        void SetBackgroundColor ( uint8_t R, uint8_t G, uint8_t B, uint8_t A );

        void SetBorderColor ( uint8_t R, uint8_t G, uint8_t B, uint8_t A );

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
