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
        /*! \brief Position and dimensions constructor.
            \param X X screen coordinate to place the widget.
            \param Y Y screen coordinate to place the widget.
            \param width Widget width.
            \param height Widget height.
        */
        Widget ( int32_t X, int32_t Y, uint32_t width, uint32_t height );

        virtual ~Widget();

        /*! \brief Set the listener object to receive keyboard messages for this widget.
            \param listener Pointer to the listener object.
        */
        void SetKeyListener ( KeyListener* listener );

        /*! \brief Set the listener object to receive mouse messages for this widget.
            \param listener Pointer to the listener object.
        */
        void SetMouseListener ( MouseListener* listener );

        /*! \brief Set this widget's parent.
            If the widget is already parented it is removed from the current parent's children list,
            and placed in the new parent's list. All children move with the widget from one parent to the next.
            \param newparent Pointer to the widget to become the new parent for this widget,
                it may be NULL in which case the widget is removed from its current parent (if any) children list
                and its parent is set to NULL.
        */
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
            \param button [in] The pressed button.
            \param x [in] absolute x coordinate for the event.
            \param y [in] absolute y coordinate for the event.*/
        void MouseButtonDown ( uint8_t button, uint32_t x, uint32_t y );

        /*! \brief Signal a mouse button up event to widget under the mouse cursor.
            \param button [in] The pressed button.
            \param x [in] absolute x coordinate for the event.
            \param y [in] absolute y coordinate for the event.*/
        void MouseButtonUp ( uint8_t button, uint32_t x, uint32_t y );

        /* @} */
        /*!\name Event handling functions */
        /* @{ */
        /*!\brief Handles own Mouse Move event.
            \param x [in] absolute x coordinate for the event.
            \param y [in] absolute y coordinate for the event.
        */
        virtual void OnMouseMove ( uint32_t x, uint32_t y ) {};

        /*!\brief Handles own Mouse Button Down event.
           \param button [in] The pressed button.
           \param x [in] absolute x coordinate for the event.
           \param y [in] absolute y coordinate for the event.
        */
        virtual void OnMouseButtonDown ( uint8_t button, uint32_t x, uint32_t y ) {};

        /*!\brief Handles own Mouse Button Up event.
           \param button [in] The pressed button.
           \param x [in] absolute x coordinate for the event.
           \param y [in] absolute y coordinate for the event.
        */
        virtual void OnMouseButtonUp ( uint8_t button, uint32_t x, uint32_t y ) {};
        /*!\brief Handles own Mouse Click event.
           \param button [in] The pressed button.
           \param x [in] absolute x coordinate for the event.
           \param y [in] absolute y coordinate for the event.
        */
        virtual void OnMouseClick ( uint8_t button, uint32_t x, uint32_t y ) {};
        /* @} */


        /*!\name Rect component functions */
        /* @{ */
        /*! \copydoc Rect::GetX() */
        int32_t GetX ( );
        /*! \copydoc Rect::SetX() */
        void SetX ( int X );

        /*! \copydoc Rect::GetY() */
        int32_t GetY ( );

        /*! \copydoc Rect::SetY() */
        void SetY ( int Y );

        /*! \copydoc Rect::SetPosition() */
        void SetPosition ( int X, int Y );

        /*! \copydoc Rect::Move() */
        void Move ( int X, int Y );

        /*! \copydoc Rect::SetDimensions() */
        void SetDimensions ( int width, int height );
        /* @} */

        /*! \brief Retrieve a copy of the widget rect.
            \param outrect [out] Reference to a variable to copy the rect to.
        */
        void GetRect ( Rect& outrect );

        /*! \brief Retrieve a copy of the widget client rect.
            A client rect is the drawable area of a window.
            \param outrect [out] Reference to a variable to copy the rect to.
        */
        void GetClientRect ( Rect& outrect );

        /*! \brief Retrieve a copy of the screen rect.
            A screen rect is a rect that has been mapped directly to the screen,
            X,Y provide the screen coordinates of the widget regardless of its parent.
            \param outrect [out] Reference to a variable to copy the rect to.
        */
        void GetScreenRect ( Rect* outrect ) const;

        /*! \brief Transform a rect from client space to screen space.
            \param inoutrect [in/out] Reference to a rect variable containing the rect to transform,
            uppon return the variable will hold the transformed values.
            \sa Rect::CLientToScreenCoords
            \todo Determine if this function should be static instead.
        */
        void ClientToScreenRect ( Rect* inoutrect ) const;

        /*! \brief Transform coordinates from client space to screen space.
            \param x [in/out] Reference to variable containing the X coordinate to transform,
            uppon return the variable will hold the transformed value.
            \param y [in/out] Reference to variable containing the Y coordinate to transform,
            uppon return the variable will hold the transformed value.
            \sa Rect::CLientToScreenRect
            \todo Determine if this function should be static instead.
        */
        void ClientToScreenCoords ( int& x, int& y ) const;

        /*! \brief Transform a rect from screen space to client space.
            \param inoutrect [in/out] Reference to a rect variable containing the rect to transform,
            uppon return the variable will hold the transformed values.
            \sa Rect::CLientToScreenCoords
            \todo Determine if this function should be static instead.
        */
        void ScreenToClientRect ( Rect* inoutrect ) const;

        /*! \brief Transform coordinates from screen space to client space.
            \param x [in/out] Reference to variable containing the X coordinate to transform,
            uppon return the variable will hold the transformed value.
            \param y [in/out] Reference to variable containing the Y coordinate to transform,
            uppon return the variable will hold the transformed value.
            \sa Rect::CLientToScreenRect
            \todo Determine if this function should be static instead.
        */
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
            \param renderer [in] Pointer to the renderer to handle the drawing.
            \param color [in] Pointer to the color to use for drawing.
            \param draw_rect [in] Pointer to the rect.
        */
        void DrawRect ( Renderer* renderer, Color color, const Rect* draw_rect );
        /*! \brief Draws a Rect outline in widget space.
            \param renderer [in] Pointer to the renderer to handle the drawing.
            \param color [in] Pointer to the color to use for drawing.
            \param draw_rect [in] Pointer to the rect.
        */
        void DrawRectOutline ( Renderer* renderer, Color color, const Rect* draw_rect );

        /*! \brief Draw text in widget space.
            \param renderer [in] Pointer to the renderer to handle the drawing.
            \param color [in] Pointer to the color to use for drawing.
            \param x [in] x coordinate.
            \param y [in] y coordinate.
            \param text [in] text to render.
        */
        void DrawString ( Renderer* renderer, Color color, int32_t x, int32_t y, const wchar_t* text );
        /* @} */

        // From Window: ------------------------------------------------
        /*! \brief Set wether to render the widget or not.
            \param hide If true the widget will be hidden, if false it will be shown.
        */
        void Hide ( bool hide );

        /*! \brief Set wether the widget has a border or not.
            \param drawborder If true the window will have a border, if false it will not.
            \todo Review the wording and effects of this function.
        */
        void HasBorder ( bool drawborder );

        /*! \brief Set wether the widget is drawn with a solid background or not.
            \param isfilled If true the window will be drawn with a solid background, if false it will not.
            \todo Review the wording and effects of this function.
        */
        void DrawFilled ( bool isfilled );

        /*! \brief Set the border width of a widget if any.
            \param newsize New border width.
        */
        void SetBorderSize ( uint32_t newsize );

        /*! \brief Set the widget's background color.
            \param R Red color component value.
            \param G Green color component value.
            \param B Blue color component value.
            \param A Alpha color component value.
        */
        void SetBackgroundColor ( uint8_t R, uint8_t G, uint8_t B, uint8_t A );

        /*! \brief Set the widget's border color.
            \param R Red color component value.
            \param G Green color component value.
            \param B Blue color component value.
            \param A Alpha color component value.
        */
        void SetBorderColor ( uint8_t R, uint8_t G, uint8_t B, uint8_t A );

    protected:
        friend class Renderer;
        /*! \brief Trigger widget tree rendering.
        This function is only accesible from derived classes and renderers,
        to render a widget you must add it to a renderer's widget list
        and then call the renderer's RenderWidgets member function between calls for BeginRender and EndRender.
        \param renderer [in] Renderer to use.*/
        void Render ( Renderer* renderer );

        /*!\brief On Render event handler function.
            Called when the widget needs to be drawn,
            all drawing operations should be done inside this function.
            Override to do custom drawing.
            \param renderer [in] Renderer to use.
        */
        virtual void OnRender ( Renderer* renderer );

        /*!\brief On Size event handler function.
            Called when the widget is resized.
        */
        virtual void OnSize() {}

        /*!\brief On Move event handler function.
            Called when the widget is moved.
        */
        virtual void OnMove() {}

        KeyListener* keyListener;       ///< Pointer to this widget's key listener if any.
        MouseListener* mouseListener;   ///< Pointer to this widget's mouse listener if any.
        Widget* parent;                 ///< Pointer to this widget's parent if any.
        Widget* next;                   ///< Pointer to this widget's sibling if any.
        Widget* children;               ///< Pointer to this widget's first child if any.
        Rect rect;                      ///< The widget's rect.
        // From Window:
        Rect clientrect;                ///< The widget's client rect.
        Color backgroundcolor;          ///< The widget's background color.
        Color textcolor;                ///< The widget's text color.
        Color bordercolor;              ///< The widget's border color.
        uint32_t bordersize;            ///< The widget's border width.
        bool wantsupdate;               ///< Wether the widget needs to be updated or not.
        bool hasborder;                 ///< Wether the widget has a border.
        bool hidden;                    ///< Wether the widget is hidden or not.
        bool drawfilled;                ///< Wether the widget needs to have a background.
        static Widget* focusedWidget;   ///< The currently focused widget.
        static bool mouseCaptured;      ///< Wether the mouse has been captured.
    };
}
#endif
