/*
Copyright (C) 2010-2012,2019 Rodrigo Jose Hernandez Cordoba

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
#ifndef AEONGUI_RECT_H
#define AEONGUI_RECT_H
#include "Integer.h"
#include "aeongui/Platform.h"
namespace AeonGUI
{
    /*! \brief Rectangle class. */
    class DLL Rect
    {
    public:
        Rect();

        /*! \brief Dimensions constructor.
            \param X1 Left coordinate for the rect.
            \param Y1 Top coordinate for the rect.
            \param X2 Right coordinate for the rect.
            \param Y2 Bottom coordinate for the rect.
        */
        Rect ( int32_t X1, int32_t Y1, int32_t X2, int32_t Y2 );

        /*! \name Getters */
        //@{
        /// Get Width.
        int32_t GetWidth() const;

        /// Get Height.
        int32_t GetHeight() const;

        /// Get left bound.
        int32_t GetLeft() const;

        /// Get top bound.
        int32_t GetTop() const;

        /// Get right bound.
        int32_t GetRight() const;

        /// Get bottom bound.
        int32_t GetBottom() const;

        /*! \brief Get rect position.
            \param x [out] Reference to a variable to hold the X coordinate.
            \param y [out] Reference to a variable to hold the Y coordinate.
        */
        void GetPosition ( int32_t& x, int32_t& y ) const;

        /*! \brief Get rect Dimensions.
            \param width [out] Reference to a variable to hold the rect's width.
            \param height [out] Reference to a variable to hold the rect's height.
        */
        void GetDimensions ( int32_t& width, int32_t& height );

        /*! \brief Get X coordinate.
            \return Value of the rect's X coordinate.
        */
        int32_t GetX();

        /*! \brief Get X coordinate.
            \return Value of the rect's X coordinate.
        */
        int32_t GetY();
        //@}

        /*! \name Setters */
        //@{
        /// Set Width.
        void SetW ( int32_t width );

        /// Set Height.
        void SetHeight ( int32_t height );

        /*! \brief Set X coordinate.
            \param X Value to set the X coordinate of the rect to.
        */
        void SetX ( int32_t X );

        /*! \brief Set Y coordinate.
            \param Y Value to set the Y coordinate of the rect to.
        */
        void SetY ( int32_t Y );

        /// Set left bound.
        void SetLeft ( int32_t newleft );

        /// Set top bound.
        void SetTop ( int32_t newtop );

        /// Set right bound.
        void SetRight ( int32_t newright );

        /// Set bottom bound.
        void SetBottom ( int32_t newbottom );

        void Set ( int32_t X1, int32_t Y1, int32_t X2, int32_t Y2 );

        /*! \brief Set Rect absolute position.
            Unlike Rect::Move, this function sets the poisition without regard to the current values.
            \param X Position's X coordinate.
            \param Y Position's Y coordinate.
            \sa Rect::Move
        */
        void SetPosition ( int32_t X, int32_t Y );

        /*! \brief Set Rect relative position from current one.
            Unlike Rect::SetPosition, this function sets the poisition with regard to the current values,
            that is the new position is calculated relative to the current one.
            \param X Position's X coordinate.
            \param Y Position's Y coordinate.
            \sa Rect::SetPosition
        */
        void Move ( int32_t X, int32_t Y );

        /*! \brief Set the Rect's dimensions (Width and Height).
            \param width Rect width.
            \param height Rect height.
        */
        void SetDimensions ( int32_t width, int32_t height );

        //@}
        /*! \brief Test to find out if a point32_t (x,y) lays inside the rect perimeter.
            \param x Point32_t's X coordinate.
            \param y Point32_t's Y coordinate.
            \return Wether or not the point32_t x,y lies inside the rect perimeter.
        */
        bool IsPointInside ( int32_t x, int32_t y );

        /*! \brief Scale Rect.
            \param amount [in] amount of pixels to scale rect in pixels.
        */
        void Scale ( int32_t amount );

    private:
        /// Left boundary (X Coordinate).
        int left;
        /// Top boundary (Y Coordinate).
        int top;
        /// Right boundary (X Coordinate).
        int right;
        /// Bottom boundary (Y Coordinate).
        int bottom;
    };
}
#endif
