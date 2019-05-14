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
#include <cstdint>
#include "aeongui/Platform.h"
namespace AeonGUI
{
    /*! \brief Rectangle class. */
    class Rect
    {
    public:
        /*! \brief Position and Dimensions constructor.
            \param aX X Position.
            \param aY Y Position.
            \param aWidth Rectangle Width.
            \param aHeight Rectangle Height.
        */
        DLL Rect ( int32_t aX = 0, int32_t aY = 0, uint32_t aWidth = 0, uint32_t aHeight = 0 );

        /*! \name Getters */
        //@{
        /*! \brief Get X coordinate.
            \return Value of the rect's X coordinate.
        */
        DLL int32_t GetX() const;

        /*! \brief Get X coordinate.
            \return Value of the rect's X coordinate.
        */
        DLL int32_t GetY() const;

        /// Get Width.
        DLL uint32_t GetWidth() const;

        /// Get Height.
        DLL uint32_t GetHeight() const;

        /*! \brief Get rect position.
            \param x [out] Reference to a variable to hold the X coordinate.
            \param y [out] Reference to a variable to hold the Y coordinate.
        */
        //void GetPosition ( int32_t& x, int32_t& y ) const;

        /*! \brief Get rect Dimensions.
            \param width [out] Reference to a variable to hold the rect's width.
            \param height [out] Reference to a variable to hold the rect's height.
        */
        //void GetDimensions ( int32_t& width, int32_t& height );

        //@}

        /*! \name Setters */
        //@{
        /*! \brief Set X coordinate.
            \param X Value to set the X coordinate of the rect to.
        */
        DLL void SetX ( int32_t X );

        /*! \brief Set Y coordinate.
            \param Y Value to set the Y coordinate of the rect to.
        */
        DLL void SetY ( int32_t Y );

        /// Set Width.
        DLL void SetWidth ( uint32_t width );

        /// Set Height.
        DLL void SetHeight ( uint32_t height );

        //void Set ( int32_t X1, int32_t Y1, int32_t X2, int32_t Y2 );

        /*! \brief Set Rect absolute position.
            Unlike Rect::Move, this function sets the poisition without regard to the current values.
            \param X Position's X coordinate.
            \param Y Position's Y coordinate.
            \sa Rect::Move
        */
        //void SetPosition ( int32_t X, int32_t Y );

        /*! \brief Set Rect relative position from current one.
            Unlike Rect::SetPosition, this function sets the poisition with regard to the current values,
            that is the new position is calculated relative to the current one.
            \param X Position's X coordinate.
            \param Y Position's Y coordinate.
            \sa Rect::SetPosition
        */
        //void Move ( int32_t X, int32_t Y );

        /*! \brief Set the Rect's dimensions (Width and Height).
            \param width Rect width.
            \param height Rect height.
        */
        //void SetDimensions ( int32_t width, int32_t height );

        //@}
        /*! \brief Test to find out if a point32_t (x,y) lays inside the rect perimeter.
            \param x Point32_t's X coordinate.
            \param y Point32_t's Y coordinate.
            \return Wether or not the point32_t x,y lies inside the rect perimeter.
        */
        //bool IsPointInside ( int32_t x, int32_t y );

        /*! \brief Scale Rect.
            \param amount [in] amount of pixels to scale rect in pixels.
        */
        //void Scale ( int32_t amount );

    private:
        int32_t mX;
        int32_t mY;
        uint32_t mWidth;
        uint32_t mHeight;
    };
}
#endif
