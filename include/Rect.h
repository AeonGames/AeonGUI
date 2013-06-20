#ifndef AEONGUI_RECT_H
#define AEONGUI_RECT_H
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
namespace AeonGUI
{
    /*! \brief Rectangle class.
        \todo Remove inlines, move implementations to source file.
    */
    class Rect
    {
    public:
        Rect() : left ( 0 ), top ( 0 ), right ( 0 ), bottom ( 0 ) {}
        /*! \brief Dimensions constructor.
            \param X1 Left coordinate for the rect.
            \param Y1 Top coordinate for the rect.
            \param X2 Right coordinate for the rect.
            \param Y2 Bottom coordinate for the rect.
        */
        Rect ( int X1, int Y1, int X2, int Y2 ) : left ( X1 ), top ( Y1 ), right ( X2 ), bottom ( Y2 ) {}
        /*! \name Getters */
        //@{

        /// Get Width.
        inline int GetWidth() const
        {
            return right - left;
        }

        /// Get Height.
        inline int GetHeight() const
        {
            return bottom - top;
        }

        /// Get left bound.
        inline int GetLeft() const
        {
            return left;
        }

        /// Get top bound.
        inline int GetTop() const
        {
            return top;
        }

        /// Get right bound.
        inline int GetRight() const
        {
            return right;
        }

        /// Get bottom bound.
        inline int GetBottom() const
        {
            return bottom;
        }

        /*! \brief Get rect position.
            \param x [out] Reference to a variable to hold the X coordinate.
            \param y [out] Reference to a variable to hold the Y coordinate.
        */
        inline void GetPosition ( int& x, int& y ) const
        {
            x = left;
            y = top;
        }

        /*! \brief Get rect Dimensions.
            \param width [out] Reference to a variable to hold the rect's width.
            \param height [out] Reference to a variable to hold the rect's height.
        */
        inline void GetDimensions ( int& width, int& height )
        {
            width = right - left;
            height = bottom - top;
        }
        /*! \brief Get X coordinate.
            \return Value of the rect's X coordinate.
        */
        inline int32_t GetX()
        {
            return left;
        }
        /*! \brief Get X coordinate.
            \return Value of the rect's X coordinate.
        */
        inline int32_t GetY()
        {
            return top;
        }
        //@}
        /*! \name Setters */
        //@{
        /// Set Width.
        inline void SetW ( int width )
        {
            right = left + width;
        }
        /// Set Height.
        inline void SetHeight ( int height )
        {
            bottom = top + height;
        }
        /*! \brief Set X coordinate.
            \param X Value to set the X coordinate of the rect to.
        */
        inline void SetX ( int X )
        {
            right = X + right - left;
            left = X;
        }

        /*! \brief Set Y coordinate.
            \param Y Value to set the Y coordinate of the rect to.
        */
        inline void SetY ( int Y )
        {
            bottom = Y + bottom - top;
            top = Y;
        }

        /// Set left bound.
        inline void SetLeft ( int newleft )
        {
            left = newleft;
        }

        /// Set top bound.
        inline void SetTop ( int newtop )
        {
            top = newtop;
        }

        /// Set right bound.
        inline void SetRight ( int newright )
        {
            right = newright;
        }

        /// Set bottom bound.
        inline void SetBottom ( int newbottom )
        {
            bottom = newbottom;
        }

        inline void Set ( int X1, int Y1, int X2, int Y2 )
        {
            left = X1;
            top = Y1;
            right = X2;
            bottom = Y2;
        }
        /*! \brief Set Rect absolute position.
            Unlike Rect::Move, this function sets the poisition without regard to the current values.
            \param X Position's X coordinate.
            \param Y Position's Y coordinate.
            \sa Rect::Move
        */
        inline void SetPosition ( int X, int Y )
        {
            right = X + right - left;
            bottom = Y + bottom - top;
            left = X;
            top = Y;
        }
        /*! \brief Set Rect relative position from current one.
            Unlike Rect::SetPosition, this function sets the poisition with regard to the current values,
            that is the new position is calculated relative to the current one.
            \param X Position's X coordinate.
            \param Y Position's Y coordinate.
            \sa Rect::SetPosition
        */
        inline void Move ( int X, int Y )
        {
            left += X;
            top += Y;
            right += X;
            bottom += Y;
        }
        /*! \brief Set the Rect's dimensions (Width and Height).
            \param width Rect width.
            \param height Rect height.
        */
        inline void SetDimensions ( int width, int height )
        {
            right = left + width;
            bottom = top + height;
        }
        //@}
        /*! \brief Test to find out if a point (x,y) lays inside the rect perimeter.
            \param x Point's X coordinate.
            \param y Point's Y coordinate.
            \return Wether or not the point x,y lies inside the rect perimeter.
        */
        inline bool IsPointInside ( int x, int y )
        {
            if ( ( x < left ) || ( y < top ) || ( x > right ) || ( y > bottom ) )
            {
                return false;
            }
            return true;
        }

        /*! \brief Scale Rect.
            \param amount [in] amount of pixels to scale rect in pixels.
        */
        inline void Scale ( int amount )
        {
            left = left - amount;
            top = top - amount;
            right = right + amount;
            bottom = bottom + amount;
        }
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
