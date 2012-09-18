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
    /*! \brief Rectangle class. */
    class Rect
    {
    public:
        Rect() : left ( 0 ), top ( 0 ), right ( 0 ), bottom ( 0 ) {}
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

        inline void GetPosition ( int& X, int& Y ) const
        {
            X = left;
            Y = top;
        }
        inline void GetDimensions ( int& width, int& height )
        {
            width = right - left;
            height = bottom - top;
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
        /// Set X coordinate.
        inline void SetX ( int X )
        {
            right = X + right - left;
            left = X;
        }
        inline int32_t GetX()
        {
            return left;
        }
        /// Set Y coordinate
        inline void SetY ( int Y )
        {
            bottom = Y + bottom - top;
            top = Y;
        }
        inline int32_t GetY()
        {
            return top;
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
        inline void SetPosition ( int X, int Y )
        {
            right = X + right - left;
            bottom = Y + bottom - top;
            left = X;
            top = Y;
        }
        inline void Move ( int X, int Y )
        {
            left += X;
            top += Y;
            right += X;
            bottom += Y;
        }
        inline void SetDimensions ( int width, int height )
        {
            right = left + width;
            bottom = top + height;
        }
        //@}
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
