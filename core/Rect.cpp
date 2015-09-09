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

#include "Rect.h"

namespace AeonGUI
{
    Rect::Rect() : left ( 0 ), top ( 0 ), right ( 0 ), bottom ( 0 ) {}

    Rect::Rect ( int32_t X1, int32_t Y1, int32_t X2, int32_t Y2 ) : left ( X1 ), top ( Y1 ), right ( X2 ), bottom ( Y2 ) {}

    int32_t Rect::GetWidth() const
    {
        return right - left;
    }

    int32_t Rect::GetHeight() const
    {
        return bottom - top;
    }

    int32_t Rect::GetLeft() const
    {
        return left;
    }


    int32_t Rect::GetTop() const
    {
        return top;
    }

    int32_t Rect::GetRight() const
    {
        return right;
    }

    int32_t Rect::GetBottom() const
    {
        return bottom;
    }

    void Rect::GetPosition ( int32_t& x, int32_t& y ) const
    {
        x = left;
        y = top;
    }

    void Rect::GetDimensions ( int32_t& width, int32_t& height )
    {
        width = right - left;
        height = bottom - top;
    }

    int32_t Rect::GetX()
    {
        return left;
    }

    int32_t Rect::GetY()
    {
        return top;
    }

    void Rect::SetW ( int32_t width )
    {
        right = left + width;
    }

    void Rect::SetHeight ( int32_t height )
    {
        bottom = top + height;
    }

    void Rect::SetX ( int32_t X )
    {
        right = X + right - left;
        left = X;
    }

    void Rect::SetY ( int32_t Y )
    {
        bottom = Y + bottom - top;
        top = Y;
    }

    void Rect::SetLeft ( int32_t newleft )
    {
        left = newleft;
    }

    void Rect::SetTop ( int32_t newtop )
    {
        top = newtop;
    }

    void Rect::SetRight ( int32_t newright )
    {
        right = newright;
    }

    void Rect::SetBottom ( int32_t newbottom )
    {
        bottom = newbottom;
    }

    void Rect::Set ( int32_t X1, int32_t Y1, int32_t X2, int32_t Y2 )
    {
        left = X1;
        top = Y1;
        right = X2;
        bottom = Y2;
    }

    void Rect::SetPosition ( int32_t X, int32_t Y )
    {
        right = X + right - left;
        bottom = Y + bottom - top;
        left = X;
        top = Y;
    }

    void Rect::Move ( int32_t X, int32_t Y )
    {
        left += X;
        top += Y;
        right += X;
        bottom += Y;
    }

    void Rect::SetDimensions ( int32_t width, int32_t height )
    {
        right = left + width;
        bottom = top + height;
    }

    bool Rect::IsPointInside ( int32_t x, int32_t y )
    {
        if ( ( x < left ) || ( y < top ) || ( x > right ) || ( y > bottom ) )
        {
            return false;
        }
        return true;
    }

    void Rect::Scale ( int32_t amount )
    {
        left = left - amount;
        top = top - amount;
        right = right + amount;
        bottom = bottom + amount;
    }
}