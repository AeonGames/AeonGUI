/*
Copyright (C) 2010-2012,2019,2025 Rodrigo Jose Hernandez Cordoba

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

#include "aeongui/Rect.hpp"

namespace AeonGUI
{
    Rect::Rect ( int32_t aX, int32_t aY, uint32_t aWidth, uint32_t aHeight ) : mX{aX}, mY{aY}, mWidth{aWidth}, mHeight{aHeight}
    {

    }

    uint32_t Rect::GetWidth() const
    {
        return mWidth;
    }

    uint32_t Rect::GetHeight() const
    {
        return mHeight;
    }

    int32_t Rect::GetX() const
    {
        return mX;
    }

    int32_t Rect::GetY() const
    {
        return mY;
    }

    void Rect::SetWidth ( uint32_t aWidth )
    {
        mWidth = aWidth;
    }

    void Rect::SetHeight ( uint32_t aHeight )
    {
        mHeight = aHeight;
    }

    void Rect::SetX ( int32_t aX )
    {
        mX = aX;
    }

    void Rect::SetY ( int32_t aY )
    {
        mY = aY;
    }
#if 0
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
#endif
}