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
#include "ScrollBar.h"
#include "resources/scrolldown.h"
#include "resources/scrolldownpressed.h"
#include "resources/scrollup.h"
#include "resources/scrolluppressed.h"
#include "resources/scrollleft.h"
#include "resources/scrollleftpressed.h"
#include "resources/scrollright.h"
#include "resources/scrollrightpressed.h"

#include <iostream>

namespace AeonGUI
{
    ScrollBar::ScrollBar ( Orientation starting_orientation ) :
        sliderdrag ( false ),
        slideroffset ( 0 ),
        value ( 0 ),
        pagestep ( 10 ),
        singlestep ( 1 ),
        min ( 0 ),
        max ( 100 )
    {
        backgroundcolor.r = 180;
        backgroundcolor.g = 180;
        backgroundcolor.b = 180;
        backgroundcolor.a = 255;
        drawfilled = true;

        back.SetParent ( this );
        back.SetMouseListener ( this );

        forward.SetParent ( this );
        forward.SetMouseListener ( this );

        scrollup = new Image;
        scrollup->Load ( scrollup_width, scrollup_height, AeonGUI::Image::RGBA, AeonGUI::Image::BYTE, scrollup_data );
        scrolluppressed = new Image;
        scrolluppressed->Load ( scrolluppressed_width, scrolluppressed_height, AeonGUI::Image::RGBA, AeonGUI::Image::BYTE, scrolluppressed_data );
        scrolldown = new Image;
        scrolldown->Load ( scrolldown_width, scrolldown_height, AeonGUI::Image::RGBA, AeonGUI::Image::BYTE, scrolldown_data );
        scrolldownpressed = new Image;
        scrolldownpressed->Load ( scrolldownpressed_width, scrolldownpressed_height, AeonGUI::Image::RGBA, AeonGUI::Image::BYTE, scrolldownpressed_data );

        scrollleft = new Image;
        scrollleft->Load ( scrollleft_width, scrollleft_height, AeonGUI::Image::RGBA, AeonGUI::Image::BYTE, scrollleft_data );
        scrollleftpressed = new Image;
        scrollleftpressed->Load ( scrollleftpressed_width, scrollleftpressed_height, AeonGUI::Image::RGBA, AeonGUI::Image::BYTE, scrollleftpressed_data );
        scrollright = new Image;
        scrollright->Load ( scrollright_width, scrollright_height, AeonGUI::Image::RGBA, AeonGUI::Image::BYTE, scrollright_data );
        scrollrightpressed = new Image;
        scrollrightpressed->Load ( scrollrightpressed_width, scrollrightpressed_height, AeonGUI::Image::RGBA, AeonGUI::Image::BYTE, scrollrightpressed_data );

        slider.SetParent ( this );
        slider.SetMouseListener ( this );

        slider.SetBackgroundColor ( 200, 200, 200, 255 );
        slider.SetBorderColor ( 220, 220, 220, 255 );
        slider.SetBorderSize ( 1 );
        slider.HasBorder ( true );
        slider.DrawFilled ( true );

        SetOrientation ( starting_orientation );
    }
    ScrollBar::~ScrollBar()
    {
        delete scrollup;
        delete scrolluppressed;
        delete scrolldown;
        delete scrolldownpressed;
        delete scrollleft;
        delete scrollleftpressed;
        delete scrollright;
        delete scrollrightpressed;
    }

    void ScrollBar::SetOrientation ( Orientation new_orientation )
    {
        if ( orientation == new_orientation )
        {
#if WIN32
            std::cout << "Warning: ScrollBar::SetOrientation Called with already set orientation." << std::endl;
#else
            std::cout << "\033[33mWarning:\033[0m ScrollBar::SetOrientation Called with already set orientation." << std::endl;
#endif
            return;
        }
        orientation = new_orientation;
        switch ( orientation )
        {
        case VERTICAL:
            back.SetNormalImage ( scrollup );
            back.SetPressedImage ( scrolluppressed );
            back.SetDimensions ( scrollup->GetWidth(), scrollup->GetHeight() );
            forward.SetNormalImage ( scrolldown );
            forward.SetPressedImage ( scrolldownpressed );
            forward.SetDimensions ( scrolldown->GetWidth(), scrolldown->GetHeight() );
            break;
        case HORIZONTAL:
            back.SetNormalImage ( scrollleft );
            back.SetPressedImage ( scrollleftpressed );
            back.SetDimensions ( scrollleft->GetWidth(), scrollleft->GetHeight() );
            forward.SetNormalImage ( scrollright );
            forward.SetPressedImage ( scrollrightpressed );
            forward.SetDimensions ( scrollright->GetWidth(), scrollright->GetHeight() );
            break;
        }
        Update();
    }

    ScrollBar::Orientation ScrollBar::GetOrientation()
    {
        return orientation;
    }

    void ScrollBar::SetValue ( int32_t new_value )
    {
        if ( new_value > max - pagestep )
        {
            value = max - pagestep;
        }
        else if ( new_value < min )
        {
            value = min;
        }
        else
        {
            value = new_value;
        }

        int32_t rangelength = ( max - min );
        //int32_t pagecount = rangelength / pagestep;

        int32_t position;
        int32_t pagecontrolsize;

        switch ( orientation )
        {
        case VERTICAL:
            pagecontrolsize = ( rect.GetHeight() - ( scrolldown->GetHeight() * 2 ) );
            position = scrolldown->GetHeight() +
                       ( ( pagecontrolsize * value ) / rangelength );
            slider.SetPosition ( 1, position );
            break;
        case HORIZONTAL:
            pagecontrolsize = ( rect.GetWidth() - ( scrolldown->GetWidth() * 2 ) );
            position = scrolldown->GetWidth() +
                       ( ( pagecontrolsize * value ) / rangelength );
            slider.SetPosition ( position, 1 );
            break;
        }
    }

    int32_t ScrollBar::GetValue()
    {
        return value;
    }

    void ScrollBar::SetPageStep ( int32_t pageStep )
    {
        pagestep = pageStep;
    }

    int32_t ScrollBar::GetPageStep()
    {
        return pagestep;
    }

    void ScrollBar::SetSingleStep ( int32_t singleStep )
    {
        singlestep = singleStep;
    }

    int32_t ScrollBar::GetSingleStep()
    {
        return singlestep;
    }

    void ScrollBar::SetMinimum ( int32_t minimum )
    {
        min = minimum;
    }
    int32_t ScrollBar::GetMinimum()
    {
        return min;
    }

    void ScrollBar::SetMaximum ( int32_t maximum )
    {
        max = maximum;
    }
    int32_t ScrollBar::GetMaximum()
    {
        return max;
    }

    void ScrollBar::OnMouseButtonDown ( Widget* widget, uint8_t button, uint32_t X, uint32_t Y )
    {
        Rect slider_rect;
        if ( widget == &slider )
        {
            slider.GetRect ( slider_rect );
            CaptureMouse();
            sliderdrag = true;
            switch ( orientation )
            {
            case VERTICAL:
                slideroffset = Y - slider_rect.GetTop();
                break;
            case HORIZONTAL:
                slideroffset = X - slider_rect.GetLeft();
                break;
            }
        }
    }

    void ScrollBar::OnMouseButtonUp ( Widget* widget, uint8_t button, uint32_t X, uint32_t Y )
    {
        if ( sliderdrag )
        {
            ReleaseMouse();
            sliderdrag = false;
        }
    }

    void ScrollBar::OnMouseClick ( uint8_t button, uint32_t x, uint32_t y )
    {
        int32_t lx = x;
        int32_t ly = y;
        ScreenToClientCoords ( lx, ly );
        switch ( orientation )
        {
        case VERTICAL:
            if ( ly > slider.GetY() )
            {
                SetValue ( value + pagestep );
            }
            else
            {
                SetValue ( value - pagestep );
            }
            break;
        case HORIZONTAL:
            if ( lx > slider.GetX() )
            {
                SetValue ( value + pagestep );
            }
            else
            {
                SetValue ( value - pagestep );
            }
            break;
        }
    }

    void ScrollBar::OnMouseClick ( Widget* widget, uint8_t button, uint32_t X, uint32_t Y )
    {
        if ( widget == &back )
        {
            SetValue ( value - singlestep );
        }
        else if ( widget == &forward )
        {
            SetValue ( value + singlestep );
        }
    }

    void ScrollBar::OnMouseMove ( uint32_t X, uint32_t Y )
    {
        int16_t move;
        Rect slider_rect;
        Rect back_rect;
        Rect forward_rect;
        int32_t rangelength = ( max - min );
        int32_t pagecontrolsize;
        if ( sliderdrag )
        {
            slider.GetRect ( slider_rect );
            back.GetRect ( back_rect );
            forward.GetRect ( forward_rect );
            switch ( orientation )
            {
            case VERTICAL:
                pagecontrolsize = ( rect.GetHeight() - ( scrolldown->GetHeight() * 2 ) );
                move = Y - slideroffset;
                if ( move < back_rect.GetBottom() )
                {
                    move = back_rect.GetBottom();
                }
                else if ( ( move + slider_rect.GetHeight() ) > forward_rect.GetTop() )
                {
                    move = forward_rect.GetTop() - slider_rect.GetHeight();
                }
                slider.SetY ( move );
                break;
            case HORIZONTAL:
                pagecontrolsize = ( rect.GetWidth() - ( scrolldown->GetWidth() * 2 ) );
                move = X - slideroffset;
                if ( move < back_rect.GetRight() )
                {
                    move = back_rect.GetRight();
                }
                else if ( ( move + slider_rect.GetWidth() ) > forward_rect.GetLeft() )
                {
                    move = forward_rect.GetLeft() - slider_rect.GetWidth();
                }
                slider.SetX ( move );
                break;
            }
            value = min + ( ( - ( scrolldown->GetHeight() - move ) * rangelength ) / pagecontrolsize );
        }
    }

    void ScrollBar::OnSize()
    {
        Update();
    }

    void ScrollBar::Update()
    {
        assert ( pagestep > 0 );
        assert ( ( value >= min ) && ( value <= max ) );

        int32_t slider_length;
        int32_t rangelength = ( max - min );
        int32_t pagecount = rangelength / pagestep;

        int32_t position;
        int32_t pagecontrolsize;

        if ( pagecount == 0 )
        {
            pagecount = 1;
        }

        back.SetPosition ( 0, 0 );
        switch ( orientation )
        {
        case VERTICAL:
            pagecontrolsize = ( rect.GetHeight() - ( scrolldown->GetHeight() * 2 ) );
            slider_length =
                pagecontrolsize / pagecount;
            slider.SetDimensions ( 14, slider_length );
            position = scrolldown->GetHeight() +
                       ( ( pagecontrolsize * value ) / rangelength );
            slider.SetPosition ( 1, position );
            forward.SetPosition ( 0, rect.GetHeight() - scrolldown->GetHeight() );
            break;
        case HORIZONTAL:
            pagecontrolsize = ( rect.GetWidth() - ( scrolldown->GetWidth() * 2 ) );
            slider_length =
                ( rect.GetWidth() - ( scrolldown->GetWidth() * 2 ) ) / pagecount;
            slider.SetDimensions ( slider_length, 14 );
            position = scrolldown->GetWidth() +
                       ( ( pagecontrolsize * value ) / rangelength );
            slider.SetPosition ( position, 1 );
            forward.SetPosition ( rect.GetWidth() - scrollleft->GetWidth(), 0 );
            break;
        }
    }
}
