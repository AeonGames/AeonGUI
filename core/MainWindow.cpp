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
#include "MainWindow.h"
#include <iostream>

#include "resources/close.h"
#include "resources/maximize.h"
#include "resources/minimize.h"
#include "resources/restore.h"
#include "resources/close_down.h"
#include "resources/maximize_down.h"
#include "resources/minimize_down.h"
#include "resources/restore_down.h"

namespace AeonGUI
{
    MainWindow::MainWindow () :
        xoffset ( 0 ),
        yoffset ( 0 ),
        verticalscroll ( ScrollBar::VERTICAL ),
        horizontalscroll ( ScrollBar::HORIZONTAL )
    {
#if 0
        Font* font = renderer->GetFont();
        if ( font != NULL )
        {
            captionheight = font->GetHeight() + ( padding * 2 );
        }
        else
        {
            captionheight = 16 + ( padding * 2 );
        }
#endif
        padding = 4;
        captionheight = 16 + ( padding * 2 );
        captioncolor.r = 64;
        captioncolor.g = 64;
        captioncolor.b = 255;
        captioncolor.a = 255;
        hascaption = true;
        hasborder = true;
        moving = false;
        drawfilled = true;
        Image* image = NULL;
        close.SetParent ( this );

        image = new Image;
        image->Load ( close_width, close_height, AeonGUI::Image::RGBA, AeonGUI::Image::BYTE, close_data );
        close.SetNormalImage ( image );

        image = new Image;
        image->Load ( close_down_width, close_down_height, AeonGUI::Image::RGBA, AeonGUI::Image::BYTE, close_down_data );
        close.SetPressedImage ( image );
        close.SetDimensions ( close_width, close_height );
        close.SetMouseListener ( this );

        maximize.SetParent ( this );
        image = new Image;
        image->Load ( maximize_width, maximize_height, AeonGUI::Image::RGBA, AeonGUI::Image::BYTE, maximize_data );
        maximize.SetNormalImage ( image );

        image = new Image;
        image->Load ( maximize_down_width, maximize_down_height, AeonGUI::Image::RGBA, AeonGUI::Image::BYTE, maximize_down_data );
        maximize.SetPressedImage ( image );
        maximize.SetDimensions ( maximize_width, maximize_height );
        maximize.SetMouseListener ( this );

        minimize.SetParent ( this );
        image = new Image;
        image->Load ( minimize_width, minimize_height, AeonGUI::Image::RGBA, AeonGUI::Image::BYTE, minimize_data );
        minimize.SetNormalImage ( image );
        image = new Image;
        image->Load ( minimize_down_width, minimize_down_height, AeonGUI::Image::RGBA, AeonGUI::Image::BYTE, minimize_down_data );
        minimize.SetPressedImage ( image );
        minimize.SetDimensions ( minimize_width, minimize_height );
        minimize.SetMouseListener ( this );

        close.SetPosition ( ( rect.GetWidth() - ( 16 + padding ) ), padding );
        maximize.SetPosition ( ( rect.GetWidth() - ( 32 + padding * 2 ) ), padding );
        minimize.SetPosition ( ( rect.GetWidth() - ( 48 + padding * 3 ) ), padding );

        verticalscroll.SetParent ( this );
        verticalscroll.SetDimensions ( 16, rect.GetHeight() - ( bordersize * 2 ) - captionheight - 16 );
        verticalscroll.SetPosition ( rect.GetWidth() - ( 16 + bordersize ), bordersize + captionheight );

        horizontalscroll.SetParent ( this );
        horizontalscroll.SetDimensions ( rect.GetWidth() - ( bordersize * 2 ) - 16, 16 );
        horizontalscroll.SetPosition ( bordersize, rect.GetHeight() - ( 16 + bordersize ) );
    }

    MainWindow::~MainWindow()
    {
        delete close.GetNormalImage();
        delete maximize.GetNormalImage();
        delete minimize.GetNormalImage() ;
        delete close.GetPressedImage();
        delete maximize.GetPressedImage();
        delete minimize.GetPressedImage() ;
    }

    void MainWindow::SetCaption ( std::wstring& newcaption )
    {
        caption = newcaption;
    }

    void MainWindow::SetCaption ( wchar_t* newcaption )
    {
        caption = newcaption;
    }

    void MainWindow::OnRender ( Renderer* renderer )
    {
        Widget::OnRender ( renderer );
        if ( hascaption )
        {
            GetClientRect ( captionrect );
            captionrect.Scale ( -static_cast<int32_t> ( bordersize ) );
            captionrect.SetHeight ( captionheight );
            DrawRect ( renderer, captioncolor, &captionrect );
#if 0
            Rect textrect = captionrect;
            textrect.Scale ( -static_cast<int32_t> ( padding ) );
            DrawRect ( bordercolor, &textrect );
#endif
            DrawString ( renderer,
                         textcolor,
                         captionrect.GetLeft() + padding,
                         captionrect.GetTop() + renderer->GetFont()->GetHeight() + renderer->GetFont()->GetDescender() + padding,
                         caption.c_str() );
        }
    }

    void MainWindow::OnMouseButtonDown ( Widget* widget, uint8_t button, uint32_t X, uint32_t Y )
    {

    }

    void MainWindow::OnMouseButtonUp ( Widget* widget, uint8_t button, uint32_t X, uint32_t Y )
    {

    }

    void MainWindow::OnMouseButtonDown ( uint8_t button, uint32_t X, uint32_t Y )
    {
        int x = X;
        int y = Y;
        ScreenToClientCoords ( x, y );
        if ( captionrect.IsPointInside ( x, y ) )
        {
            std::cout << "Caption Down " << std::dec << static_cast<int> ( button ) << std::endl;
            moving = true;
            xoffset = X - rect.GetLeft();
            yoffset = Y - rect.GetTop();
        }
#if 0
        else
        {
            std::cout << "MainWindow::OnButtonDown " << std::dec << static_cast<int> ( button ) << std::endl;
        }
#endif
    }

    void MainWindow::OnMouseButtonUp ( uint8_t button, uint32_t X, uint32_t Y )
    {
        int x = X;
        int y = Y;
        ScreenToClientCoords ( x, y );
        if ( captionrect.IsPointInside ( x, y ) )
        {
            std::cout << "Caption Up " << std::dec << static_cast<int> ( button ) << std::endl;
            moving = false;
        }
#if 0
        else
        {
            std::cout << "MainWindow::OnButtonUp " << std::dec << static_cast<int> ( button ) << std::endl;
        }
#endif
    }
    void MainWindow::OnMouseClick ( Widget* clicked_widget, uint8_t button, uint32_t x, uint32_t y )
    {
        if ( clicked_widget == &close )
        {
            std::cout << "Close" << std::endl;
        }
        else if ( clicked_widget == &minimize )
        {
            std::cout << "Minimize" << std::endl;
        }
        else if ( clicked_widget == &maximize )
        {
            std::cout << "Maximize" << std::endl;
        }
    }

    void MainWindow::OnMouseMove ( uint32_t X, uint32_t Y )
    {
#if 0
        std::cout << "Mouse Moved " << X << " " << Y << " " << Xrel << " " << Yrel << std::endl;
#endif
        if ( moving )
        {
            rect.SetPosition ( X - xoffset, Y - yoffset );
        }
    }
}
