#ifndef AEONGUI_MAINWINDOW_H
#define AEONGUI_MAINWINDOW_H
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
#include "Widget.h"
#include "Button.h"
#include "MouseListener.h"
#include "ScrollBar.h"

namespace AeonGUI
{
    /*! \brief Top window with borders. */
    class MainWindow : public Widget, public MouseListener
    {
    public:
        MainWindow();
        virtual ~MainWindow();
        inline void SetCaption ( std::wstring& newcaption )
        {
            caption = newcaption;
        }
        inline void SetCaption ( wchar_t* newcaption )
        {
            caption = newcaption;
        }
        virtual void OnRender ( Renderer* renderer );
    protected:
        virtual void OnMouseButtonDown ( Widget* widget, uint8_t button, uint32_t X, uint32_t Y ) {};
        virtual void OnMouseButtonUp ( Widget* widget, uint8_t button, uint32_t X, uint32_t Y ) {};
        virtual void OnMouseClick ( Widget* widget, uint8_t button, uint32_t x, uint32_t y );
        virtual void OnMouseMove ( Widget* widget, uint32_t X, uint32_t Y ) {};

        virtual void OnMouseButtonDown ( uint8_t button, uint32_t X, uint32_t Y );
        virtual void OnMouseButtonUp ( uint8_t button, uint32_t X, uint32_t Y );
        virtual void OnMouseMove ( uint32_t X, uint32_t Y );
        Rect captionrect;
        std::wstring caption;
        uint32_t padding;
        uint32_t captionheight;
        Color captioncolor;
        uint16_t xoffset;
        uint16_t yoffset;
        bool hascaption;
        bool moving;
        Button close;
        Button maximize;
        Button minimize;
        ScrollBar verticalscroll;
        ScrollBar horizontalscroll;
    };
}
#endif
