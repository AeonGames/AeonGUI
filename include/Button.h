#ifndef AEONGUI_BUTTON_H
#define AEONGUI_BUTTON_H
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
#include "Image.h"
#include <cstring>

namespace AeonGUI
{
    /*! \brief Common push button. */
    class Button : public Widget
    {
    public:
        /// The different states a button can be at.
        enum BUTTONSTATE
        {
            DEFAULT = 0,
            HIGHLIGHTED,
            FOCUSED,
            PRESSED
        };

        Button();

        virtual ~Button();

        /*! \brief Set the image to be used for the default state.
            \param image [in] Pointer to the Image object to be used.
        */
        void SetNormalImage ( Image* image );

        /*! \brief Set the image to be used for the focused state.
            \param image [in] Pointer to the Image object to be used.
        */
        void SetFocusedImage ( Image* image );

        /*! \brief Set the image to be used for the pressed state.
            \param image [in] Pointer to the Image object to be used.
        */
        void SetPressedImage ( Image* image );

        /*! \brief Sets the caption text for the button.
            \param newcaption [in] The caption text to be set.
        */
        void SetCaption ( wchar_t* newcaption );

        /*! \brief Get the pointer to the current default state image.
            \return Image object pointer.
        */
        Image* GetNormalImage();

        /*! \brief Get the pointer to the current focused state image.
            \return Image object pointer.
        */
        Image* GetFocusedImage();

        /*! \brief Get the pointer to the current pressed state image.
            \return Image object pointer.
        */
        Image* GetPressedImage();

        /*! \brief Get the button's caption text.
            \return The button's caption text.
        */
        const wchar_t* GetCaption();

        /** \copydoc Widget::OnMouseButtonDown */
        virtual bool OnMouseButtonDown ( uint8_t button, uint32_t x, uint32_t y, Widget* widget = NULL );
        /** \copydoc Widget::OnMouseButtonUp */
        virtual bool OnMouseButtonUp ( uint8_t button, uint32_t x, uint32_t y, Widget* widget = NULL );
    protected:
        virtual void OnRender ( Renderer* renderer );
        wchar_t* caption; ///< Button Text.
        Image* normal;        ///< Default state image pointer.
        Image* highlighted;   ///< Highlighed state image pointer.
        Image* focused;       ///< Focused state image pointer.
        Image* pressed;       ///< Pressed state image pointer.
        BUTTONSTATE state;    ///< The button's current state
    };
}
#endif
