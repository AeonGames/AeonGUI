#ifndef AEONGUI_SCROLLBAR_H
#define AEONGUI_SCROLLBAR_H
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

#include "Button.h"

namespace AeonGUI
{
    /*! \brief Scroll bar control. */
    class ScrollBar : public Widget, public MouseListener
    {
    public:
        enum Orientation
        {
            VERTICAL,
            HORIZONTAL
        };
        ScrollBar ( Orientation starting_orientation = VERTICAL );
        virtual ~ScrollBar();

        /*! \brief Set Scrollbar orientation.
            \param new_orientation [in] Orientation to set the Scrollbar to.*/
        void SetOrientation ( Orientation new_orientation );
        /*! \brief Retreieve Scrollbar orientation.
            \return The Scrollbar orientation.*/
        Orientation GetOrientation();

        /*! \brief Set Scrollbar value.
            \param new_value [in] value to set the Scrollbar to.*/
        void SetValue ( int32_t new_value );
        /*! \brief Retreieve Scrollbar value.
            \return The Scrollbar value.*/
        int32_t GetValue();

        /*! \brief Set Scrollbar page step.
            \param pageStep [in] new value for the page step.*/
        void SetPageStep ( int32_t pageStep );
        /*! \brief Retreieve Scrollbar page step.
            \return The Scrollbar page step.*/
        int32_t GetPageStep();

        /*! \brief Set Scrollbar single step.
            \param pageStep [in] new value for the single step.*/
        void SetSingleStep ( int32_t singleStep );
        /*! \brief Retreieve Scrollbar single step.
            \return The Scrollbar single step.*/
        int32_t GetSingleStep();

        /*! \brief Set Scrollbar minimum value.
            \param pageStep [in] new minimum value.*/
        void SetMinimum ( int32_t minimum );
        /*! \brief Retreieve Scrollbar minimum value.
            \return The Scrollbar minimum value.*/
        int32_t GetMinimum();

        /*! \brief Set Scrollbar maximum value.
            \param pageStep [in] new maximum value.*/
        void SetMaximum ( int32_t maximum );
        /*! \brief Retreieve Scrollbar maximum value.
            \return The Scrollbar maximum value.*/
        int32_t GetMaximum();

    protected:
        virtual void OnMouseButtonDown ( Widget* widget, uint8_t button, uint32_t X, uint32_t Y );
        virtual void OnMouseButtonUp ( Widget* widget, uint8_t button, uint32_t X, uint32_t Y );
        virtual void OnMouseMove ( Widget* widget, uint32_t X, uint32_t Y ) {};
        virtual void OnMouseClick ( uint8_t button, uint32_t x, uint32_t y );
        virtual void OnMouseClick ( Widget* widget, uint8_t button, uint32_t X, uint32_t Y );
        virtual void OnMouseMove ( uint32_t X, uint32_t Y );
        virtual void OnSize();
        void Update();
    private:
        Button back;
        Button forward;
        Button slider;
        Orientation orientation;
        Image* scrollup;
        Image* scrolluppressed;
        Image* scrolldown;
        Image* scrolldownpressed;
        Image* scrollleft;
        Image* scrollleftpressed;
        Image* scrollright;
        Image* scrollrightpressed;
        bool sliderdrag;
        int32_t slideroffset;
        // User modifiable variables
        int32_t value;
        int32_t pagestep;
        int32_t singlestep;
        int32_t min;
        int32_t max;
    };
}
#endif