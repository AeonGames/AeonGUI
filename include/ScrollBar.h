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
        /// Scroll bar orientation enumerator.
        enum Orientation
        {
            VERTICAL,
            HORIZONTAL
        };
        /*! \brief Orientation initialization constructor.
            \param starting_orientation Orientation to start the scroll bar at.
        */
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
            \param singleStep [in] new value for the single step.*/
        void SetSingleStep ( int32_t singleStep );

        /*! \brief Retreieve Scrollbar single step.
            \return The Scrollbar single step.*/
        int32_t GetSingleStep();

        /*! \brief Set Scrollbar minimum value.
            \param minimum [in] new minimum value.*/
        void SetMinimum ( int32_t minimum );

        /*! \brief Retreieve Scrollbar minimum value.
            \return The Scrollbar minimum value.*/
        int32_t GetMinimum();

        /*! \brief Set Scrollbar maximum value.
            \param maximum [in] new maximum value.*/
        void SetMaximum ( int32_t maximum );

        /*! \brief Retreieve Scrollbar maximum value.
            \return The Scrollbar maximum value.*/
        int32_t GetMaximum();

    protected:
        /*! \copydoc MouseListener::OnMouseButtonDown */
        virtual void OnMouseButtonDown ( Widget* widget, uint8_t button, uint32_t x, uint32_t y );
        /*! \copydoc MouseListener::OnMouseButtonUp */
        virtual void OnMouseButtonUp ( Widget* widget, uint8_t button, uint32_t x, uint32_t y );
        /*! \copydoc MouseListener::OnMouseMove */
        virtual void OnMouseMove ( Widget* widget, uint32_t x, uint32_t y ) {};
        /*! \copydoc Widget::OnMouseClick */
        virtual void OnMouseClick ( uint8_t button, uint32_t x, uint32_t y );
        /*! \copydoc MouseListener::OnMouseClick */
        virtual void OnMouseClick ( Widget* widget, uint8_t button, uint32_t x, uint32_t y );
        /*! \copydoc Widget::OnMouseMove */
        virtual void OnMouseMove ( uint32_t x, uint32_t y );
        /*! \copydoc Widget::OnSize */
        virtual void OnSize();
        /// Update Scroll bar.
        void Update();
    private:
        Button back;                    ///< Back button.
        Button forward;                 ///< Forward button.
        Button slider;                  ///< Slider button.
        Orientation orientation;        ///< Scroll bar orientation.
        Image* scrollup;                ///< Scroll up image.
        Image* scrolluppressed;         ///< Scroll up pressed image.
        Image* scrolldown;              ///< Scroll down image.
        Image* scrolldownpressed;       ///< Scroll down pressed image.
        Image* scrollleft;              ///< Scroll left image.
        Image* scrollleftpressed;       ///< Scroll left pressed image.
        Image* scrollright;             ///< Scroll right image.
        Image* scrollrightpressed;      ///< Scroll right pressed image.
        bool sliderdrag;                ///< Is the slider being dragged?
        int32_t slideroffset;           ///< Slider offset for slider movement.
        // User modifiable variables
        int32_t value;                  ///< Scroll bar value
        int32_t pagestep;               ///< Page step range.
        int32_t singlestep;             ///< Single step range.
        int32_t min;                    ///< Minimum value for the scroll bar.
        int32_t max;                    ///< Maximum value for the scroll bar.
    };
}
#endif